import os, json, base64, requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import re

def _log(message):
    print(f"[NanoBanana-GeneralAPI] {message}")

# Nano Banana - General REST API (Gemini-compatible) node
# Goal: user provides api_key + base_url (+ model, version, auth_mode)
# Then call :generateContent and extract returned image automatically

def _b64_png_from_tensor(img: torch.Tensor) -> str:
    # Backward-compat helper. Always encodes PNG.
    return _b64_from_tensor(img, "image/png")


def _b64_from_tensor(img: torch.Tensor, mime: str = "image/png") -> str:
    if img is None:
        return None
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        if img.shape[0] in [1, 3, 4]:
            img_np = img.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img.cpu().numpy()
        if img_np.dtype != np.float32:
            img_np = img_np.astype(np.float32)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        pil = Image.fromarray(img_np)
        buf = BytesIO()
        fmt = 'PNG'
        if mime == 'image/jpeg':
            fmt = 'JPEG'
        elif mime == 'image/webp':
            fmt = 'WEBP'
        pil.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode()
    return None

def _auto_auth_headers(base_url: str, api_key: str, auth_mode: str):
    headers = {"Content-Type": "application/json"}
    mode = (auth_mode or "auto").lower()
    if mode == "google_xgoog" or (mode == "auto" and "generativelanguage.googleapis.com" in (base_url or "")):
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def _build_endpoint(base_url: str, model: str, version: str):
    u = (base_url or "").rstrip('/')
    if "/models/" in u and ":generateContent" in u:
        return u

    # Check if base_url already contains a version path
    if u.endswith('/v1') or u.endswith('/v1beta') or u.endswith('/v1alpha'):
        return f"{u}/models/{model}:generateContent"

    ver = (version or "Auto").lower()
    if ver == "auto":
        ver = "v1beta" if "generativelanguage.googleapis.com" in u else "v1"

    return f"{u}/{ver}/models/{model}:generateContent"


def _deep_merge(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst



# Redact big/base64 strings for logging to avoid noisy output
def _redact_for_log(obj, max_len=256):
    def is_base64_like(s: str) -> bool:
        try:
            return bool(re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", s))
        except Exception:
            return False

    def walk(v):
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                if k == "data" and isinstance(val, str) and len(val) > max_len:
                    out[k] = f"[redacted {len(val)} chars]"
                else:
                    out[k] = walk(val)
            return out
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            if len(v) > max_len and is_base64_like(v):
                return f"[redacted {len(v)} chars]"
            if len(v) > 4096:
                return v[:1024] + f"... [truncated, total {len(v)} chars]"
            return v
        return v

    try:
        return walk(obj)
    except Exception:
        return obj

def _download_image(url: str, proxies=None, timeout=120):
    try:
        _log(f"Downloading image: {url}")
        r = requests.get(url, timeout=timeout, proxies=proxies)
        if r.status_code == 200:
            return r.content
        _log(f"Download failed: HTTP {r.status_code}")
    except Exception as e:
        _log(f"Error downloading image: {e}")
    return None

def _extract_first_image(resp_json, strict_native=False, proxies=None, timeout=120):
    # 1) Gemini style: candidates -> content.parts.inlineData/inline_data.image/*
    try:
        cands = resp_json.get("candidates") or []
        for cand in cands:
            parts = (cand.get("content") or {}).get("parts") or []
            for p in parts:
                # 尝试驼峰命名和下划线命名
                data = p.get("inlineData") or p.get("inline_data") or {}
                mt = (data.get("mimeType") or data.get("mime_type") or "")
                if mt.startswith("image/"):
                    b64 = data.get("data")
                    if b64:
                        _log(f"Found inline image: mime={mt}, data_length={len(b64)}")
                        return base64.b64decode(b64)

                # 检查是否有 Markdown 格式的图像链接（非严格原生模式）
                if not strict_native:
                    text = p.get("text", "")
                    if text:
                        _log(f"🔍 检查文本中的图像URL: {text[:200]}")
                        # 匹配 ![...](url) 或直接的 http(s):// 链接
                        md_match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', text)
                        if md_match:
                            url = md_match.group(1)
                            _log(f"✅ 从Markdown格式提取到图像URL: {url}")
                            img_data = _download_image(url, proxies=proxies, timeout=timeout)
                            if img_data:
                                _log(f"✅ 成功下载图像，大小: {len(img_data)} bytes")
                                return img_data
                            else:
                                _log(f"❌ 下载图像失败: {url}")

                        # 尝试匹配纯 URL（扩展支持更多格式）
                        url_match = re.search(r'(https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|webp|bmp))', text, re.IGNORECASE)
                        if url_match:
                            url = url_match.group(1)
                            _log(f"✅ 从文本中提取到图像URL: {url}")
                            img_data = _download_image(url, proxies=proxies, timeout=timeout)
                            if img_data:
                                _log(f"✅ 成功下载图像，大小: {len(img_data)} bytes")
                                return img_data
                            else:
                                _log(f"❌ 下载图像失败: {url}")
    except Exception as e:
        _log(f"Error in Gemini-style image extraction: {e}")
        import traceback
        _log(traceback.format_exc())
        pass

    # 2) OpenAI/DALL·E style
    try:
        d = resp_json.get("data")
        if isinstance(d, list) and d:
            b64 = d[0].get("b64_json")
            if b64:
                return base64.b64decode(b64)
    except Exception as e:
        _log(f"Error in OpenAI-style image extraction: {e}")
        pass

    # 3) Generic fallbacks
    try:
        for k in ["image", "images"]:
            v = resp_json.get(k)
            if isinstance(v, list) and v:
                b64 = v[0].get("base64") or v[0].get("b64")
                if b64:
                    return base64.b64decode(b64)
    except Exception as e:
        _log(f"Error in fallback image extraction: {e}")
        pass

    return None

class NanoBananaGeneralAPINode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "生成一张清晰的香水产品图", "multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com"}),
                "model": ("STRING", {"default": "gemini-3-pro-image-preview"}),
                "version": (["Auto", "v1", "v1alpha", "v1beta"], {"default": "Auto"}),
                "auth_mode": (["auto", "google_xgoog", "bearer"], {"default": "auto"}),
                "response_mode": (["TEXT_AND_IMAGE", "IMAGE_ONLY", "TEXT_ONLY"], {"default": "TEXT_AND_IMAGE"}),
                "aspect_ratio": (["Auto","1:1","16:9","9:16","4:3","3:4","3:2","2:3","5:4","4:5","21:9"], {"default": "Auto"}),
                "image_size": (["Auto","1K","2K","4K"], {"default": "Auto"}),

                # 🔍 Topaz Gigapixel AI放大控制
                "upscale_factor": (["1x (不放大)", "2x", "4x", "6x"], {
                    "default": "1x (不放大)",
                    "tooltip": "使用Topaz Gigapixel AI进行智能放大"
                }),
                "gigapixel_model": (["High Fidelity", "Standard", "Art & CG", "Lines", "Very Compressed", "Low Resolution", "Text & Shapes", "Redefine", "Recover"], {
                    "default": "High Fidelity",
                    "tooltip": "Gigapixel AI放大模型"
                }),

                # 按顺序：temperature -> top_p -> top_k -> max_output_tokens
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 1000}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),

                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "strict_native": ("BOOLEAN", {"default": False}),
                "system_instruction": ("STRING", {"default": "", "multiline": True}),
                "image_mime": (["image/png","image/jpeg","image/webp"], {"default": "image/png"}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 600, "tooltip": "API请求超时时间（秒），默认300秒"}),

                # 🌐 代理设置
                "use_system_proxy": ("BOOLEAN", {"default": True, "tooltip": "True=使用系统代理, False=禁用代理"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "extra_payload_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "call_api"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"

    def call_api(self, prompt, api_key, base_url, model, version, auth_mode,
                 response_mode, aspect_ratio, image_size, upscale_factor, gigapixel_model,
                 temperature, top_p, top_k, max_output_tokens, seed, strict_native,
                 system_instruction, image_mime, timeout, use_system_proxy,
                 image=None, image2=None, image3=None, image4=None, extra_payload_json=""):
        if not (api_key or "").strip():
            return ("错误: 请提供 API Key", torch.zeros(1, 512, 512, 3))
        endpoint = _build_endpoint(base_url, model, version)
        headers = _auto_auth_headers(base_url, api_key.strip(), auth_mode)

        # Build parts: prompt then up to 4 images
        parts = [{"text": prompt}]
        for img_tensor in [image, image2, image3, image4]:
            b64_img = _b64_from_tensor(img_tensor, image_mime or "image/png")
            if b64_img:
                # 使用驼峰命名以兼容原生 Gemini
                parts.append({"inlineData": {"mimeType": image_mime or "image/png", "data": b64_img}})

        # Base payload per Gemini docs
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": float(temperature),
                "topP": float(top_p),
                "topK": int(top_k),
                "maxOutputTokens": int(max_output_tokens),
            },
        }

        # responseModalities
        if response_mode == "IMAGE_ONLY":
            mods = ["IMAGE"]
        elif response_mode == "TEXT_ONLY":
            mods = ["TEXT"]
        else:
            mods = ["TEXT", "IMAGE"]
        # 设置在 generationConfig 下（官方标准位置）
        payload.setdefault("generationConfig", {})["responseModalities"] = mods

        # imageConfig: aspectRatio + imageSize (1K/2K/4K)
        # 只在 generationConfig 下设置（官方标准位置）
        gen_cfg = payload.setdefault("generationConfig", {})

        if aspect_ratio and aspect_ratio != "Auto":
            gen_cfg.setdefault("imageConfig", {})["aspectRatio"] = aspect_ratio
        if image_size and image_size != "Auto":
            val = str(image_size).upper()
            gen_cfg.setdefault("imageConfig", {})["imageSize"] = val

        # seed (0 means no seed)
        try:
            if isinstance(seed, int) and seed > 0:
                payload.setdefault("generationConfig", {})["seed"] = int(seed)
        except Exception:
            pass

        # systemInstruction（官方标准位置：顶层）
        if system_instruction and system_instruction.strip():
            payload["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": system_instruction.strip()}]
            }

        # Merge extra JSON (allows official fields like safetySettings, tools, toolConfig, responseSchema, clientContext, etc.)
        if extra_payload_json and extra_payload_json.strip():
            try:
                user_extra = json.loads(extra_payload_json)
                payload = _deep_merge(payload, user_extra)
            except Exception as e:
                _log(f"extra_payload_json parse error: {e}")

        try:
            _log(f"Request URL: {endpoint}")
            logged_headers = headers.copy()
            if "Authorization" in logged_headers:
                logged_headers["Authorization"] = "Bearer sk-..."
            if "x-goog-api-key" in logged_headers:
                logged_headers["x-goog-api-key"] = "AIzaSy..."
            _log(f"Request Headers: {logged_headers}")
            _log(f"Request Payload: {json.dumps(_redact_for_log(payload), ensure_ascii=False, indent=2)}")

            # 🌐 配置代理
            # use_system_proxy=True: 使用系统代理（requests默认行为）
            # use_system_proxy=False: 显式禁用代理
            import os
            proxies = None if use_system_proxy else {'http': None, 'https': None}

            if not use_system_proxy:
                _log("🚫 Proxy disabled (use_system_proxy=False) - Direct connection")
                _log(f"   Target: {endpoint}")
            else:
                _log("🌐 Using system proxy settings (use_system_proxy=True)")
                # 显示系统代理环境变量（如果有）
                http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
                https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
                if http_proxy or https_proxy:
                    _log(f"   HTTP_PROXY: {http_proxy or 'Not set'}")
                    _log(f"   HTTPS_PROXY: {https_proxy or 'Not set'}")
                else:
                    _log("   No system proxy environment variables detected")

            resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=timeout, proxies=proxies)

            _log(f"Response Status Code: {resp.status_code}")
            resp_json = resp.json() if resp.status_code == 200 else None
            _log(f"Response Body: {json.dumps(_redact_for_log(resp_json or resp.text), ensure_ascii=False, indent=2)}")

            if resp.status_code != 200:
                return (f"HTTP {resp.status_code}: {resp.text}", torch.zeros(1, 512, 512, 3))

            data = resp_json
            img_bytes = _extract_first_image(data, strict_native=strict_native, proxies=proxies, timeout=timeout)
            text = ""
            try:
                # Primary: candidates[].content.parts[].text
                if 'candidates' in data and data['candidates']:
                    for candidate in data['candidates']:
                        if 'content' in candidate and 'parts' in candidate['content']:
                            for part in candidate['content']['parts']:
                                if 'text' in part:
                                    text += part['text']
                # Fallback: response.parts (some proxies/SDKs flatten)
                if not text and 'parts' in data:
                    for part in data.get('parts') or []:
                        if 'text' in part:
                            text += part['text']
            except Exception as e:
                _log(f"Error extracting text: {e}")
                pass

            if img_bytes:
                _log("成功从响应中提取图像数据。")
                try:
                    pil = Image.open(BytesIO(img_bytes))
                    _log(f"Decoded image mode={pil.mode} size={pil.size}")
                    pil = pil.convert("RGB")
                except Exception as e:
                    _log(f"PIL open/convert failed: {e}")
                    pil = Image.open(BytesIO(img_bytes)).convert("RGB")

                # 🔍 Topaz Gigapixel AI智能放大
                if upscale_factor and upscale_factor != "1x (不放大)":
                    try:
                        # 提取放大倍数
                        scale = int(upscale_factor.replace("x", "").strip().split()[0])
                        if scale > 1:
                            _log(f"🔍 使用智能AI放大进行{scale}x放大，模型: {gigapixel_model}")

                            # 导入放大函数
                            try:
                                from .banana_upscale import smart_upscale
                            except ImportError:
                                from banana_upscale import smart_upscale

                            # 计算目标尺寸
                            target_w = pil.width * scale
                            target_h = pil.height * scale

                            # 使用智能放大
                            upscaled_image = smart_upscale(
                                pil,
                                target_w,
                                target_h,
                                gigapixel_model
                            )

                            if upscaled_image:
                                pil = upscaled_image
                                _log(f"✅ 智能AI放大完成: {pil.size}")
                            else:
                                _log("⚠️ 智能AI放大失败，使用原始图像")
                    except Exception as e:
                        _log(f"⚠️ 智能AI放大失败: {e}，使用原始图像")

                arr = np.array(pil)
                img_t = torch.from_numpy(arr).float() / 255.0
                return (text or "(图像已生成)", img_t.unsqueeze(0))

            _log("警告: 未能从响应中提取图像数据。")

            # 🔍 诊断空响应
            error_details = []
            if data.get('candidates'):
                candidate = data['candidates'][0]
                parts = candidate.get('content', {}).get('parts', [])
                finish_reason = candidate.get('finishReason')
                candidates_tokens = data.get('usageMetadata', {}).get('candidatesTokenCount', 0)

                if not parts or len(parts) == 0:
                    error_details.append("=" * 60)
                    error_details.append("❌ API调用成功但未返回任何内容")
                    error_details.append("=" * 60)
                    error_details.append(f"📊 响应状态:")
                    error_details.append(f"   • finishReason: {finish_reason}")
                    error_details.append(f"   • candidatesTokenCount: {candidates_tokens}")
                    error_details.append(f"   • parts: [] (空数组)")

                    # 检查请求配置
                    error_details.append(f"\n🔧 当前配置:")
                    error_details.append(f"   • 模型: {model}")
                    error_details.append(f"   • API地址: {base_url}")
                    error_details.append(f"   • responseModalities: {mods}")
                    error_details.append(f"   • response_mode: {response_mode}")

                    if gen_cfg.get('imageConfig'):
                        error_details.append(f"   • imageConfig: {json.dumps(gen_cfg['imageConfig'])}")

                    error_details.append(f"\n📝 Prompt (前100字符):")
                    error_details.append(f"   {prompt[:100]}...")

                    # 根据不同情况给出建议
                    error_details.append(f"\n" + "=" * 60)
                    error_details.append("💡 可能的原因和解决方案:")
                    error_details.append("=" * 60)

                    if candidates_tokens == 0:
                        error_details.append("\n1️⃣ 模型不支持图像生成")
                        error_details.append(f"   当前模型 '{model}' 可能是:")
                        error_details.append("   • 纯文本模型（只能生成文本）")
                        error_details.append("   • 视觉理解模型（只能理解图像，不能生成）")
                        error_details.append("\n   ✅ 解决方案:")
                        error_details.append("   • 确认该API是否支持图像生成功能")
                        error_details.append("   • 查看API文档，确认正确的模型名称")
                        error_details.append("   • 联系API提供商确认模型能力")

                    if "IMAGE" in mods:
                        error_details.append("\n2️⃣ API端点可能不正确")
                        error_details.append(f"   当前端点: {endpoint}")
                        error_details.append("\n   ✅ 解决方案:")
                        error_details.append("   • 检查API文档，图像生成可能需要不同的端点")
                        error_details.append("   • 例如: /generateImage 而不是 /generateContent")
                        error_details.append("   • 尝试在 extra_payload_json 中添加特殊参数")

                    error_details.append("\n3️⃣ 需要特殊配置")
                    error_details.append("   ✅ 解决方案:")
                    error_details.append("   • 查看API文档中的图像生成示例")
                    error_details.append("   • 可能需要在 extra_payload_json 中添加:")
                    error_details.append('     {"imageGenerationConfig": {...}}')
                    error_details.append("   • 或其他特定的配置参数")

                    error_details.append("\n4️⃣ Prompt格式问题")
                    error_details.append("   ✅ 解决方案:")
                    error_details.append("   • 某些API需要特定的prompt格式")
                    error_details.append("   • 尝试: '生成一张[描述]的图片'")
                    error_details.append("   • 或: 'Generate an image of [description]'")

                    error_details.append("\n" + "=" * 60)
                    error_details.append("📚 建议操作:")
                    error_details.append("=" * 60)
                    error_details.append("1. 查看 API 文档确认图像生成的正确用法")
                    error_details.append("2. 确认模型名称和端点是否正确")
                    error_details.append("3. 查看是否需要特殊的 extra_payload_json 配置")
                    error_details.append("4. 联系 API 提供商获取图像生成示例")

            error_msg = "\n".join(error_details) if error_details else f"错误: 响应中未找到图像。响应全文: {resp.text}"
            return (text or error_msg, torch.zeros(1, 512, 512, 3))

        except requests.exceptions.SSLError as e:
            error_msg = f"SSL连接错误: {e}"
            _log(error_msg)

            # 提供诊断建议
            suggestions = []
            if not use_system_proxy:
                suggestions.append("💡 当前已禁用代理，如果API需要代理访问，请启用 use_system_proxy")
            else:
                suggestions.append("💡 当前使用系统代理，如果代理有问题，可以尝试禁用 use_system_proxy")

            suggestions.append("💡 检查网络连接是否正常")
            suggestions.append(f"💡 确认API地址是否正确: {base_url}")
            suggestions.append("💡 如果使用自签名证书，可能需要配置SSL验证")

            full_msg = f"{error_msg}\n\n诊断建议:\n" + "\n".join(suggestions)
            return (full_msg, torch.zeros(1, 512, 512, 3))

        except requests.exceptions.ProxyError as e:
            error_msg = f"代理连接错误: {e}"
            _log(error_msg)

            suggestions = []
            if use_system_proxy:
                import os
                http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
                https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
                if http_proxy or https_proxy:
                    suggestions.append(f"💡 检测到系统代理: HTTP={http_proxy}, HTTPS={https_proxy}")
                    suggestions.append("💡 请确认代理服务器是否正常运行")
                else:
                    suggestions.append("💡 未检测到系统代理环境变量，但可能通过其他方式配置了代理")
                suggestions.append("💡 尝试禁用 use_system_proxy 测试直连")
            else:
                suggestions.append("💡 当前已禁用代理，但仍然出现代理错误（可能是系统强制代理）")

            full_msg = f"{error_msg}\n\n诊断建议:\n" + "\n".join(suggestions)
            return (full_msg, torch.zeros(1, 512, 512, 3))

        except requests.exceptions.Timeout as e:
            error_msg = f"请求超时 (timeout={timeout}s): {e}"
            _log(error_msg)
            suggestions = [
                f"💡 当前超时设置: {timeout}秒",
                "💡 可以尝试增加 timeout 参数",
                "💡 检查网络连接速度",
            ]
            full_msg = f"{error_msg}\n\n诊断建议:\n" + "\n".join(suggestions)
            return (full_msg, torch.zeros(1, 512, 512, 3))

        except requests.exceptions.ConnectionError as e:
            error_msg = f"网络连接错误: {e}"
            _log(error_msg)

            suggestions = []
            if not use_system_proxy:
                suggestions.append("💡 当前禁用代理，直连失败")
                suggestions.append("💡 如果API需要代理访问，请启用 use_system_proxy")
                suggestions.append("💡 检查防火墙设置")
            else:
                suggestions.append("💡 当前使用系统代理")
                suggestions.append("💡 检查代理服务器是否正常")
                suggestions.append("💡 尝试禁用 use_system_proxy 测试直连")

            suggestions.append(f"💡 确认API地址是否可访问: {base_url}")

            full_msg = f"{error_msg}\n\n诊断建议:\n" + "\n".join(suggestions)
            return (full_msg, torch.zeros(1, 512, 512, 3))

        except Exception as e:
            error_msg = f"请求失败: {e}"
            _log(error_msg)
            import traceback
            _log(traceback.format_exc())
            return (error_msg, torch.zeros(1, 512, 512, 3))

NODE_CLASS_MAPPINGS = {
    "NanoBananaGeneralAPI": NanoBananaGeneralAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaGeneralAPI": "NanoBanana-GeneralAPI",
}

