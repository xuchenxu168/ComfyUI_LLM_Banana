import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import traceback
import tempfile
import wave
import random
import time
from typing import Tuple

try:
    from server import PromptServer
except Exception:
    PromptServer = None

def _log_info(message):
    try:
        print(f"[LLM Agent Assistant][Gemini-Banana] {message}")
    except UnicodeEncodeError:
        print(f"[LLM Prompt][Gemini-Banana] INFO: {repr(message)}")

def _log_warning(message):
    try:
        print(f"[LLM Agent Assistant][Gemini-Banana] WARNING: {message}")
    except UnicodeEncodeError:
        print(f"[LLM Prompt][Gemini-Banana] WARNING: {repr(message)}")

def _log_error(message):
    try:
        print(f"[LLM Agent Assistant][Gemini-Banana] ERROR: {message}")
    except UnicodeEncodeError:
        print(f"[LLM Prompt][Gemini-Banana] ERROR: {repr(message)}")

def smart_retry_delay(attempt, error_code=None):
    """智能重试延迟 - 根据错误类型调整等待时间"""
    base_delay = 2 ** attempt  # 指数退避
    
    if error_code == 429:  # 限流错误
        # 对于429错误，使用更长的等待时间
        rate_limit_delay = 60 + random.uniform(10, 30)  # 60-90秒随机等待
        return max(base_delay, rate_limit_delay)
    elif error_code in [500, 502, 503, 504]:  # 服务器错误
        return base_delay + random.uniform(1, 5)  # 添加随机抖动
    else:
        return base_delay

def resize_image_for_api(image, max_size=2048):
    """调整图像大小以满足API限制"""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        _log_info(f"Image resized to {new_size} for API compatibility")
    return image

def process_image_controls(size: str, quality: str, style: str, custom_size: str = "") -> dict:
    """
    处理图像控制参数，返回标准化的控制配置
    
    Args:
        size: 预设尺寸
        quality: 质量设置
        style: 风格设置
        custom_size: 自定义尺寸
        
    Returns:
        dict: 包含处理后的图像控制参数
    """
    # 处理尺寸
    final_size = custom_size.strip() if custom_size and custom_size.strip() else size
    
    # 验证自定义尺寸格式
    if custom_size and custom_size.strip():
        import re
        size_pattern = r'^\d+x\d+$'
        if not re.match(size_pattern, custom_size.strip()):
            print(f"⚠️ 自定义尺寸格式无效: {custom_size}，使用预设尺寸: {size}")
            final_size = size
    
    # 构建控制配置
    controls = {
        "size": final_size,
        "quality": quality,
        "style": style,
        "is_custom_size": bool(custom_size and custom_size.strip())
    }
    
    return controls


def enhance_prompt_with_controls(prompt: str, controls: dict) -> str:
    """
    使用图像控制参数增强提示词，参考 OpenRouter 的实现方式

    当 style 为 None/空字符串 时，不在提示词中写入风格要求。
    """
    size_line = f"1. 输出尺寸：{controls['size']}"
    quality_line = f"2. 质量：{'高质量' if controls['quality'] == 'hd' else '标准质量'}"

    style_raw = str(controls.get('style', '') or '').strip().lower()
    style_line = ""
    if style_raw not in ("none", ""):
        # 简单映射：vivid->生动风格；natural->自然风格；其他直接显示原值
        if style_raw == "vivid":
            style_text = "生动风格"
        elif style_raw == "natural":
            style_text = "自然风格"
        else:
            style_text = controls.get('style', '')
        style_line = f"3. 风格：{style_text}"

    # 组装要求块
    lines = [size_line, quality_line]
    if style_line:
        lines.append(style_line)
    req_block = "\n".join(lines)

    enhanced_prompt = f"""你是一个专业的图像生成专家。请根据以下要求生成图像：

{prompt}

具体要求：
{req_block}

构图要求：
- 使用平衡的构图，主体与背景比例适当
- 主体应清晰可见且完整展现，占据图像面积的40-60%
- 包含丰富的背景环境和上下文，创造层次感和氛围
- 使用中景拍摄，展现主体在其环境中的状态
- 避免过度特写和过于遥远的拍摄角度
- 确保主体完全在画面边界内且与环境和谐统一

请严格按照上述要求生成图像，确保输出尺寸、质量和构图完全符合要求。不要描述图片，直接生成符合规格的图像。"""
    return enhanced_prompt


def load_gemini_banana_config():
    """Load Gemini Banana configuration file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gemini_Banana_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            _log_info(f"Successfully loaded Gemini Banana config file: {config_path}")
            return config
        else:
            # Use default Gemini config as fallback
            fallback_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gemini_config.json")
            if os.path.exists(fallback_config_path):
                with open(fallback_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                _log_info(f"Using default Gemini config file: {fallback_config_path}")
                return config
            else:
                _log_warning(f"Gemini Banana config file not found: {config_path}")
                return {}
    except Exception as e:
        _log_error(f"Failed to load Gemini Banana config file: {e}")
        return {}

def get_gemini_banana_config():
    """Get Gemini Banana configuration, prioritize config file"""
    config = load_gemini_banana_config()
    return config

def process_input_image(image_tensor):
    """Process input image tensor and convert to PIL Image"""
    try:
        # Handle 4D tensor (batch)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # Take the first image
        
        # Convert CHW to HWC format
        if image_tensor.shape[0] in [1, 3, 4]:  # CHW format
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:  # Already HWC format
            image_np = image_tensor.cpu().numpy()
        
        # Normalize and convert to uint8
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        image_np = (image_np * 255).astype(np.uint8)
        
        # Handle channels
        if len(image_np.shape) == 3:
            if image_np.shape[2] == 1:
                image_np = np.repeat(image_np, 3, axis=2)  # Grayscale to RGB
            elif image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]  # RGBA to RGB
        elif len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)  # Grayscale to RGB
        
        pil_image = Image.fromarray(image_np)
        
        # Resize if too large for API
        pil_image = resize_image_for_api(pil_image)
        
        return pil_image
    except Exception as e:
        _log_error(f"Failed to process input image: {e}")
        return None

def image_to_base64(image, format='JPEG', quality=95):
    """Convert PIL Image to base64 string"""
    try:
        buffer = io.BytesIO()
        
        # Handle alpha channel for JPEG
        if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode in ('RGBA', 'LA'):
                background.paste(image, mask=image.split()[-1])
                image = background
        
        image.save(buffer, format=format, quality=quality)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        _log_error(f"Failed to convert image to base64: {e}")
        return None

def process_generated_image_from_response(response_json):
    """从REST API响应中提取生成的图像"""
    try:
        if "candidates" not in response_json:
            _log_warning("响应中未找到candidates")
            return create_dummy_image()
        
        for candidate in response_json["candidates"]:
            if "content" not in candidate or "parts" not in candidate["content"]:
                continue
                
            for part in candidate["content"]["parts"]:
                # 检查inline_data字段（图像数据）
                inline_data = part.get("inline_data") or part.get("inlineData")
                if inline_data and "data" in inline_data:
                    try:
                        # 解码图片数据
                        image_data = inline_data["data"]
                        image_bytes = base64.b64decode(image_data)
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # 确保RGB格式
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        # 转换为ComfyUI tensor格式
                        img_array = np.array(pil_image)
                        img_tensor = torch.from_numpy(img_array).float() / 255.0
                        if len(img_tensor.shape) == 3:
                            img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
                        
                        _log_info("✅ 成功提取生成的图像")
                        return img_tensor
                    except Exception as e:
                        _log_error(f"解码图片失败: {e}")
        
        # 如果没有找到图像，返回占位符
        _log_warning("未检测到生成的图像，创建占位符")
        return create_dummy_image()
        
    except Exception as e:
        _log_error(f"处理响应失败: {e}")
        return create_dummy_image()

def extract_text_from_response(response_json):
    """从REST API响应中提取文本"""
    try:
        response_text = ""
        
        if "candidates" not in response_json:
            return "No valid response received"
        
        for candidate in response_json["candidates"]:
            if "content" not in candidate or "parts" not in candidate["content"]:
                continue
                
            for part in candidate["content"]["parts"]:
                if "text" in part:
                    response_text += part["text"] + "\n"
        
        return response_text.strip() if response_text.strip() else "Response received but no text content"
        
    except Exception as e:
        _log_error(f"提取文本失败: {e}")
        return "Failed to extract text from response"

def create_dummy_image(width=512, height=512):
    """Create a placeholder image"""
    dummy_array = np.zeros((height, width, 3), dtype=np.uint8)
    dummy_tensor = torch.from_numpy(dummy_array).float() / 255.0
    return dummy_tensor.unsqueeze(0)

def prepare_media_content(image=None, audio=None):
    """Prepare multimedia content for API calls"""
    content_parts = []
    
    if image is not None:
        pil_image = process_input_image(image)
        if pil_image:
            img_base64 = image_to_base64(pil_image, format='PNG')
            if img_base64:
                content_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })
    
    if audio is not None:
        # Process audio data
        try:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = np.array(audio)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Assume sample rate of 16000
                sample_rate = 16000
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())
                
                # Read audio file and encode
                with open(temp_file.name, 'rb') as f:
                    audio_data = base64.b64encode(f.read()).decode()
                
                content_parts.append({
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_data
                    }
                })
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
        except Exception as e:
            _log_error(f"Failed to process audio data: {e}")
    
    return content_parts

def generate_with_rest_api(api_key, model, content_parts, generation_config, max_retries=5, proxy=None, base_url=None):
    """使用REST API的智能重试机制调用"""
    
    # 构建API URL - 支持镜像站
    if base_url and base_url.strip():
        # 移除末尾的斜杠，确保URL格式正确
        base_url = base_url.rstrip('/')
        
        # 如果用户提供的是完整URL，直接使用
        if '/models/' in base_url and ':generateContent' in base_url:
            url = base_url
        # 如果是基础URL，构建完整路径
        elif base_url.endswith('/v1beta') or base_url.endswith('/v1'):
            url = f"{base_url}/models/{model}:generateContent"
        else:
            # 默认添加v1beta路径
            url = f"{base_url}/v1beta/models/{model}:generateContent"
        
        _log_info(f"🔗 使用镜像站: {base_url}")
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        _log_info(f"🌐 使用官方API: generativelanguage.googleapis.com")
    
    # 构建请求数据
    request_data = {
        "contents": [{
            "parts": content_parts
        }],
        "generationConfig": generation_config
    }
    
    # 设置请求头
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key.strip()
    }
    
    # 处理代理设置
    proxies = None
    if proxy and proxy.strip() and "None" not in proxy:
        proxies = {
            "http": proxy.strip(),
            "https": proxy.strip()
        }
        _log_info(f"🔌 使用代理: {proxy.strip()}")
    
    # 设置合理的超时：连接超时10秒，读取超时60秒
    timeout = (10, 60)  # (connect_timeout, read_timeout)
    
    for attempt in range(max_retries):
        try:
            _log_info(f"🌐 REST API调用 ({attempt + 1}/{max_retries}) 模型: {model}")
            
            # 发送请求
            response = requests.post(url, headers=headers, json=request_data, timeout=timeout, proxies=proxies)
            
            # 成功响应
            if response.status_code == 200:
                return response.json()
            
            # 处理错误响应
            else:
                _log_error(f"HTTP状态码: {response.status_code}")
                try:
                    error_detail = response.json()
                    _log_error(f"错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    
                    # 检查是否是配额错误
                    if response.status_code == 429:
                        error_message = error_detail.get("error", {}).get("message", "")
                        if "quota" in error_message.lower():
                            _log_warning("检测到配额限制错误")
                except:
                    _log_error(f"错误文本: {response.text}")
                
                # 如果是最后一次尝试，抛出异常
                if attempt == max_retries - 1:
                    response.raise_for_status()
                
                # 智能等待
                delay = smart_retry_delay(attempt, response.status_code)
                _log_info(f"🔄 等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
                
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            _log_error(f"请求失败: {error_msg}")
            if attempt == max_retries - 1:
                raise e
            else:
                delay = smart_retry_delay(attempt)
                _log_info(f"🔄 等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
    
    raise Exception("所有重试都失败了")

class KenChenLLMGeminiBananaTextToImageBananaNode:
    CATEGORY = "Ken-Chen/LLM Agent Assistant"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generation_text", "generated_image")
    FUNCTION = "generate_image"
    

    
    @classmethod
    def INPUT_TYPES(s):
        config = get_gemini_banana_config()
        default_params = config.get('default_params', {})
        image_settings = config.get('image_settings', {})
        
        # Get image generation models from config, with fallback to core Banana models
        models = config.get('models', {}).get('image_gen_models', [])
        if not models:
            # Fallback to core Banana models if config is empty
            models = [
                "gemini-2.5-flash-image-preview",  # Latest Banana model
                "gemini-2.0-flash-preview-image-generation"
            ]
        
        # Get default model from config, prioritize latest Banana model
        default_model = config.get('default_model', {}).get('image_gen', "gemini-2.5-flash-image-preview")
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        
        # Get image control presets
        size_presets = image_settings.get('size_presets', ["512x512", "768x768", "1024x1024", "1024x1792", "1792x1024"])
        quality_presets = image_settings.get('quality_presets', ["standard", "hd"])
        style_presets = image_settings.get('style_presets', ["vivid", "natural"])
        if "None" not in style_presets:
            style_presets = ["None"] + style_presets
        
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "API密钥（留空时自动从配置文件读取）"
                }),
                "prompt": ("STRING", {"default": "A beautiful landscape with mountains and lake", "multiline": True}),
                "model": (
                    models,
                    {"default": default_model},
                ),
                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),
                "size": (size_presets, {"default": image_settings.get('default_size', "1024x1024")}),
                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),
                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.9), "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": default_params.get('top_k', 40), "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": default_params.get('max_output_tokens', 2048), "min": 0, "max": 32768}),
                "seed": ("INT", {"default": default_params.get('seed', 0), "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                "custom_size": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "自定义尺寸 (如: 1920x1080)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    


    def _push_chat(self, user_prompt: str, response_text: str, unique_id: str):
        if not PromptServer or not unique_id:
            return
        try:
            # 限制提示词长度，避免过长文本导致显示问题
            max_prompt_length = 500
            max_response_length = 1000
            
            # 截断过长的提示词和响应
            display_prompt = user_prompt[:max_prompt_length] + ("..." if len(user_prompt) > max_prompt_length else "")
            display_response = response_text[:max_response_length] + ("..." if len(response_text) > max_response_length else "")
            
            render_spec = {
                "node_id": unique_id,
                "component": "ChatHistoryWidget",
                "props": {
                    "history": json.dumps([
                        {
                            "prompt": display_prompt,
                            "response": display_response,
                            "response_id": str(random.randint(100000, 999999)),
                            "timestamp": time.time(),
                        }
                    ], ensure_ascii=False)
                },
            }
            PromptServer.instance.send_sync("display_component", render_spec)
        except Exception as e:
            print(f"[LLM Agent Assistant] Chat push failed: {e}")
            pass

    def generate_image(
        self,
        api_key,
        prompt,
        model,
        proxy,
        size,
        quality,
        style,
        temperature,
        top_p,
        top_k,
        max_output_tokens,
        seed,
        custom_size: str = "",
        unique_id: str = "",
    ):
        try:
            # 如果用户没有输入API密钥，自动从配置文件获取
            if not api_key or not api_key.strip():
                config = get_gemini_banana_config()
                auto_api_key = config.get('api_key', '')
                if auto_api_key and auto_api_key.strip():
                    api_key = auto_api_key.strip()
                    _log_info(f"🔑 自动使用配置文件中的API密钥: {api_key[:8]}...")
                else:
                    error_msg = "API密钥不能为空，请在配置文件中设置api_key或手动输入"
                    _log_error(error_msg)
                    return (error_msg, create_dummy_image())
            
            # 验证提示词
            if not prompt.strip():
                error_msg = "提示词不能为空"
                _log_error(error_msg)
                return (error_msg, create_dummy_image())
            
            # 处理图像控制参数
            controls = process_image_controls(size, quality, style, custom_size)
            enhanced_prompt = enhance_prompt_with_controls(prompt.strip(), controls)
            
            _log_info(f"🎨 图像控制参数: 尺寸={controls['size']}, 质量={controls['quality']}, 风格={controls['style']}")
            if controls['is_custom_size']:
                _log_info(f"📏 使用自定义尺寸: {controls['size']}")
            
            # 代理处理：使用 proxies 参数，不设置环境变量，避免冲突
            if proxy and proxy.strip() and "None" not in proxy:
                # 使用 proxies 参数，不设置环境变量
                _log_info(f"使用代理: {proxy.strip()}")
            else:
                _log_info("未使用代理")
            
            # 构建生成配置
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"]  # 关键：启用图像生成
            }
            
            # 智能种子控制
            if seed > 0:
                generation_config["seed"] = seed
            
            # 准备内容
            content_parts = [{"text": enhanced_prompt}]
            
            # 使用REST API调用
            _log_info(f"🎨 使用模型 {model} 生成图像...")
            _log_info(f"📝 提示词: {enhanced_prompt[:100]}...")
            
            response_json = generate_with_rest_api(api_key, model, content_parts, generation_config, proxy=proxy, base_url=None)
            
            # 处理响应
            raw_text = extract_text_from_response(response_json)
            generated_image = process_generated_image_from_response(response_json)
            
            # 强制调整图像尺寸到用户指定的尺寸
            if generated_image is not None:
                try:
                    target_width, target_height = map(int, controls['size'].split('x'))
                    current_width, current_height = generated_image.size
                    
                    if (current_width, current_height) != (target_width, target_height):
                        _log_info(f"🔄 强制调整图像尺寸: {current_width}x{current_height} -> {target_width}x{target_height}")
                        
                        # 如果目标尺寸比例不同，使用智能填充方法
                        if current_width/current_height != target_width/target_height:
                            _log_info(f"📐 检测到比例变化，使用智能填充方法")
                            
                            # 方法1: 拉伸填充（保持宽高比，可能裁剪部分内容）
                            # 计算填充比例，确保覆盖整个目标区域
                            scale_x = target_width / current_width
                            scale_y = target_height / current_height
                            scale = max(scale_x, scale_y)  # 使用较大的缩放比例
                            
                            new_width = int(current_width * scale)
                            new_height = int(current_height * scale)
                            
                            # 缩放图像
                            scaled_image = generated_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            # 创建目标尺寸的画布
                            new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                            
                            # 计算居中位置
                            paste_x = (target_width - new_width) // 2
                            paste_y = (target_height - new_height) // 2
                            
                            # 粘贴到新画布中心
                            new_image.paste(scaled_image, (paste_x, paste_y))
                            
                            # 如果图像没有完全覆盖画布，使用边缘扩展
                            if paste_x < 0 or paste_y < 0:
                                _log_info(f"🔧 使用边缘扩展填充空白区域")
                                # 创建更大的临时画布
                                temp_width = max(target_width, new_width)
                                temp_height = max(target_height, new_height)
                                temp_image = Image.new('RGB', (temp_width, temp_height), (255, 255, 255))
                                
                                # 将缩放后的图像居中放置
                                temp_paste_x = (temp_width - new_width) // 2
                                temp_paste_y = (temp_height - new_height) // 2
                                temp_image.paste(scaled_image, (temp_paste_x, temp_paste_y))
                                
                                # 裁剪到目标尺寸
                                crop_x = (temp_width - target_width) // 2
                                crop_y = (temp_height - target_height) // 2
                                new_image = temp_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                            
                            resized_image = new_image
                        else:
                            # 比例相同，直接调整尺寸
                            resized_image = generated_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        
                        generated_image = resized_image
                        _log_info(f"✅ 图像尺寸调整完成: {generated_image.size}")
                    else:
                        _log_info(f"✅ 图像尺寸已符合要求: {generated_image.size}")
                        
                except Exception as e:
                    _log_warning(f"尺寸调整失败: {e}, 保持原始尺寸")
            
            if not raw_text or raw_text == "Response received but no text content":
                assistant_text = "遵命！这是你所要求的图片："
            else:
                assistant_text = raw_text.strip()
            
            self._push_chat(enhanced_prompt, assistant_text, unique_id)
            
            _log_info("✅ 图像生成成功完成")
            return (assistant_text, generated_image)
            
        except Exception as e:
            error_msg = str(e)
            _log_error(f"图像生成失败: {error_msg}")
            
            # 增强的错误分类处理
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                friendly_error = (
                    "API配额超限。解决方案:\n"
                    "1. 等待配额重置（通常24小时）\n"
                    "2. 升级到付费账户\n" 
                    "3. 使用免费模型\n"
                    "4. 检查计费设置"
                )
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                friendly_error = f"网络连接错误: 请检查代理设置和网络连接"
            elif "not found" in error_msg.lower() or "404" in error_msg:
                friendly_error = f"模型不可用: {model} 可能不支持图像生成或暂时不可用"
            elif "API key" in error_msg or "401" in error_msg or "403" in error_msg:
                friendly_error = "API密钥无效，请检查配置"
            elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                friendly_error = "内容被安全过滤器阻止，请修改提示词"
            else:
                friendly_error = f"生成失败: {error_msg}"
            
            return (friendly_error, create_dummy_image())

class KenChenLLMGeminiBananaImageToImageBananaNode:
    CATEGORY = "Ken-Chen/LLM Agent Assistant"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generation_text", "generated_image")
    FUNCTION = "transform_image"
    

    
    @classmethod
    def INPUT_TYPES(s):
        config = get_gemini_banana_config()
        default_params = config.get('default_params', {})
        image_settings = config.get('image_settings', {})
        
        # Get image generation models from config, with fallback to core Banana models
        models = config.get('models', {}).get('image_gen_models', [])
        if not models:
            # Fallback to core Banana models if config is empty
            models = [
                "gemini-2.5-flash-image-preview",  # Latest Banana model
                "gemini-2.0-flash-preview-image-generation"
            ]
        
        # Get default model from config, prioritize latest Banana model
        default_model = config.get('default_model', {}).get('image_gen', "gemini-2.5-flash-image-preview")
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        
        # Get image control presets
        size_presets = image_settings.get('size_presets', ["512x512", "768x768", "1024x1024", "1024x1792", "1792x1024"])
        quality_presets = image_settings.get('quality_presets', ["standard", "hd"])
        style_presets = image_settings.get('style_presets', ["vivid", "natural"])
        if "None" not in style_presets:
            style_presets = ["None"] + style_presets
        
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "API密钥（留空时自动从配置文件读取）"
                }),
                "prompt": ("STRING", {"default": "Transform this image", "multiline": True}),
                "image": ("IMAGE",),
                "model": (
                    models,
                    {"default": default_model},
                ),
                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),
                "size": (size_presets, {"default": image_settings.get('default_size', "1024x1024")}),
                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),
                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.9), "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": default_params.get('top_k', 40), "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": default_params.get('max_output_tokens', 2048), "min": 0, "max": 32768}),
                "seed": ("INT", {"default": default_params.get('seed', 0), "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                "custom_size": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "自定义尺寸 (如: 1920x1080)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generation_text", "generated_image")
    FUNCTION = "transform_image"
    CATEGORY = "Ken-Chen/LLM Agent Assistant"

    def transform_image(
        self,
        api_key,
        prompt,
        image,
        model,
        proxy,
        size,
        quality,
        style,
        temperature,
        top_p,
        top_k,
        max_output_tokens,
        seed,
        custom_size: str = "",
        unique_id: str = "",
    ):
        try:
            # 如果用户没有输入API密钥，自动从配置文件获取
            if not api_key or not api_key.strip():
                config = get_gemini_banana_config()
                auto_api_key = config.get('api_key', '')
                if auto_api_key and auto_api_key.strip():
                    api_key = auto_api_key.strip()
                    _log_info(f"🔑 自动使用配置文件中的API密钥: {api_key[:8]}...")
                else:
                    error_msg = "API密钥不能为空，请在配置文件中设置api_key或手动输入"
                    _log_error(error_msg)
                    return (error_msg, create_dummy_image())
            
            # 验证提示词
            if not prompt.strip():
                error_msg = "提示词不能为空"
                _log_error(error_msg)
                return (error_msg, create_dummy_image())
            
            # 处理图像控制参数
            controls = process_image_controls(size, quality, style, custom_size)
            enhanced_prompt = enhance_prompt_with_controls(prompt.strip(), controls)
            
            _log_info(f"🎨 图像控制参数: 尺寸={controls['size']}, 质量={controls['quality']}, 风格={controls['style']}")
            if controls['is_custom_size']:
                _log_info(f"📏 使用自定义尺寸: {controls['size']}")
            
            # 代理处理：使用 proxies 参数，不设置环境变量，避免冲突
            if proxy and proxy.strip() and "None" not in proxy:
                # 使用 proxies 参数，不设置环境变量
                _log_info(f"使用代理: {proxy.strip()}")
            else:
                _log_info("未使用代理")
            
            # 构建生成配置
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"]  # 关键：启用图像生成
            }
            
            # 智能种子控制
            if seed > 0:
                generation_config["seed"] = seed
            
            # 准备内容 - 文本 + 图像
            content_parts = [{"text": enhanced_prompt}]
            content_parts.extend(prepare_media_content(image=image))
            
            # 使用REST API调用
            _log_info(f"🖼️ 使用模型 {model} 进行图像转换...")
            _log_info(f"📝 转换指令: {enhanced_prompt[:100]}...")
            
            response_json = generate_with_rest_api(api_key, model, content_parts, generation_config, proxy=proxy, base_url=None)
            
            # 处理响应
            raw_text = extract_text_from_response(response_json)
            generated_image = process_generated_image_from_response(response_json)
            
            # 强制调整图像尺寸到用户指定的尺寸
            if generated_image is not None:
                try:
                    target_width, target_height = map(int, controls['size'].split('x'))
                    current_width, current_height = generated_image.size
                    
                    if (current_width, current_height) != (target_width, target_height):
                        _log_info(f"🔄 强制调整图像尺寸: {current_width}x{current_height} -> {target_width}x{target_height}")
                        
                        # 如果目标尺寸比例不同，使用智能填充方法
                        if current_width/current_height != target_width/target_height:
                            _log_info(f"📐 检测到比例变化，使用智能填充方法")
                            
                            # 方法1: 拉伸填充（保持宽高比，可能裁剪部分内容）
                            # 计算填充比例，确保覆盖整个目标区域
                            scale_x = target_width / current_width
                            scale_y = target_height / current_height
                            scale = max(scale_x, scale_y)  # 使用较大的缩放比例
                            
                            new_width = int(current_width * scale)
                            new_height = int(current_height * scale)
                            
                            # 缩放图像
                            scaled_image = generated_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            # 创建目标尺寸的画布
                            new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                            
                            # 计算居中位置
                            paste_x = (target_width - new_width) // 2
                            paste_y = (target_height - new_height) // 2
                            
                            # 粘贴到新画布中心
                            new_image.paste(scaled_image, (paste_x, paste_y))
                            
                            # 如果图像没有完全覆盖画布，使用边缘扩展
                            if paste_x < 0 or paste_y < 0:
                                _log_info(f"🔧 使用边缘扩展填充空白区域")
                                # 创建更大的临时画布
                                temp_width = max(target_width, new_width)
                                temp_height = max(target_height, new_height)
                                temp_image = Image.new('RGB', (temp_width, temp_height), (255, 255, 255))
                                
                                # 将缩放后的图像居中放置
                                temp_paste_x = (temp_width - new_width) // 2
                                temp_paste_y = (temp_height - new_height) // 2
                                temp_image.paste(scaled_image, (temp_paste_x, temp_paste_y))
                                
                                # 裁剪到目标尺寸
                                crop_x = (temp_width - target_width) // 2
                                crop_y = (temp_height - target_height) // 2
                                new_image = temp_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                            
                            resized_image = new_image
                        else:
                            # 比例相同，直接调整尺寸
                            resized_image = generated_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        
                        generated_image = resized_image
                        _log_info(f"✅ 图像尺寸调整完成: {generated_image.size}")
                    else:
                        _log_info(f"✅ 图像尺寸已符合要求: {generated_image.size}")
                        
                except Exception as e:
                    _log_warning(f"尺寸调整失败: {e}, 保持原始尺寸")
            
            # 如果没有生成新图像，返回原图像
            if generated_image is None:
                _log_warning("未检测到编辑后的图像，返回原图像")
                generated_image = image if image is not None else create_dummy_image()
            
            if not raw_text or raw_text == "Response received but no text content":
                assistant_text = "遵命！这是根据你的编辑指令生成的图片："
            else:
                assistant_text = raw_text.strip()
            
            self._push_chat(enhanced_prompt, assistant_text, unique_id)
            
            _log_info("✅ 图像转换成功完成")
            return (assistant_text, generated_image)
            
        except Exception as e:
            error_msg = str(e)
            _log_error(f"图像转换失败: {error_msg}")
            
            # 增强的错误分类处理
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                friendly_error = (
                    "API配额超限。解决方案:\n"
                    "1. 等待配额重置（通常24小时）\n"
                    "2. 升级到付费账户\n"
                    "3. 使用免费模型\n"
                    "4. 检查计费设置"
                )
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                friendly_error = f"网络连接错误: 请检查代理设置和网络连接"
            elif "not found" in error_msg.lower() or "404" in error_msg:
                friendly_error = f"模型不可用: {model} 可能不支持图像转换或暂时不可用"
            elif "API key" in error_msg or "401" in error_msg or "403" in error_msg:
                friendly_error = "API密钥无效，请检查配置"
            elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                friendly_error = "内容被安全过滤器阻止，请修改提示词或图像"
            else:
                friendly_error = f"转换失败: {error_msg}"
            
            return (friendly_error, image if image is not None else create_dummy_image())

class KenChenLLMGeminiBananaMultimodalBananaNode:
    CATEGORY = "Ken-Chen/LLM Agent Assistant"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze_multimodal"
    

    
    @classmethod
    def INPUT_TYPES(s):
        config = get_gemini_banana_config()
        default_params = config.get('default_params', {})
        
        # Get multimodal models from config, with fallback to latest models
        models = config.get('models', {}).get('multimodal_models', [])
        if not models:
            # Fallback to latest models if config is empty - only include true multimodal models
            models = [
                "gemini-2.5-pro-exp-03-25",  # Latest multimodal model - supports image+text
                "gemini-2.5-flash-preview-04-17",  # Supports image+text
                "gemini-2.5-pro-preview-05-06",  # Supports image+text
                "gemini-2.0-flash-exp-image-generation",  # Supports image+text
            ]
        
        # Get default model from config, prioritize latest models
        default_model = config.get('default_model', {}).get('multimodal', "gemini-2.5-pro-exp-03-25")
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": get_gemini_banana_config().get('multimodal_api_key', ''), 
                    "multiline": False,
                    "placeholder": "API密钥（留空时自动从配置文件读取）"
                }),
                "prompt": ("STRING", {"default": "Describe what you see", "multiline": True}),
                "model": (
                    models,
                    {"default": default_model},
                ),

                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),
                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.9), "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": default_params.get('top_k', 40), "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": default_params.get('max_output_tokens', 2048), "min": 0, "max": 8192}),
                "seed": ("INT", {"default": default_params.get('seed', 0), "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }
    

    
    def analyze_multimodal(
        self,
        api_key,
        prompt,
        model,
        proxy,
        temperature,
        top_p,
        top_k,
        max_output_tokens,
        seed,
        image=None,
        audio=None,
    ):
        try:
            # 如果用户没有输入API密钥，自动从配置文件获取
            if not api_key or not api_key.strip():
                config = get_gemini_banana_config()
                auto_api_key = config.get('multimodal_api_key', '')
                if auto_api_key and auto_api_key.strip():
                    api_key = auto_api_key.strip()
                    _log_info(f"🔑 自动使用配置文件中的API密钥: {api_key[:8]}...")
                else:
                    error_msg = "API密钥不能为空，请在配置文件中设置multimodal_api_key或手动输入"
                    _log_error(error_msg)
                    return (error_msg,)
            
            # 设置代理
            # 代理处理：使用 proxies 参数，不设置环境变量，避免冲突
            if proxy and proxy.strip() and "None" not in proxy:
                # 使用 proxies 参数，不设置环境变量
                _log_info(f"使用代理: {proxy.strip()}")
            else:
                _log_info("未使用代理")
            
            # 构建生成配置（多模态分析只需要TEXT输出）
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT"]  # 只需要文本输出
            }
            
            if seed > 0:
                generation_config["seed"] = seed
            

            
            # 准备内容 - 文本 + 多媒体
            content_parts = [{"text": prompt.strip()}]
            content_parts.extend(prepare_media_content(image=image, audio=audio))
            
            # 使用REST API调用
            _log_info(f"🔍 使用模型 {model} 进行多模态分析...")
            _log_info(f"📝 分析提示: {prompt[:100]}...")
            
            response_json = generate_with_rest_api(api_key, model, content_parts, generation_config, proxy=proxy)
            
            # 提取文本响应
            generated_text = extract_text_from_response(response_json)
            
            if not generated_text or generated_text == "Response received but no text content":
                generated_text = "模型未返回有效的分析结果"
            
            _log_info("✅ 多模态分析成功完成")
            return (generated_text,)
            
        except Exception as e:
            error_msg = str(e)
            _log_error(f"多模态分析失败: {error_msg}")
            
            # 增强的错误分类处理
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                friendly_error = (
                    "API配额超限。解决方案:\n"
                    "1. 等待配额重置（通常24小时）\n"
                    "2. 升级到付费账户\n"
                    "3. 使用免费模型\n"
                    "4. 检查计费设置"
                )
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                friendly_error = f"网络连接错误: 请检查代理设置和网络连接"
            elif "not found" in error_msg.lower() or "404" in error_msg:
                friendly_error = f"模型不可用: {model} 可能不支持多模态分析或暂时不可用"
            elif "API key" in error_msg or "401" in error_msg or "403" in error_msg:
                friendly_error = "API密钥无效，请检查配置"
            elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                friendly_error = "内容被安全过滤器阻止，请修改提示词或媒体内容"
            else:
                friendly_error = f"分析失败: {error_msg}"
            
            return (friendly_error,)

class KenChenLLMGeminiBananaMultiImageEditNode:
    """
    Gemini Banana 多图像编辑节点
    
    功能特性:
    - 支持多张输入图像（最多4张）
    - 专业的图像编辑提示词
    - 支持尺寸、质量、风格控制
    - 智能图像组合编辑
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        from .gemini_banana import get_gemini_banana_config
        config = get_gemini_banana_config()
        default_params = config.get('default_params', {})
        image_settings = config.get('image_settings', {})
        
        # Get image control presets
        size_presets = image_settings.get('size_presets', ["512x512", "768x768", "1024x1024", "1024x1792", "1792x1024"])
        quality_presets = image_settings.get('quality_presets', ["standard", "hd"])
        style_presets = image_settings.get('style_presets', ["vivid", "natural"])
        
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "请根据这些图片进行专业的图像编辑", "multiline": True}),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.0-flash"], {"default": "gemini-2.5-flash-image-preview"}),
                "size": (size_presets, {"default": image_settings.get('default_size', "1024x1024")}),
                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),
                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.95), "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": default_params.get('top_k', 40), "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": default_params.get('max_output_tokens', 8192), "min": 0, "max": 32768}),
                "seed": ("INT", {"default": default_params.get('seed', 0), "min": 0, "max": 999999}),
            },
            "optional": {
                "custom_size": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "自定义尺寸 (如: 1920x1080)"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "edit_multiple_images"
    CATEGORY = "Ken-Chen/LLM Agent Assistant"

    def _push_chat(self, user_prompt: str, response_text: str, unique_id: str):
        if not PromptServer or not unique_id:
            return
        try:
            render_spec = {
                "node_id": unique_id,
                "component": "ChatHistoryWidget",
                "props": {
                    "history": json.dumps([
                        {
                            "prompt": user_prompt,
                            "response": response_text,
                            "response_id": str(random.randint(100000, 999999)),
                            "timestamp": time.time(),
                        }
                    ])
                },
            }
            PromptServer.instance.send_sync("display_component", render_spec)
        except Exception:
            pass

    def edit_multiple_images(self, api_key: str, prompt: str, model: str, size: str, quality: str, style: str,
                           temperature: float, top_p: float, top_k: int, max_output_tokens: int, seed: int,
                           custom_size: str = "", image1=None, image2=None, image3=None, image4=None,
                           unique_id: str = "") -> Tuple[torch.Tensor, str]:
        """使用 Gemini API 进行多图像编辑"""
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 处理图像控制参数
        controls = process_image_controls(size, quality, style, custom_size)
        enhanced_prompt = enhance_prompt_with_controls(prompt.strip(), controls)
        
        print(f"🎨 图像控制参数: 尺寸={controls['size']}, 质量={controls['quality']}, 风格={controls['style']}")
        if controls['is_custom_size']:
            print(f"📏 使用自定义尺寸: {controls['size']}")
        
        # 收集所有输入的图像
        all_input_pils = []
        input_images = [image1, image2, image3, image4]
        
        for i, img_tensor in enumerate(input_images):
            if img_tensor is not None:
                try:
                    pil_image = tensor_to_pil(img_tensor)
                    if pil_image:
                        all_input_pils.append(pil_image)
                        print(f"📸 添加输入图像 {i+1}: {pil_image.size}")
                except Exception as e:
                    print(f"⚠️ 图像 {i+1} 处理失败: {e}")
        
        if not all_input_pils:
            raise ValueError("错误：请输入至少一张要编辑的图像")
        
        print(f"🖼️ 总共收集到 {len(all_input_pils)} 张输入图像")
        
        # 智能生成多图编辑提示词
        # 首先处理图片引用转换，确保所有情况下都能使用
        user_intent = prompt.strip()
        converted_prompt = user_intent
        
        # 转换所有图片引用 - 通用化处理
        if len(all_input_pils) >= 1:
            converted_prompt = converted_prompt.replace("图1", "第一张图片")
        if len(all_input_pils) >= 2:
            converted_prompt = converted_prompt.replace("图2", "第二张图片")
        if len(all_input_pils) >= 3:
            converted_prompt = converted_prompt.replace("图3", "第三张图片")
        if len(all_input_pils) >= 4:
            converted_prompt = converted_prompt.replace("图4", "第四张图片")
        
        # 根据图片数量生成不同的提示词 - 完全通用化
        if len(all_input_pils) == 2:
            # 2张图片：通用组合编辑
            full_prompt = f"""请严格按照以下要求进行图像编辑：

{converted_prompt}

重要说明：
- 请仔细分析两张图片的内容和关系
- 根据用户的具体指令，将第二张图片中的元素应用到第一张图片中
- 保持第一张图片的核心特征和背景环境
- 确保第二张图片中的元素与第一张图片完美融合
- 编辑结果应该看起来自然真实，符合用户意图

{enhanced_prompt}

请严格按照上述要求执行，确保编辑结果完全符合用户意图。"""
        elif len(all_input_pils) == 1:
            full_prompt = f"""你是一个专业的图像编辑专家。请根据以下要求编辑这张图片：

{enhanced_prompt}

请使用你的图像编辑能力，生成高质量的编辑结果。"""
        else:
            full_prompt = f"""你是一个专业的图像编辑专家。请根据这些图片和以下指令进行图像编辑：

{enhanced_prompt}

请使用你的图像编辑能力，生成高质量的编辑结果。确保编辑后的图像符合所有要求。"""
        
        # 构建API请求内容
        content = [{"type": "text", "text": full_prompt}]
        
        # 添加所有图像作为参考
        for i, pil_image in enumerate(all_input_pils):
            # 调整图像尺寸以符合API要求
            pil_image = resize_image_for_api(pil_image)
            # 转换为base64
            image_base64 = image_to_base64(pil_image, format='JPEG')
            content.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            })
        
        # 构建请求数据
        request_data = {
            "contents": [{
                "parts": content
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"],
                "seed": seed if seed and seed > 0 else None
            }
        }
        
        # 清理 None 值
        if request_data["generationConfig"]["seed"] is None:
            del request_data["generationConfig"]["seed"]
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip()
        }
        
        # 智能重试机制
        max_retries = 5
        timeout = 120
        
        for attempt in range(max_retries):
            try:
                print(f"🖼️ 正在编辑图片... (尝试 {attempt + 1}/{max_retries})")
                print(f"📝 编辑指令: {enhanced_prompt[:100]}...")
                print(f"🔗 使用模型: {model}")
                
                # 发送请求
                response = requests.post(
                    f"{get_gemini_banana_config().get('base_url', 'https://generativelanguage.googleapis.com')}/v1beta/models/{model}:generateContent",
                    headers=headers,
                    json=request_data,
                    timeout=timeout
                )
                
                # 成功响应
                if response.status_code == 200:
                    # 解析响应
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取文本响应和编辑后的图片
                    response_text = ""
                    edited_image = None
                    
                    if "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                # 提取文本
                                if "text" in part:
                                    response_text += part["text"]
                                
                                # 提取编辑后的图片
                                if "inline_data" in part or "inlineData" in part:
                                    inline_data = part.get("inline_data") or part.get("inlineData")
                                    if inline_data and "data" in inline_data:
                                        try:
                                            # 解码图片数据
                                            image_data = inline_data["data"]
                                            image_bytes = base64.b64decode(image_data)
                                            edited_image = Image.open(io.BytesIO(image_bytes))
                                            print("✅ 成功提取编辑后的图片")
                                        except Exception as e:
                                            print(f"⚠️ 解码图片失败: {e}")
                    
                    # 如果没有编辑后的图片，返回原图片
                    if edited_image is None:
                        print("⚠️ 未检测到编辑后的图片，返回原图片")
                        edited_image = all_input_pils[0]  # 返回第一张图片
                        if not response_text:
                            response_text = "图片编辑请求已发送，但未收到编辑后的图片"
                    
                    # 强制调整图像尺寸到用户指定的尺寸
                    try:
                        target_width, target_height = map(int, controls['size'].split('x'))
                        current_width, current_height = edited_image.size
                        
                        if (current_width, current_height) != (target_width, target_height):
                            print(f"🔄 强制调整图像尺寸: {current_width}x{current_height} -> {target_width}x{target_height}")
                            
                            # 如果目标尺寸比例不同，使用智能填充方法
                            if current_width/current_height != target_width/target_height:
                                print(f"📐 检测到比例变化，使用智能填充方法")
                                
                                # 计算填充比例，确保覆盖整个目标区域
                                scale_x = target_width / current_width
                                scale_y = target_height / current_height
                                scale = max(scale_x, scale_y)  # 使用较大的缩放比例
                                
                                new_width = int(current_width * scale)
                                new_height = int(current_height * scale)
                                
                                # 缩放图像
                                scaled_image = edited_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                
                                # 创建目标尺寸的画布
                                new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                                
                                # 计算居中位置
                                paste_x = (target_width - new_width) // 2
                                paste_y = (target_height - new_height) // 2
                                
                                # 粘贴到新画布中心
                                new_image.paste(scaled_image, (paste_x, paste_y))
                                
                                # 如果图像没有完全覆盖画布，使用边缘扩展
                                if paste_x < 0 or paste_y < 0:
                                    print(f"🔧 使用边缘扩展填充空白区域")
                                    # 创建更大的临时画布
                                    temp_width = max(target_width, new_width)
                                    temp_height = max(target_height, new_height)
                                    temp_image = Image.new('RGB', (temp_width, temp_height), (255, 255, 255))
                                    
                                    # 将缩放后的图像居中放置
                                    temp_paste_x = (temp_width - new_width) // 2
                                    temp_paste_y = (temp_height - new_height) // 2
                                    temp_image.paste(scaled_image, (temp_paste_x, temp_paste_y))
                                    
                                    # 裁剪到目标尺寸
                                    crop_x = (temp_width - target_width) // 2
                                    crop_y = (temp_height - target_height) // 2
                                    new_image = temp_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                                
                                edited_image = new_image
                            else:
                                # 比例相同，直接调整尺寸
                                edited_image = edited_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                            
                            print(f"✅ 图像尺寸调整完成: {edited_image.size}")
                        else:
                            print(f"✅ 图像尺寸已符合要求: {edited_image.size}")
                            
                    except Exception as e:
                        print(f"⚠️ 尺寸调整失败: {e}, 保持原始尺寸")
                    
                    # 如果没有响应文本，提供默认文本
                    if not response_text:
                        response_text = "多图像编辑完成！这是根据您的指令和参考图像生成的编辑结果。"
                        print("📝 使用默认响应文本")
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(edited_image)
                    
                    print("✅ 多图像编辑完成")
                    print(f"📝 响应文本长度: {len(response_text)}")
                    print(f"📝 响应文本内容: {response_text[:200]}...")
                    self._push_chat(enhanced_prompt, response_text or "", unique_id)
                    return (image_tensor, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        print(f"❌ 错误文本: {response.text}")
                    
                    # 如果是最后一次尝试，抛出异常
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    
                    # 智能等待
                    delay = smart_retry_delay(attempt, response.status_code)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except requests.exceptions.RequestException as e:
                error_msg = format_error_message(e)
                print(f"❌ 请求失败: {error_msg}")
                if attempt == max_retries - 1:
                    raise ValueError(f"API请求失败: {error_msg}")
                else:
                    delay = smart_retry_delay(attempt)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except Exception as e:
                error_msg = format_error_message(e)
                print(f"❌ 处理失败: {error_msg}")
                raise ValueError(f"多图像编辑失败: {error_msg}")

# Node mappings
NODE_CLASS_MAPPINGS = {
    "KenChenLLMGeminiBananaTextToImageBananaNode": KenChenLLMGeminiBananaTextToImageBananaNode,
    "KenChenLLMGeminiBananaImageToImageBananaNode": KenChenLLMGeminiBananaImageToImageBananaNode,
    "KenChenLLMGeminiBananaMultimodalBananaNode": KenChenLLMGeminiBananaMultimodalBananaNode,
    "GeminiBananaMultiImageEdit": KenChenLLMGeminiBananaMultiImageEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KenChenLLMGeminiBananaTextToImageBananaNode": "Gemini Banana Text to Image Banana",
    "KenChenLLMGeminiBananaImageToImageBananaNode": "Gemini Banana Image to Image Banana", 
    "KenChenLLMGeminiBananaMultimodalBananaNode": "Gemini Banana Multimodal Banana",
    "GeminiBananaMultiImageEdit": "Gemini Banana Multi Image Edit",
}

