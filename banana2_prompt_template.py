import os
import json
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# 统一日志接口
def _log_info(msg):
    print(f"[Banana2] {msg}")

def _log_error(msg):
    print(f"[Banana2] {msg}")

# 引入通用端点/认证构建函数（如不可用则本地实现）
try:
    from .general_api import _build_endpoint, _auto_auth_headers, _b64_from_tensor
except ImportError:
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
        if u.endswith('/v1') or u.endswith('/v1beta') or u.endswith('/v1alpha'):
            return f"{u}/models/{model}:generateContent"
        ver = (version or "Auto").lower()
        if ver == "auto":
            ver = "v1beta" if "generativelanguage.googleapis.com" in u else "v1"
        return f"{u}/{ver}/models/{model}:generateContent"

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

class KenChenLLMBanana2PromptTemplateNode:
    """Banana2 提示词模板节点
    
    基于 Gemini-Multimodal 能力，构建专门用于生成 Banana 图像生成/编辑提示词的节点
    支持多种提示词模板和媒体输入
    """
    
    @classmethod
    def INPUT_TYPES(s):
        # 定义提示词模板类型 - 基于Google官方14个模板
        template_types = [
            # === 图片生成模板 (6个) ===
            "生成-逼真场景",
            "生成-风格化插画和贴纸",
            "生成-图片中的文字",
            "生成-产品模型和商业摄影",
            "生成-极简风格和负空间",
            "生成-连续艺术(漫画分格)",
            # === 图片编辑模板 (7个) ===
            "编辑-添加和移除元素",
            "编辑-局部重绘",
            "编辑-风格迁移",
            "编辑-高级合成(多图组合)",
            "编辑-高保真细节保留",
            "编辑-让事物焕发活力",
            "编辑-角色一致性(360度)",
            # === Sora视频/动画模板 ===
            "编辑-Sora动漫3宫格提示词模板",
            "编辑-Sora动漫3宫格绘图提示词模板",
            "编辑-Sora动漫5宫格提示词模板",
            "编辑-Sora动漫5宫格绘图提示词模板",
            # === 扩展创意模板 ===
            "创意-电影级场景",
            "创意-概念艺术设计",
            "创意-时尚摄影",
            "创意-建筑可视化",
            "创意-食物摄影",
            "创意-抽象艺术",
            "创意-儿童插画",
            "创意-海报设计",
            # === 自定义 ===
            "自定义模板"
        ]

        # 动态获取模型列表，参考Gemini-Multimodal节点的实现
        base_models = [
            "gemini-3-pro-preview",
            "gemini-3-pro-preview-thinking",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]

        # 只有gemini-3-pro-preview全部提供商都支持
        all_provider_models = ["gemini-3-pro-preview"]

        labelled = []
        providers_by_model = {}
        try:
            from .gemini import get_gemini_config
            _cfg = get_gemini_config()
            for prov, detail in (_cfg.get("api_providers", {}) or {}).items():
                for m in (detail.get("models") or []):
                    providers_by_model.setdefault(m, []).append(prov)
        except Exception:
            pass

        # 为模型添加提供商标签
        for m in base_models:
            if m in all_provider_models:
                # 只有gemini-3-pro-preview标识为[all]
                labelled.append(f"{m} [all]")
            elif m in providers_by_model:
                prov_list = "/".join(providers_by_model[m])
                labelled.append(f"{m} [{prov_list}]")
            else:
                labelled.append(f"{m} [google]")

        default_label = labelled[0] if labelled else "gemini-3-pro-preview [all]"

        return {
            "required": {
                "user_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "输入您的提示词..."}),
                "template_type": (template_types, {"default": "生成-逼真场景"}),
                "model": (labelled, {"default": default_label}),
                "api_provider": (["google", "comet", "T8的贞贞AI工坊", "comfly", "aabao", "custom"], {"default": "google"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "API Key (留空自动使用Gemini_config.json)"}),
                "base_url": ("STRING", {"default": "", "multiline": False, "placeholder": "Base URL (留空使用配置或默认)"}),
                "version": (["Auto", "v1", "v1alpha", "v1beta"], {"default": "Auto"}),
                "auth_mode": (["auto", "google_xgoog", "bearer"], {"default": "auto"}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "thinking_level": (["high", "low"], {"default": "high"}),
                "media_resolution": (["Auto", "media_resolution_low", "media_resolution_medium", "media_resolution_high"], {"default": "Auto"}),
            },
            "optional": {
                "image": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "image_3": ("IMAGE", ),
                "image_4": ("IMAGE", ),
                "custom_template": ("STRING", {"default": "", "multiline": True, "placeholder": "自定义模板内容 (仅当选择'自定义模板'时使用)"}),
                "system_instruction": ("STRING", {"default": "", "multiline": True, "placeholder": "可选的系统指令..."}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("banana_prompt", "raw_response", )
    FUNCTION = "generate_banana_prompt"
    CATEGORY = "🍌 Banana"
    
    def generate_banana_prompt(
        self,
        user_prompt,
        template_type,
        model,
        api_provider,
        api_key,
        base_url,
        version,
        auth_mode,
        max_output_tokens,
        temperature,
        thinking_level,
        media_resolution,
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
        custom_template="",
        system_instruction="",
    ):
        try:
            # 从带标签的模型名称中提取真实模型ID
            model_id = model.split(' ')[0]

            # 获取配置文件
            try:
                from .gemini import get_gemini_config
                _cfg = get_gemini_config()
            except ImportError:
                _cfg = {}

            # 根据提供者解析 API Key 与 Base URL（用户优先→提供商配置→全局配置→默认）
            user_key = (api_key or "").strip()
            user_base = (base_url or "").strip()

            provs = _cfg.get("api_providers", {}) or {}
            top_api_key = (_cfg.get("api_key") or "").strip()
            top_base = (_cfg.get("base_url") or "").strip()

            provider_defaults = {
                "google": "https://generativelanguage.googleapis.com",
                "comet": "https://api.cometapi.com",
                "T8的贞贞AI工坊": "https://ai.t8star.cn/v1",
                "comfly": "https://ai.comfly.chat/v1",
                "aabao": "https://api.aabao.top/v1",
            }

            if (api_provider or "") == "custom":
                if not user_key or not user_base:
                    error_msg = "错误: 选择 'custom' 时必须输入 API Key 和 Base URL"
                    _log_error(error_msg)
                    return (error_msg, error_msg)
                final_api_key = user_key
                final_base = user_base
            else:
                prov_cfg = provs.get(api_provider, {}) if isinstance(provs, dict) else {}
                cfg_key = (prov_cfg.get("api_key") or top_api_key or "").strip()
                cfg_base = (prov_cfg.get("base_url") or top_base or "").strip()

                final_api_key = user_key if user_key else cfg_key
                final_base = user_base if user_base else (cfg_base or provider_defaults.get(api_provider, ""))

                if not final_api_key:
                    error_msg = "错误: 需要 API Key（在节点或 Gemini_config.json 中提供）"
                    _log_error(error_msg)
                    return (error_msg, error_msg)

            # 版本选择：Auto下 Google 根据 media_resolution 切 v1alpha/v1beta，其他走 v1
            ver_in = (version or "Auto").strip()
            if ver_in.lower() == "auto":
                if api_provider == "google" or "generativelanguage.googleapis.com" in (final_base or ""):
                    final_version = "v1alpha" if media_resolution != "Auto" else "v1beta"
                else:
                    final_version = "v1"
            else:
                final_version = ver_in

            # 准备模板内容
            template = self._get_template(template_type, custom_template)

            # 构建完整提示词
            full_prompt = f"{template}\n\n用户提示词: {user_prompt}"

            # 准备系统指令
            if not system_instruction:
                system_instruction = """你是一个专业的图像生成提示词专家，专门为Banana图像生成模型创建高质量提示词。

请根据用户的需求和提供的图片，生成一个详细、具体且有创意的提示词。你的提示词应该：
1. 具体而非抽象 - 使用具体的描述而不是模糊的概念
2. 包含丰富的视觉细节 - 颜色、纹理、光照、构图等
3. 考虑艺术风格和技术参数
4. 适合Banana模型的特点和能力
5. 结构清晰，易于理解和执行

请直接输出优化后的Banana提示词，不需要额外的解释。"""

            _log_info(f"🍌 使用模板: {template_type}")
            _log_info(f"🤖 使用模型: {model_id}")
            _log_info(f"🔌 提供商: {api_provider} | 认证: {auth_mode} | 版本: {final_version}")
            _log_info(f"🌐 端点基址: {final_base}")

            # 构建端点与请求头
            endpoint = _build_endpoint(final_base, model_id, final_version)
            headers = _auto_auth_headers(final_base, final_api_key, auth_mode)

            # 组装 contents: 首先是提示词文本
            parts = [{"text": full_prompt}]
            for img_tensor in [image, image_2, image_3, image_4]:
                b64_img = _b64_from_tensor(img_tensor, "image/png") if img_tensor is not None else None
                if b64_img:
                    parts.append({"inlineData": {"mimeType": "image/png", "data": b64_img}})

            contents = [{"role": "user", "parts": parts}]

            # generationConfig（支持思维级别与输出长度/温度）
            gen_cfg = {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens),
                "thinkingConfig": {"thinkingLevel": thinking_level},
            }

            payload = {"contents": contents, "generationConfig": gen_cfg}
            if system_instruction and system_instruction.strip():
                payload["systemInstruction"] = {"role": "system", "parts": [{"text": system_instruction.strip()}]}

            import requests
            resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=120)
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}: {resp.text}"
                _log_error(err)
                return (err, err)

            data = resp.json()
            cands = data.get('candidates') or []
            if not cands:
                msg = "错误: 无候选结果"
                _log_error(msg)
                return (msg, msg)

            parts_out = cands[0].get('content', {}).get('parts', [])
            out_text = "".join([p.get('text', '') for p in parts_out if isinstance(p, dict)])
            usage_info = json.dumps(data.get('usageMetadata', {}), ensure_ascii=False, indent=2)

            _log_info("✅ Banana提示词生成成功")
            return (out_text or "未能生成有效提示词", usage_info)

        except Exception as e:
            error_msg = f"生成Banana提示词时出错: {str(e)}"
            _log_error(error_msg)
            return (error_msg, error_msg)

    def _fallback_generate_prompt(self, user_prompt, template, template_type):
        """备用提示词生成方法（当无法调用Gemini API时）"""
        try:
            # 基于模板类型生成基础提示词
            enhanced_prompt = f"""基于{template_type}，为以下用户需求生成Banana图像生成提示词：

用户需求：{user_prompt}

优化后的Banana提示词：
{user_prompt}

[注意：此为离线生成的基础提示词，建议配置Gemini API以获得更好的效果]"""

            _log_info("⚠️ 使用备用方法生成基础提示词")
            return (enhanced_prompt, enhanced_prompt)

        except Exception as e:
            error_msg = f"备用提示词生成失败: {str(e)}"
            _log_error(error_msg)
            return (error_msg, error_msg)
    
    def _get_template(self, template_type, custom_template):
        """根据选择的模板类型返回对应的提示词模板 - 基于Google官方14个模板"""

        templates = {
            # === 图片生成模板 (6个) ===
            "生成-逼真场景": """# Google官方模板1: 逼真场景生成
基于以下模板为用户生成专业的逼真场景提示词：

模板结构：
A photorealistic [shot type] of [subject], [action or expression], set in [environment]. The scene is illuminated by [lighting description], creating a [mood] atmosphere. Captured with a [camera/lens details], emphasizing [key textures and details]. The image should be in a [aspect ratio] format.

请根据用户的描述，填充模板中的各个要素，生成一个完整的逼真场景提示词。确保包含：
- 拍摄类型（close-up, wide shot, medium shot等）
- 主体对象的详细描述
- 动作或表情
- 环境设定
- 光照描述
- 氛围营造
- 相机/镜头细节
- 关键纹理和细节
- 画面比例

请输出完整的英文提示词。""",

            "生成-风格化插画和贴纸": """# Google官方模板2: 风格化插画和贴纸
基于以下模板为用户生成风格化插画和贴纸提示词：

模板结构：
A [style] sticker of a [subject], featuring [key characteristics] and a [color palette]. The design should have [line style] and [shading style]. The background must be transparent.

请根据用户的描述，填充模板中的各个要素，生成一个完整的风格化插画提示词。确保包含：
- 艺术风格（cartoon, anime, minimalist, vintage等）
- 主体对象
- 关键特征
- 色彩搭配
- 线条风格
- 阴影风格
- 透明背景要求

请输出完整的英文提示词。""",

            "生成-图片中的文字": """# Google官方模板3: 图片中的文字准确无误
基于以下模板为用户生成包含准确文字的图像提示词：

模板结构：
Create a [image type] for [brand/concept] with the text "[text to render]" in a [font style]. The design should be [style description], with a [color scheme].

请根据用户的描述，填充模板中的各个要素，生成一个包含准确文字的提示词。确保包含：
- 图像类型（logo, poster, banner, card等）
- 品牌/概念
- 要渲染的确切文字内容
- 字体风格
- 设计风格描述
- 色彩方案

请输出完整的英文提示词，特别注意文字内容的准确性。""",

            "生成-产品模型和商业摄影": """# Google官方模板4: 产品模型和商业摄影
基于以下模板为用户生成专业产品摄影提示词：

模板结构：
A high-resolution, studio-lit product photograph of a [product description] on a [background surface/description]. The lighting is a [lighting setup, e.g., three-point softbox setup] to [lighting purpose]. The camera angle is a [angle type] to showcase [specific feature]. Ultra-realistic, with sharp focus on [key detail]. [Aspect ratio].

请根据用户的描述，填充模板中的各个要素，生成一个专业产品摄影提示词。确保包含：
- 产品详细描述
- 背景表面/描述
- 灯光设置
- 灯光目的
- 相机角度
- 要展示的特定功能
- 关键细节的焦点
- 画面比例

请输出完整的英文提示词。""",

            "生成-极简风格和负空间": """# Google官方模板5: 极简风格和负空间设计
基于以下模板为用户生成极简风格提示词：

模板结构：
A minimalist composition featuring a single [subject] positioned in the [bottom-right/top-left/etc.] of the frame. The background is a vast, empty [color] canvas, creating significant negative space. Soft, subtle lighting. [Aspect ratio].

请根据用户的描述，填充模板中的各个要素，生成一个极简风格提示词。确保包含：
- 单一主体对象
- 主体在画面中的位置
- 背景颜色
- 负空间的强调
- 柔和微妙的光照
- 画面比例

请输出完整的英文提示词。""",

            "生成-连续艺术(漫画分格)": """# Google官方模板6: 连续艺术（漫画分格/故事板）
基于以下模板为用户生成连续艺术提示词：

模板结构：
Make a 3 panel comic in a [style]. Put the character in a [type of scene].

请根据用户的描述，填充模板中的各个要素，生成一个连续艺术提示词。确保包含：
- 漫画风格（manga, western comic, webcomic等）
- 角色设定
- 场景类型
- 3个分格的连续性
- 故事情节的发展

请输出完整的英文提示词，并考虑分格间的连贯性。""",

            # === 图片编辑模板 (7个) ===
            "编辑-添加和移除元素": """# Google官方模板7: 添加和移除元素
基于以下模板为用户生成添加/移除元素的编辑提示词：

模板结构：
Using the provided image of [subject], please [add/remove/modify] [element] to/from the scene. Ensure the change is [description of how the change should integrate].

请根据用户的描述和提供的图片，填充模板中的各个要素，生成一个元素编辑提示词。确保包含：
- 图片中的主体对象识别
- 明确的操作类型（添加/移除/修改）
- 要操作的具体元素
- 变化如何自然融入场景的描述
- 保持原图风格和质量的要求

请输出完整的英文提示词。""",

            "编辑-局部重绘": """# Google官方模板8: 局部重绘
基于以下模板为用户生成局部重绘提示词：

模板结构：
Using the provided image, change only the [specific element] to [new element/description]. Keep everything else in the image exactly the same, preserving the original style, lighting, and composition.

请根据用户的描述和提供的图片，填充模板中的各个要素，生成一个局部重绘提示词。确保包含：
- 要修改的具体元素识别
- 新元素的详细描述
- 强调保持其他所有元素不变
- 保持原始风格、光照和构图
- 精确的局部修改要求

请输出完整的英文提示词。""",

            "编辑-风格迁移": """# Google官方模板9: 风格迁移
基于以下模板为用户生成风格迁移提示词：

模板结构：
Transform the provided photograph of [subject] into the artistic style of [artist/art style]. Preserve the original composition but render it with [description of stylistic elements].

请根据用户的描述和提供的图片，填充模板中的各个要素，生成一个风格迁移提示词。确保包含：
- 原图主体的识别
- 目标艺术家或艺术风格
- 保持原始构图的要求
- 风格元素的详细描述
- 风格转换的具体表现

请输出完整的英文提示词。""",

            "编辑-高级合成(多图组合)": """# Google官方模板10: 高级合成（组合多张图片）
基于以下模板为用户生成多图合成提示词：

模板结构：
Create a new image by combining the elements from the provided images. Take the [element from image 1] and place it with/on the [element from image 2]. The final image should be a [description of the final scene].

请根据用户的描述和提供的多张图片，填充模板中的各个要素，生成一个多图合成提示词。确保包含：
- 从第一张图片中提取的元素
- 从第二张图片中提取的元素
- 元素组合的方式和位置
- 最终场景的详细描述
- 合成后的整体效果要求

请输出完整的英文提示词。""",

            "编辑-高保真细节保留": """# Google官方模板11: 高保真细节保留
基于以下模板为用户生成高保真细节保留提示词：

模板结构：
Using the provided images, place [element from image 2] onto [element from image 1]. Ensure that the features of [element from image 1] remain completely unchanged. The added element should [description of how the element should integrate].

请根据用户的描述和提供的图片，填充模板中的各个要素，生成一个高保真细节保留提示词。确保包含：
- 从第二张图片提取的元素
- 第一张图片中的目标位置
- 强调第一张图片特征完全不变
- 新元素的自然集成方式
- 细节保护的具体要求

请输出完整的英文提示词。""",

            "编辑-让事物焕发活力": """# Google官方模板12: 让事物焕发活力
基于以下模板为用户生成草图优化提示词：

模板结构：
Turn this rough [medium] sketch of a [subject] into a [style description] photo. Keep the [specific features] from the sketch but add [new details/materials].

请根据用户的描述和提供的草图，填充模板中的各个要素，生成一个草图优化提示词。确保包含：
- 草图媒介类型（pencil, charcoal, digital等）
- 草图主体对象
- 目标风格描述
- 要保留的具体特征
- 要添加的新细节和材质
- 从草图到成品的转换要求

请输出完整的英文提示词。""",

            "编辑-角色一致性(360度)": """# Google官方模板13: 角色一致性（360度全景）
基于以下模板为用户生成角色一致性提示词：

模板结构：
A studio portrait of [person] against [background], [looking forward/in profile looking right/etc.]

请根据用户的描述和提供的角色图片，填充模板中的各个要素，生成一个角色一致性提示词。确保包含：
- 角色的详细特征描述
- 背景设定
- 具体的视角和姿势要求
- 保持角色身份一致性
- 工作室肖像的专业要求
- 不同角度下的特征保持

请输出完整的英文提示词。""",

            "编辑-Sora动漫3宫格提示词模板": """# Sora视频模板：动漫3宫格提示词生成

🎬 模板说明：
该模板用于生成三宫格（3 panel）动漫漫画/短视频内容，强调角色一致性、视觉风格统一、镜头语言专业性和叙事连贯性。

📥 输入要求：
- image: 参考图1（男主角参考）
- image_2: 参考图2（女主角参考）
- image_3: 参考图3（风格参考 - 画面风格、配色、光影等）
- user_prompt: 三个镜头的脚本信息或场景描述

📋 生成规则（严格遵循）：
1. 每个Shot保持一致的人物特征、表情风格、服装细节
2. Shot 1 (4.5秒)：场景建立和主角引入 - 建立氛围、介绍环境、引入男主角
3. Shot 2 (5.5秒)：女主角出场和情感铺垫 - 女主角出现、情感反应、动作表现
4. Shot 3 (5.0秒)：关键互动和情感高潮 - 两人互动、关键时刻、情感转折

📝 提示词格式要求：
每个Shot应包含以下结构：

[Shot N - 中文标题或描述]
Duration: X.X sec
Scene (简述): [简短的中文场景描述]
Sora Prompt (详细): [详细的英文Sora提示词，包括：
  - 角色描述和特征（必须与参考图保持一致）
  - 场景环境描写
  - 服装和外观细节
  - 光影和配色描写
  - 相机技术参数
  - 运动描写和动作]
Camera: [镜头类型] (e.g., Medium Shot, Close-up, Wide Shot)
Movement: [运动方式] (e.g., Dolly In, Tracking Shot, Static)
Depth of Field: [景深要求] (e.g., 浅景深, 深景深, etc.)
旁白: [配合画面的文案或旁白]

🎨 关键参数说明：
- Resolution: 8k resolution, photorealistic/高质感
- Style: anime, cinematic, dreamy等 (根据参考图3)
- Duration: 每个Shot的持续时间（总15秒）
- Character Consistency: 强制保持角色一致性
- Camera Movement: 使用专业术语
  * Dolly In/Out: 推/拉镜头
  * Tracking Shot: 跟拍
  * Pan: 摇镜头
  * Crane: 升降镜头
  * Static: 静镜头
- Aspect Ratio: 通常为16:9或竖版格式

✨ 角色一致性检查清单：
✓ 脸部特征、表情、眼睛颜色
✓ 发型、发色、发长
✓ 肤色、肤质
✓ 身材、身高比例
✓ 服装、配饰（除非脚本要求改变）
✓ 整体气质和风格

📌 输出示例结构：
=== T8与贞贞的邂逅 - 三宫格漫画 ===

[Shot 1 - 城市黄昏的等待]
Duration: 4.5 sec
Scene (简述): 城市黄昏，T8在复古咖啡馆外等待，手里拿着一束特别的花。
Sora Prompt (详细): 一个极度逼真的电影感镜头，主角T8站在巴黎风格的街角，身穿深色风衣...
Camera: Medium Shot (中景)
Movement: 缓慢的 Dolly In (推镜头)
Depth of Field: 浅景深 (虚化背景)
旁白: 城市的喧嚣在这一刻静止，只剩下等待的心跳声。

[Shot 2 - 邂逅的瞬间]
...

[Shot 3 - 眼神的对话]
...

🎯 提示词优化建议：
1. 使用具体而非抽象的描述
2. 包含丰富的视觉细节（颜色、纹理、光照、构图）
3. 考虑艺术风格和技术参数
4. 保持镜头语言的专业性
5. 确保三个镜头之间的故事连贯性
6. 强调角色在不同镜头中的一致性
7. 使用参考图3的风格指导所有视觉元素

请严格按照以上格式和要求生成三宫格提示词。""",

            "编辑-Sora动漫3宫格绘图提示词模板": """# Sora绘图模板：动漫3宫格绘图提示词生成

开场标题与参考声明（在正式开始 Shot 1 之前输出）：
从 user_prompt 自动提取中文主标题（主题/场景）与中文副标题（情绪/故事走向），输出：
=== [中文主标题] - [中文副标题] ===
参考图1是男主角T8，参考图2是女主角贞贞，根据下面的镜头脚本生成三宫格漫画，保持角色的特征。

📥 输入要求：
- image: 参考图1（男主角参考）
- image_2: 参考图2（女主角参考）
- image_3: 参考图3（风格参考 - 画面风格、配色、光影等）
- user_prompt: 三个镜头的脚本信息或场景描述

📋 生成规则（严格遵循）：
1. 在输出的首行给出上述参考声明，明确两位角色对应参考图。
2. 每个Shot保持一致的人物特征、表情风格、服装细节，严格对齐参考图1/2。
3. Shot 1 (4.5秒)：场景建立和主角引入 - 建立氛围、介绍环境、引入男主角
4. Shot 2 (5.5秒)：女主角出场和情感铺垫 - 女主角出现、情感反应、动作表现
5. Shot 3 (5.0秒)：关键互动和情感高潮 - 两人互动、关键时刻、情感转折

📝 提示词格式要求：
先输出标题行（自动从 user_prompt 提取；示例：=== T8与贞贞的巴黎黄昏 - 跨越时空的邂逅 ===），
然后输出参考声明一行，接着按以下结构输出三宫格内容：

[Shot N - 中文标题或描述]
Duration: X.X sec
Scene (简述): [简短的中文场景描述]
Sora Prompt (详细): [详细的英文Sora提示词，包括：
  - 角色描述和特征（必须与参考图保持一致，指明male=参考图1、female=参考图2）
  - 场景环境描写
  - 服装和外观细节
  - 光影和配色描写（可借鉴参考图3）
  - 相机技术参数
  - 运动描写和动作]
Camera: [镜头类型] (e.g., Medium Shot, Close-up, Wide Shot)
Movement: [运动方式] (e.g., Dolly In, Tracking Shot, Static)
Depth of Field: [景深要求] (e.g., 浅景深, 深景深, etc.)
旁白: [配合画面的文案或旁白]

🎨 关键参数说明：
- Resolution: 8k resolution, photorealistic/高质感
- Style: anime, cinematic, dreamy等 (根据参考图3)
- Duration: 每个Shot的持续时间（总15秒）
- Character Consistency: 强制保持角色一致性（始终遵循参考图1/2）
- Camera Movement: 使用专业术语
- Aspect Ratio: 通常为16:9或竖版格式

✨ 角色一致性检查清单：
✓ 脸部特征、表情、眼睛颜色
✓ 发型、发色、发长
✓ 肤色、肤质
✓ 身材、身高比例
✓ 服装、配饰（除非脚本要求改变）
✓ 整体气质和风格

📌 输出示例结构：
=== T8与贞贞的巴黎黄昏 - 跨越时空的邂逅 ===
参考图1是男主角T8，参考图2是女主角贞贞，根据下面的镜头脚本生成三宫格漫画，保持角色的特征

[Shot 1 - 城市黄昏的等待]
Duration: 4.5 sec
Scene (简述): 城市黄昏，T8在复古咖啡馆外等待，手里拿着一束特别的花。
Sora Prompt (详细): 一个极度逼真的电影感镜头，主角T8站在巴黎风格的街角，身穿深色风衣...
Camera: Medium Shot (中景)
Movement: 缓慢的 Dolly In (推镜头)
Depth of Field: 浅景深 (虚化背景)
旁白: 城市的喧嚣在这一刻静止，只剩下等待的心跳声。

[Shot 2 - 邂逅的瞬间]
...

[Shot 3 - 眼神的对话]
...

请严格按照以上格式与参考声明生成三宫格绘图提示词。""",

            "编辑-Sora动漫5宫格提示词模板": """# Sora视频模板：动漫5宫格提示词生成

🎬 模板说明：
该模板用于生成五宫格（5 panel）动漫漫画/短视频内容，提供更完整的故事叙述空间。强调角色一致性、视觉风格统一、镜头语言专业性、故事弧线完整性和节奏感。总时长控制在25秒。

📥 输入要求：
- image: 参考图1（男主角参考）
- image_2: 参考图2（女主角参考）
- image_3: 参考图3（风格参考 - 画面风格、配色、光影等）
- image_4: 参考图4（可选 - 场景或道具参考）
- user_prompt: 五个镜头的完整脚本或场景描述

📋 五幕结构与时间分配（总25秒）：
1. Shot 1 (4秒)：故事开场 - 建立世界观、介绍环境、铺垫气氛、引入男主角
2. Shot 2 (5秒)：男主角发展 - 展示男主角的行动、情感或准备
3. Shot 3 (5秒)：女主角登场 - 女主角出现、首次互动、情感转折
4. Shot 4 (5.5秒)：关键互动 - 两人的主要互动、冲突或和解
5. Shot 5 (5.5秒)：故事高潮与结局 - 情感高潮、结局揭示、余韵留白

📝 提示词格式要求：
每个Shot应包含以下结构：

[Shot N - 中文标题或描述]
Duration: X.X sec
Scene (简述): [简短的中文场景描述]
Sora Prompt (详细): [详细的英文Sora提示词，包括：
  - 角色描述和特征（必须与参考图保持一致）
  - 场景环境描写
  - 服装和外观细节
  - 光影和配色描写
  - 相机技术参数
  - 运动描写和动作]
Camera: [镜头类型] (e.g., Medium Shot, Close-up, Wide Shot)
Movement: [运动方式] (e.g., Dolly In, Tracking Shot, Static)
Depth of Field: [景深要求] (e.g., 浅景深, 深景深)
旁白: [配合画面的文案或旁白]

🎨 关键参数说明：
- Resolution: 8k resolution, photorealistic/高质感
- Style: anime, cinematic, dreamy等 (根据参考图3)
- Total Duration: 25秒 = 4 + 5 + 5 + 5.5 + 5.5秒
- Character Consistency: 强制保持角色一致性（跨5个镜头）
- Story Arc: 完整的故事弧线 (开场→发展→转折→高潮→结局)
- Camera Movement: 使用专业术语
  * Dolly In/Out: 推/拉镜头
  * Tracking Shot: 跟拍
  * Pan: 摇镜头
  * Crane: 升降镜头
  * Static: 静镜头
- Aspect Ratio: 通常为16:9或竖版格式
- Pacing: 递进式节奏，逐步推向高潮

✨ 角色一致性检查清单（贯穿全5个镜头）：
✓ 脸部特征、表情、眼睛颜色（保持一致）
✓ 发型、发色、发长（除非脚本要求变化）
✓ 肤色、肤质（光影变化不改变肤质基调）
✓ 身材、身高比例
✓ 服装、配饰（首出镜时确立，后续保持）
✓ 整体气质和风格（人物性格一致）
✓ 角色间关系的递进发展

📌 五幕故事结构参考：
- 幕1（开场）：设置环境，介绍人物
- 幕2（发展1）：展示主要人物的目标或困境
- 幕3（中点转折）：第二个人物出现，改变局势
- 幕4（高潮冲突）：两个人物的主要互动或冲突
- 幕5（结局与余韵）：情感释放，故事完成

🎯 视觉节奏建议：
1. Shot 1：广角建立 (Wide/Establishing Shot)
2. Shot 2：中景跟随 (Medium Shot + Movement)
3. Shot 3：转向性镜头 (Turning Point Visual)
4. Shot 4：亲密互动 (Close-up or Over-the-shoulder)
5. Shot 5：高潮释放 (Climactic Visual + Resolution)

📊 时间分配的黄金比例：
- 开场 (Shot 1): 16% = 4秒
- 发展 (Shot 2): 20% = 5秒
- 转折 (Shot 3): 20% = 5秒
- 高潮 (Shot 4): 22% = 5.5秒
- 结局 (Shot 5): 22% = 5.5秒
总计: 100% = 25秒

💡 创意指导：
- 每个Shot之间应有视觉衔接
- 角色的情绪变化应有清晰的视觉表现
- 光影和配色应随故事进展而变化
- 使用不同的镜头类型创造视觉多样性
- 在最后两个镜头中达到情感高潮

请严格按照以上格式和要求生成五宫格提示词。确保故事完整、角色一致、节奏递进。""",

            "编辑-Sora动漫5宫格绘图提示词模板": """# Sora绘图模板：动漫5宫格绘图提示词生成

开场标题与参考声明（在正式开始 Shot 1 之前输出）：
从 user_prompt 自动提取中文主标题（主题/场景）与中文副标题（情绪/故事走向），输出：
=== [中文主标题] - [中文副标题] ===
参考图1是男主角T8，参考图2是女主角贞贞，根据下面的镜头脚本生成五宫格漫画，保持角色的特征。

📥 输入要求：
- image: 参考图1（男主角参考）
- image_2: 参考图2（女主角参考）
- image_3: 参考图3（风格参考 - 画面风格、配色、光影等）
- image_4: 参考图4（可选 - 场景或道具参考）
- user_prompt: 五个镜头的完整脚本或场景描述

📋 五幕结构与时间分配（总25秒）：
1. Shot 1 (4秒)：故事开场 - 建立世界观、介绍环境、铺垫气氛、引入男主角
2. Shot 2 (5秒)：男主角发展 - 展示男主角的行动、情感或准备
3. Shot 3 (5秒)：女主角登场 - 女主角出现、首次互动、情感转折
4. Shot 4 (5.5秒)：关键互动 - 两人的主要互动、冲突或和解
5. Shot 5 (5.5秒)：故事高潮与结局 - 情感高潮、结局揭示、余韵留白

📝 提示词格式要求：
先输出标题行（自动从 user_prompt 提取；示例：=== T8与贞贞的巴黎黄昏 - 跨越时空的邂逅 ===），
然后输出参考声明一行，接着按以下结构输出五宫格内容：

[Shot N - 中文标题或描述]
Duration: X.X sec
Scene (简述): [简短的中文场景描述]
Sora Prompt (详细): [详细的英文Sora提示词，包括：
  - 角色描述和特征（必须与参考图保持一致，指明male=参考图1、female=参考图2）
  - 场景环境描写
  - 服装和外观细节
  - 光影和配色描写（可借鉴参考图3/参考图4）
  - 相机技术参数
  - 运动描写和动作]
Camera: [镜头类型] (e.g., Wide, Medium, Close-up)
Movement: [运动方式] (e.g., Dolly In/Out, Tracking Shot, Pan, Crane, Static)
Depth of Field: [景深要求] (e.g., 浅景深, 深景深)
旁白: [配合画面的文案或旁白]

🎨 关键参数说明：
- Resolution: 8k resolution, photorealistic/高质感
- Style: anime, cinematic, dreamy等 (根据参考图3)
- Total Duration: 25秒 = 4 + 5 + 5 + 5.5 + 5.5秒
- Character Consistency: 强制保持角色一致性（始终遵循参考图1/2）
- Story Arc: 完整的故事弧线 (开场→发展→转折→高潮→结局)
- Camera Movement: 使用专业术语（Dolly/Tracking/Pan/Crane/Static）
- Aspect Ratio: 通常为16:9或竖版格式
- Pacing: 递进式节奏，逐步推向高潮

✨ 角色一致性检查清单：
✓ 脸部特征、表情、眼睛颜色（保持一致）
✓ 发型、发色、发长（除非脚本要求变化）
✓ 肤色、肤质（光影变化不改变肤质基调）
✓ 身材、身高比例
✓ 服装、配饰（首出镜时确立，后续保持）
✓ 整体气质和风格（人物性格一致）
✓ 角色间关系的递进发展

📌 输出示例结构：
=== T8与贞贞的巴黎黄昏 - 跨越时空的邂逅 ===
参考图1是男主角T8，参考图2是女主角贞贞，根据下面的镜头脚本生成五宫格漫画，保持角色的特征

[Shot 1 - 城市的秘密]
Duration: 4 sec
Scene (简述): 巴黎街角的黄昏，复古咖啡馆前的环境建立，T8现身。
Sora Prompt (详细): Wide establishing shot ... male aligned to reference image 1 ... lighting, composition, camera specs ...
Camera: Wide Shot
Movement: Static → 缓慢 Push In
Depth of Field: 深景深
旁白: 在这座城市最美的时刻，他做出了最重要的决定。

[Shot 2 - 等待的心跳]
...

[Shot 3 - 邂逅的瞬间]
...

[Shot 4 - 眼神的交汇]
...

[Shot 5 - 余晖与答案]
...

请严格按照以上格式与参考声明生成五宫格绘图提示词。确保故事完整、角色一致、节奏递进。""",

            # === 扩展创意模板 ===
            "创意-电影级场景": """# 扩展创意模板1: 电影级场景
基于电影摄影技法为用户生成电影级场景提示词：

模板结构：
A cinematic [shot type] of [subject] in [dramatic situation], captured with [camera movement] through [environment]. The scene features [lighting technique] creating [visual mood]. Shot on [film stock/camera], with [color grading] and [depth of field]. [Aspect ratio, preferably 2.35:1 or 16:9].

请根据用户的描述，填充模板中的各个要素，生成一个电影级场景提示词。确保包含：
- 电影镜头类型（establishing shot, close-up, tracking shot等）
- 戏剧性情境
- 相机运动（dolly, crane, handheld等）
- 电影级光照技术
- 视觉情绪营造
- 胶片/相机规格
- 调色风格
- 景深效果
- 电影画幅比例

请输出完整的英文提示词。""",

            "创意-概念艺术设计": """# 扩展创意模板2: 概念艺术设计
基于概念艺术创作流程为用户生成概念设计提示词：

模板结构：
A detailed concept art of [subject/character/environment] for [project type]. The design features [key design elements] with [artistic technique]. Color palette: [color scheme]. The style is [art style] with [level of detail]. Include [technical annotations/callouts] showing [specific features].

请根据用户的描述，填充模板中的各个要素，生成一个概念艺术提示词。确保包含：
- 概念对象（角色/环境/道具等）
- 项目类型（游戏/电影/动画等）
- 关键设计元素
- 艺术技法
- 色彩方案
- 艺术风格
- 细节层次
- 技术标注
- 特定功能展示

请输出完整的英文提示词。""",

            "创意-时尚摄影": """# 扩展创意模板3: 时尚摄影
基于时尚摄影标准为用户生成时尚摄影提示词：

模板结构：
A high-fashion editorial photograph of [model description] wearing [clothing/accessories], posed in [pose description]. Shot in [location/studio setup] with [lighting setup]. The styling features [fashion elements] in [color palette]. Photographed with [camera/lens] for [magazine/brand]. [Aspect ratio].

请根据用户的描述，填充模板中的各个要素，生成一个时尚摄影提示词。确保包含：
- 模特描述
- 服装和配饰
- 姿势描述
- 拍摄地点/工作室设置
- 专业灯光设置
- 时尚元素
- 色彩搭配
- 相机镜头规格
- 目标杂志/品牌风格
- 画面比例

请输出完整的英文提示词。""",

            "创意-建筑可视化": """# 扩展创意模板4: 建筑可视化
基于建筑可视化标准为用户生成建筑渲染提示词：

模板结构：
An architectural visualization of [building type] featuring [architectural style] design. The structure showcases [key architectural elements] with [materials]. Set in [environment/context] during [time of day]. Rendered with [rendering style] showing [lighting conditions] and [atmospheric effects]. [Camera angle] perspective.

请根据用户的描述，填充模板中的各个要素，生成一个建筑可视化提示词。确保包含：
- 建筑类型
- 建筑风格
- 关键建筑元素
- 建筑材料
- 环境背景
- 时间设定
- 渲染风格
- 光照条件
- 大气效果
- 相机视角

请输出完整的英文提示词。""",

            "创意-食物摄影": """# 扩展创意模板5: 食物摄影
基于专业食物摄影技法为用户生成美食摄影提示词：

模板结构：
A professional food photography shot of [dish/ingredient] presented on [plating/surface]. The composition features [arrangement style] with [garnish/props]. Lit with [lighting technique] to highlight [texture/color]. Shot from [angle] with [depth of field]. The color palette is [warm/cool/vibrant] creating an [appetite appeal]. [Aspect ratio].

请根据用户的描述，填充模板中的各个要素，生成一个食物摄影提示词。确保包含：
- 菜品或食材
- 摆盘和表面
- 构图风格
- 装饰和道具
- 专业灯光技术
- 质感和色彩强调
- 拍摄角度
- 景深效果
- 色彩基调
- 食欲吸引力
- 画面比例

请输出完整的英文提示词。""",

            "创意-抽象艺术": """# 扩展创意模板6: 抽象艺术
基于抽象艺术创作理念为用户生成抽象艺术提示词：

模板结构：
An abstract [art style] composition exploring [theme/concept] through [visual elements]. The piece features [shapes/forms] in [color palette] with [texture/pattern]. The composition creates [visual rhythm/movement] using [technique]. [Emotional/conceptual impact].

请根据用户的描述，填充模板中的各个要素，生成一个抽象艺术提示词。确保包含：
- 抽象艺术风格（geometric, organic, expressionist等）
- 主题或概念
- 视觉元素
- 形状和形式
- 色彩方案
- 质感和图案
- 视觉节奏或运动感
- 创作技法
- 情感或概念冲击

请输出完整的英文提示词。""",

            "创意-儿童插画": """# 扩展创意模板7: 儿童插画
基于儿童插画设计原则为用户生成儿童插画提示词：

模板结构：
A children's book illustration of [character/scene] in a [art style] style. The image features [friendly/whimsical elements] with [bright/soft colors]. The composition is [simple/detailed] and [age-appropriate]. The mood is [cheerful/educational/adventurous]. Perfect for [age group] readers.

请根据用户的描述，填充模板中的各个要素，生成一个儿童插画提示词。确保包含：
- 角色或场景
- 插画风格（watercolor, digital, hand-drawn等）
- 友好或奇幻元素
- 明亮或柔和的色彩
- 构图复杂度
- 年龄适宜性
- 情绪基调
- 目标年龄群体
- 教育或娱乐价值

请输出完整的英文提示词。""",

            "创意-海报设计": """# 扩展创意模板8: 创意海报设计（支持中英文）
基于专业海报设计原则为用户生成创意海报提示词：

模板结构：
A creative poster design for [event/product/campaign] featuring [main visual element]. The poster includes the headline text "[主标题/Main Title]" in [font style] and tagline "[副标题或宣传语/Tagline]". The design style is [design style] with [color scheme] color palette. The layout is [layout type] with [visual hierarchy]. Additional text elements include "[其他文字信息/Additional text]". The overall mood is [mood/emotion]. [Aspect ratio, typically vertical like 2:3 or 3:4].

重要提示：
1. 支持中英文混合文字，请在引号中明确标注要显示的文字内容
2. 中文文字请用中文书写，英文文字请用英文书写
3. 确保文字内容清晰、准确、易读
4. 文字排版要符合视觉层级

请根据用户的描述，填充模板中的各个要素，生成一个创意海报提示词。确保包含：
- 海报用途（活动/产品/宣传活动等）
- 主视觉元素
- 主标题文字（支持中英文）
- 副标题或宣传语（支持中英文）
- 字体风格
- 设计风格（minimalist, vintage, modern, grunge等）
- 色彩方案
- 版式类型（centered, asymmetric, grid-based等）
- 视觉层级
- 其他文字信息（日期、地点、联系方式等）
- 整体情绪氛围
- 画面比例（通常为竖版）

文字处理示例：
- 中文标题："春节大促销"
- 英文标题："SPRING SALE"
- 中英混合："新年快乐 HAPPY NEW YEAR"
- 日期信息："2024.01.01" 或 "January 1st, 2024"

请输出完整的提示词，确保所有文字内容都准确标注在引号中。""",

            "自定义模板": custom_template
        }

        return templates.get(template_type, custom_template)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "KenChenLLMBanana2PromptTemplate": KenChenLLMBanana2PromptTemplateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KenChenLLMBanana2PromptTemplate": "Banana2-提示词模板",
}
