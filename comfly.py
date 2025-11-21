import os
import json
import random
import requests
import base64
import io
import torch
import numpy as np
from PIL import Image

# --- 全局常量和配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHATFLY_CONFIG_FILE_NAME = 'ChatFly_config.json'
IMAGE_PROMPTS_FILE_NAME = 'image_prompts.txt'
TRANSITION_PROMPTS_FILE_NAME = 'transition_prompts.txt'

# --- 辅助函数 ---
def _log_info(message):
    print(f"[LLM Prompt] 信息：{message}")

def _log_warning(message):
    print(f"[LLM Prompt] 警告：{message}")

def _log_error(message):
    print(f"[LLM Prompt] 错误：{message}")

def process_input_image(image_tensor):
    """将 ComfyUI 的 IMAGE tensor 转换为 PIL Image"""
    try:
        # 处理 4D tensor (batch)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # 取第一张图片

        # 转换 CHW 到 HWC 格式
        if image_tensor.shape[0] in [1, 3, 4]:  # CHW 格式
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:  # 已经是 HWC 格式
            image_np = image_tensor.cpu().numpy()

        # 归一化并转换为 uint8
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        image_np = (image_np * 255).astype(np.uint8)

        # 处理通道
        if len(image_np.shape) == 3:
            if image_np.shape[2] == 1:
                image_np = np.repeat(image_np, 3, axis=2)  # 灰度转 RGB
            elif image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]  # RGBA 转 RGB
        elif len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)  # 灰度转 RGB

        pil_image = Image.fromarray(image_np)
        return pil_image
    except Exception as e:
        _log_error(f"处理输入图片失败: {e}")
        return None

def image_to_base64(image, format='JPEG', quality=95):
    """将 PIL Image 转换为 base64 字符串"""
    try:
        buffer = io.BytesIO()

        # 处理 JPEG 的 alpha 通道
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
        _log_error(f"转换图片为 base64 失败: {e}")
        return None

def get_chatfly_config():
    """
    尝试从同目录下的 ChatFly_config.json 文件中读取 ChatFly 的配置。
    返回一个字典，包含 ChatFly 的 bot_id, session_id, token。
    如果文件不存在或格式不正确，则返回一个空字典。
    """
    config_path = os.path.join(CURRENT_DIR, CHATFLY_CONFIG_FILE_NAME)
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        else:
            return {}
    except json.JSONDecodeError:
        return {}
    except Exception as e:
        return {}

def get_prompt_api_providers():
    """
    从ChatFly_config.json中获取提示词扩写API提供者配置
    """
    config_path = os.path.join(CURRENT_DIR, 'ChatFly_config.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 从配置文件中读取prompt_api_providers
            if "prompt_api_providers" in config:
                return config["prompt_api_providers"]
            else:
                return {
                    "Comfly": {
                        "url": "https://ai.comfly.chat/v1",
                        "api_key": config.get("api_key", ""),
                        "api_format": "openai",
                        "models": ["gpt-4o", "gpt-4-v", "claude-sonnet-4-20250514"],
                        "description": "Comfly AI镜像站"
                    }
                }
        else:
            return {}
    except json.JSONDecodeError:
        return {}
    except Exception as e:
        return {}

def get_provider_config(provider_name):
    """
    根据提供者名称获取配置
    """
    providers = get_prompt_api_providers()
    if provider_name not in providers:
        return {}

    return providers[provider_name]

def load_prompts_from_txt(file_path, default_built_in_prompts):
    """
    从特定格式的TXT文件加载多个提示词。
    格式要求：每个提示词以 `[提示词名称]` 开头，内容在其后，直到下一个 `[` 开头或文件结束。
    空行和行首行尾的空格会被去除。
    """
    prompts = {}
    current_prompt_name = None
    current_prompt_content = []

    if not os.path.exists(file_path):
        return default_built_in_prompts

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip() # 移除行首行尾空白
                if not line: # 跳过空行
                    continue

                if line.startswith('[') and line.endswith(']'):
                    # 新的提示词名称
                    if current_prompt_name and current_prompt_content:
                        prompts[current_prompt_name] = "\n".join(current_prompt_content).strip()

                    current_prompt_name = line[1:-1].strip() # 提取名称
                    current_prompt_content = [] # 重置内容
                elif current_prompt_name is not None:
                    # 添加内容到当前提示词
                    current_prompt_content.append(line)
                # else: 忽略文件开头在第一个 [ ] 之前的行

            # 处理文件末尾的最后一个提示词
            if current_prompt_name and current_prompt_content:
                prompts[current_prompt_name] = "\n".join(current_prompt_content).strip()

        if not prompts:
            return default_built_in_prompts

        return prompts

    except Exception as e:
        return default_built_in_prompts

# --- Comfly专用节点 ---
class Comfly_Prompt_Expand_From_Image:
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("expanded_prompt",)
    FUNCTION = "expand_prompt"

    # 内置的默认识图提示词 (当TXT文件不存在或解析失败时作为备用)
    _BUILT_IN_IMAGE_PROMPTS = {
        "通用高质量英文描述 (内置)": "你是一个专业的图像描述专家，能够将图片内容转化为高质量的英文提示词，用于文本到图像的生成模型。请仔细观察提供的图片，并生成一段详细、具体、富有创造性的英文短语，描述图片中的主体对象、场景、动作、光线、材质、色彩、构图和艺术风格。要求：语言：严格使用英文。细节：尽可能多地描绘图片细节，包括但不限于物体、人物、背景、前景、纹理、表情、动作、服装、道具等。角度：尽可能从多个角度丰富描述，例如特写、广角、俯视、仰视等，但不要直接写\"角度\"。连接：使用逗号（,）连接不同的短语，形成一个连贯的提示词。人物：描绘人物时，使用第三人称（如 'a woman', 'the man'）。质量词：在生成的提示词末尾，务必添加以下质量增强词：', best quality, high resolution, 4k, high quality, masterpiece, photorealistic'"
    }

    @classmethod
    def get_image_prompts(cls):
        """加载外部或内置的图像提示词字典。"""
        return load_prompts_from_txt(
            os.path.join(CURRENT_DIR, IMAGE_PROMPTS_FILE_NAME),
            cls._BUILT_IN_IMAGE_PROMPTS
        )

    @classmethod
    def get_comfly_config(cls):
        return get_chatfly_config()

    @classmethod
    def INPUT_TYPES(cls):
        available_prompts = cls.get_image_prompts()
        prompt_keys = list(available_prompts.keys())
        default_selection = prompt_keys[0] if prompt_keys else "无可用提示词"

        # 获取所有API提供者
        providers = get_prompt_api_providers()
        provider_names = list(providers.keys())
        default_provider = provider_names[0] if provider_names else "Comfly"

        # 合并所有提供者的模型列表（去重）
        all_models = []
        seen_models = set()
        for provider_name, provider_info in providers.items():
            models = provider_info.get("models", [])
            for model in models:
                if model not in seen_models:
                    all_models.append(model)
                    seen_models.add(model)

        # 如果没有模型，使用默认值
        if not all_models:
            all_models = ["gpt-4o", "gpt-4-v", "claude-sonnet-4-20250514"]

        return {
            "required": {
                "api_provider": (provider_names, {"default": default_provider, "label": "API提供者 API Provider"}),
                "image_prompt_preset": (prompt_keys, {"default": default_selection, "label": "图像提示词预设 Image Prompt Preset"}),
                "base_url": ("STRING", {"multiline": False, "default": "", "placeholder": "API地址将自动根据提供者选择（可手动覆盖）", "label": "API地址 API Base URL"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "API密钥 (API Key)", "label": "API密钥 API Key"}),
                "model": (all_models, {"default": all_models[0] if all_models else "gpt-4o", "label": "模型 Model"}),
                "system_prompt": ("STRING", {"multiline": True, "default": available_prompts.get(default_selection, ""), "placeholder": "系统提示词（可自定义专家角色，支持中文） System prompt (custom expert role, supports Chinese)", "label": "系统提示词 System Prompt"}),
                "user_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入你的原始提示词（支持中文）Enter your original prompt (supports Chinese)", "label": "用户提示词 User Prompt"}),
                "user_requirement": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入你的额外要求（可选，支持中文）Enter your extra requirements (optional, supports Chinese)", "label": "额外要求 Extra Requirement"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "label": "采样温度 Temperature"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "label": "随机种子 Seed"})
            },
            "optional": {
                "image": ("IMAGE", {"label": "参考图片 Reference Image"}),
                "ref_image": ("STRING", {"multiline": True, "default": "", "placeholder": "Base64编码图片（可选，优先使用image输入）", "label": "Base64图片 Base64 Image"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "label": "采样概率 Top-p"}),
                "max_tokens": ("INT", {"default": 400, "min": 1, "max": 4096, "label": "最大Token数 Max Tokens"}),
                "image_url": ("STRING", {"multiline": True, "default": "", "placeholder": "可填写图片Base64或图片URL (Base64 or image URL)", "label": "image_url"})
            }
        }

    def expand_prompt(self, api_provider, image_prompt_preset, base_url, api_key, model, system_prompt, user_prompt, user_requirement, temperature=0.7, seed=0, image=None, ref_image="", top_p=0.8, max_tokens=400, image_url=""):
        import requests

        # 根据API提供者获取配置
        config = get_provider_config(api_provider)

        # 从配置中获取URL和API key
        final_base_url = base_url.strip() or config.get("url", "")
        final_api_key = api_key.strip() or config.get("api_key", "")

        if not final_base_url or not final_api_key:
            return (f"未检测到API Key或Base URL，请在节点输入框填写，或在ChatFly_config.json的prompt_api_providers中配置{api_provider}的url和api_key。\nAPI Key or Base URL not found. Please fill in the node input box, or configure url and api_key for {api_provider} in ChatFly_config.json's prompt_api_providers.",)
        api_url = final_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {final_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        available_prompts = self.get_image_prompts()
        preset_prompt = available_prompts.get(image_prompt_preset, "")
        final_system_prompt = system_prompt.strip() or preset_prompt
        content_parts = []
        if final_system_prompt:
            content_parts.append({"type": "text", "text": final_system_prompt})
        if user_prompt.strip():
            content_parts.append({"type": "text", "text": user_prompt.strip()})
        if user_requirement.strip():
            content_parts.append({"type": "text", "text": user_requirement.strip()})

        # 处理图片输入（优先级：image > image_url > ref_image）
        image_base64 = None
        has_image = False

        # 1. 优先使用 IMAGE 类型输入
        if image is not None:
            pil_image = process_input_image(image)
            if pil_image:
                image_base64 = image_to_base64(pil_image, format='JPEG', quality=95)
                if image_base64:
                    has_image = True

        # 2. 其次使用 image_url
        elif image_url and image_url.strip():
            url_val = image_url.strip()
            if url_val.startswith("http://") or url_val.startswith("https://"):
                content_parts.append({"type": "image_url", "image_url": {"url": url_val}})
                has_image = True
            else:
                # 验证 base64 数据的有效性
                try:
                    base64.b64decode(url_val, validate=True)
                    image_base64 = url_val
                    has_image = True
                except Exception:
                    pass  # 无效的base64，忽略

        # 3. 最后使用 ref_image
        elif ref_image and ref_image.strip():
            ref_image_val = ref_image.strip()
            # 验证 base64 数据的有效性
            try:
                base64.b64decode(ref_image_val, validate=True)
                image_base64 = ref_image_val
                has_image = True
            except Exception:
                pass  # 无效的base64，忽略

        # 添加 base64 图片到内容
        if image_base64:
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

        messages = [{"role": "user", "content": content_parts}]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=60)

            resp.raise_for_status()
            data = resp.json()
            expanded_prompt = data["choices"][0]["message"]["content"]
            return (expanded_prompt,)
        except requests.exceptions.HTTPError as e:
            error_message = f"{api_provider} API HTTP错误: {e}\n状态码: {resp.status_code}"
            if resp.status_code == 500:
                error_message += "\n💡 提示: 500错误通常是服务器内部错误，可能原因："
                error_message += "\n   1. 模型不支持图片输入（请尝试支持视觉的模型，如 gpt-4-v, gemini-2.5-flash 等）"
                error_message += "\n   2. 图片base64数据格式问题"
                error_message += "\n   3. 请求体格式不符合API要求"
                error_message += "\n   4. API服务暂时不可用"
                try:
                    error_detail = resp.json()
                    error_message += f"\n   服务器返回: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
                except:
                    pass
            _log_error(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"{api_provider} API 调用失败: {e}\n{api_provider} API call failed: {e}"
            _log_error(error_message)
            return (error_message,)

# --- 首尾帧提示词生成节点 ---
class Comfly_First_Last_Frame_Prompt:
    """
    首尾帧过渡提示词生成节点
    根据首帧和尾帧图片生成描述整个过渡过程的高质量提示词
    用于视频生成节点生成首帧到尾帧的连续性视频
    """
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transition_prompt",)
    FUNCTION = "generate_transition_prompt"

    # 首尾帧过渡提示词系统提示词
    _FIRST_LAST_FRAME_SYSTEM_PROMPT = """你是一个专业的视频过渡分析和提示词生成专家。你的任务是分析视频的首帧和尾帧图片，生成一个高质量的英文提示词，用于描述从首帧到尾帧的整个过渡过程和动态变化。

【核心任务】
根据首帧（初始状态）和尾帧（最终状态）两张图片，生成一个单一的、连贯的、描述整个过渡过程的提示词。这个提示词将用于视频生成模型，生成从首帧平滑过渡到尾帧的连续性视频。

【首帧分析】
首帧是视频的开始画面，代表初始状态。请分析：
- 主体对象的初始状态、位置、姿态和外观
- 场景的初始环境、背景、光线和氛围
- 色彩方案和视觉风格
- 相机角度和构图
- 所有关键的视觉元素

【尾帧分析】
尾帧是视频的结束画面，代表最终状态。请分析：
- 主体对象的最终状态、位置、姿态和外观
- 场景的最终环境、背景、光线和氛围
- 色彩方案和视觉风格
- 相机角度和构图
- 所有关键的视觉元素

【过渡分析】
比较首帧和尾帧，识别所有的变化和过渡：
- 主体对象的运动方向和轨迹
- 主体对象的形态、大小或外观的变化
- 场景背景的变化
- 光线、色彩、氛围的演变
- 相机的运动（平移、缩放、旋转等）
- 整个视频的动态节奏和流畅性

【提示词生成规则】
1. 语言：严格使用英文
2. 格式：使用逗号（,）连接不同的短语，形成一个连贯的、流畅的提示词
3. 时间性：使用动词和动作词汇来描述过程和变化，例如：
   - "transitioning from ... to ..."
   - "gradually changing from ... to ..."
   - "smoothly moving from ... to ..."
   - "evolving from ... to ..."
4. 连贯性：确保提示词描述的是一个连续的、平滑的过渡过程，而不是两个独立的状态
5. 细节：详细描绘：
   - 主体对象的运动和变化
   - 背景和环境的演变
   - 光线、色彩、氛围的过渡
   - 相机的运动（如果有）
   - 整个过程的节奏和流畅性
6. 人物描述：使用第三人称（如 'a woman', 'the man', 'a person'）
7. 质量词：在提示词末尾，务必添加以下质量增强词：
   ', best quality, high resolution, 4k, high quality, masterpiece, photorealistic, smooth transition, seamless motion'

【输出格式】
生成一个单一的、完整的、高质量的英文提示词，用于描述从首帧到尾帧的整个过渡过程。这个提示词应该能够指导视频生成模型生成平滑、连贯的过渡视频。

【示例】
首帧：一个人站在房间的左边，光线昏暗
尾帧：同一个人站在房间的右边，光线明亮
输出提示词：a person smoothly walking from the left side to the right side of a room, transitioning from dim lighting to bright lighting, the camera follows the movement, the background gradually becomes brighter, best quality, high resolution, 4k, high quality, masterpiece, photorealistic, smooth transition, seamless motion"""

    @classmethod
    def get_comfly_config(cls):
        return get_chatfly_config()

    @classmethod
    def INPUT_TYPES(cls):
        # 获取所有API提供者
        providers = get_prompt_api_providers()
        provider_names = list(providers.keys())
        default_provider = provider_names[0] if provider_names else "Comfly"

        # 合并所有提供者的模型列表（去重）
        all_models = []
        seen_models = set()
        for provider_name, provider_info in providers.items():
            models = provider_info.get("models", [])
            for model in models:
                if model not in seen_models:
                    all_models.append(model)
                    seen_models.add(model)

        # 如果没有模型，使用默认值
        if not all_models:
            all_models = ["gpt-4o", "gpt-4-v", "claude-sonnet-4-20250514"]

        return {
            "required": {
                "api_provider": (provider_names, {"default": default_provider, "label": "API提供者 API Provider"}),
                "base_url": ("STRING", {"multiline": False, "default": "", "placeholder": "API地址将自动根据提供者选择（可手动覆盖）", "label": "API地址 API Base URL"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "API密钥 (API Key)", "label": "API密钥 API Key"}),
                "model": (all_models, {"default": all_models[0] if all_models else "gpt-4o", "label": "模型 Model"}),
                "first_frame": ("IMAGE", {"label": "首帧图片 First Frame Image"}),
                "last_frame": ("IMAGE", {"label": "尾帧图片 Last Frame Image"}),
                "system_prompt": ("STRING", {"multiline": True, "default": cls._FIRST_LAST_FRAME_SYSTEM_PROMPT, "placeholder": "系统提示词（可自定义，支持中文）System prompt (customizable, supports Chinese)", "label": "系统提示词 System Prompt"}),
                "user_requirement": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入你的额外要求（可选，支持中文）Enter your extra requirements (optional, supports Chinese)", "label": "额外要求 Extra Requirement"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "label": "采样温度 Temperature"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "label": "随机种子 Seed"})
            },
            "optional": {
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "label": "采样概率 Top-p"}),
                "max_tokens": ("INT", {"default": 800, "min": 1, "max": 4096, "label": "最大Token数 Max Tokens"})
            }
        }

    def generate_transition_prompt(self, api_provider, base_url, api_key, model, first_frame, last_frame, system_prompt, user_requirement, temperature=0.7, seed=0, top_p=0.8, max_tokens=600):
        import requests

        # 根据API提供者获取配置
        config = get_provider_config(api_provider)
        _log_info(f"使用API提供者: {api_provider}")

        # 从配置中获取URL和API key
        final_base_url = base_url.strip() or config.get("url", "")
        final_api_key = api_key.strip() or config.get("api_key", "")

        if not final_base_url or not final_api_key:
            return (f"未检测到API Key或Base URL，请在节点输入框填写，或在ChatFly_config.json的prompt_api_providers中配置{api_provider}的url和api_key。\nAPI Key or Base URL not found. Please fill in the node input box, or configure url and api_key for {api_provider} in ChatFly_config.json's prompt_api_providers.",)

        api_url = final_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {final_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # 处理首帧图片
        first_frame_pil = process_input_image(first_frame)
        if not first_frame_pil:
            return ("首帧图片处理失败 Failed to process first frame image",)
        first_frame_base64 = image_to_base64(first_frame_pil, format='JPEG', quality=95)
        if not first_frame_base64:
            return ("首帧图片转换为base64失败 Failed to convert first frame to base64",)

        # 处理尾帧图片
        last_frame_pil = process_input_image(last_frame)
        if not last_frame_pil:
            return ("尾帧图片处理失败 Failed to process last frame image",)
        last_frame_base64 = image_to_base64(last_frame_pil, format='JPEG', quality=95)
        if not last_frame_base64:
            return ("尾帧图片转换为base64失败 Failed to convert last frame to base64",)

        # 构建内容
        content_parts = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": "【首帧图片】\n请分析以下首帧图片（视频的开始画面）："},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_frame_base64}"}},
            {"type": "text", "text": "【尾帧图片】\n请分析以下尾帧图片（视频的结束画面）："},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_frame_base64}"}},
            {"type": "text", "text": "请生成一个单一的、高质量的英文提示词，用于描述从首帧到尾帧的整个过渡过程。这个提示词将用于视频生成模型，生成从首帧平滑过渡到尾帧的连续性视频。"}
        ]

        if user_requirement.strip():
            content_parts.append({"type": "text", "text": f"【额外要求】\n{user_requirement.strip()}"})

        messages = [{"role": "user", "content": content_parts}]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=60)

            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]

            # 返回过渡提示词
            return (response_text,)

        except requests.exceptions.HTTPError as e:
            error_message = f"{api_provider} API HTTP错误: {e}\n状态码: {resp.status_code}"
            if resp.status_code == 500:
                error_message += "\n💡 提示: 500错误通常是服务器内部错误，可能原因："
                error_message += "\n   1. 模型不支持图片输入（请尝试支持视觉的模型，如 gpt-4-v 等）"
                error_message += "\n   2. 图片base64数据格式问题"
                error_message += "\n   3. 请求体格式不符合API要求"
                error_message += "\n   4. API服务暂时不可用"
                try:
                    error_detail = resp.json()
                    error_message += f"\n   服务器返回: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
                except:
                    pass
            _log_error(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"{api_provider} API 调用失败: {e}\n{api_provider} API call failed: {e}"
            _log_error(error_message)
            return (error_message,)


# --- 注册节点 ---
NODE_CLASS_MAPPINGS = {
    "Comfly_Prompt_Expand_From_Image": Comfly_Prompt_Expand_From_Image,
    "Comfly_First_Last_Frame_Prompt": Comfly_First_Last_Frame_Prompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfly_Prompt_Expand_From_Image": "扩写高质量提示词 (Comfly/T8)",
    "Comfly_First_Last_Frame_Prompt": "首尾帧过渡提示词生成 (Comfly/T8)",
}