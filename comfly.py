import os
import json
import random
import requests

# --- 全局常量和配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHATFLY_CONFIG_FILE_NAME = 'ChatFly_config.json'
IMAGE_PROMPTS_FILE_NAME = 'image_prompts.txt'

# --- 辅助函数 ---
def _log_info(message):
    print(f"[LLM Prompt] 信息：{message}")

def _log_warning(message):
    print(f"[LLM Prompt] 警告：{message}")

def _log_error(message):
    print(f"[LLM Prompt] 错误：{message}")

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
            _log_warning(f"未找到 ChatFly 配置文件 {CHATFLY_CONFIG_FILE_NAME}。")
            return {}
    except json.JSONDecodeError:
        _log_error(f"ChatFly 配置文件 {CHATFLY_CONFIG_FILE_NAME} 格式不正确。")
        return {}
    except Exception as e:
        _log_error(f"读取ChatFly配置文件时发生错误: {e}")
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
                _log_warning("ChatFly_config.json中未找到prompt_api_providers配置，使用默认配置。")
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
            _log_warning("未找到ChatFly_config.json配置文件。")
            return {}
    except json.JSONDecodeError:
        _log_error("ChatFly_config.json格式不正确。")
        return {}
    except Exception as e:
        _log_error(f"读取配置文件时发生错误: {e}")
        return {}

def get_provider_config(provider_name):
    """
    根据提供者名称获取配置
    """
    providers = get_prompt_api_providers()
    if provider_name not in providers:
        _log_error(f"未知的API提供者: {provider_name}")
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
        _log_warning(f"提示词文件 '{os.path.basename(file_path)}' 不存在，使用内置默认提示词。")
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
            _log_warning(f"提示词文件 '{os.path.basename(file_path)}' 内容为空或格式不正确，使用内置默认提示词。")
            return default_built_in_prompts

        _log_info(f"从 '{os.path.basename(file_path)}' 加载提示词成功。")
        return prompts

    except Exception as e:
        _log_error(f"解析提示词文件 '{os.path.basename(file_path)}' 失败: {e}。使用内置默认提示词。")
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
                "ref_image": ("STRING", {"multiline": True, "default": "", "placeholder": "Base64编码图片，或由上游节点传入 (Base64 image or from upstream node)", "label": "参考图片 Reference Image"}),
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
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "label": "采样概率 Top-p"}),
                "max_tokens": ("INT", {"default": 400, "min": 1, "max": 4096, "label": "最大Token数 Max Tokens"}),
                "image_url": ("STRING", {"multiline": True, "default": "", "placeholder": "可填写图片Base64或图片URL (Base64 or image URL)", "label": "image_url"})
            }
        }

    def expand_prompt(self, api_provider, image_prompt_preset, ref_image, base_url, api_key, model, system_prompt, user_prompt, user_requirement, temperature=0.7, seed=0, image_url="", top_p=0.8, max_tokens=400):
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
        # 优先用image_url，其次ref_image。都没有时也能扩写文本
        if image_url and image_url.strip():
            url_val = image_url.strip()
            if url_val.startswith("http://") or url_val.startswith("https://"):
                content_parts.append({"type": "image_url", "image_url": {"url": url_val}})
            else:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{url_val}"}})
        elif ref_image and ref_image.strip():
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_image.strip()}"}})
        # 如果没有图片，content_parts 只包含文本部分，API 也能处理文本扩写
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
            _log_info(f"{api_provider} API 扩写响应成功。")
            return (expanded_prompt,)
        except Exception as e:
            error_message = f"{api_provider} API 调用失败: {e}\n{api_provider} API call failed: {e}"
            _log_error(error_message)
            return (error_message,)

# --- 注册节点 ---
NODE_CLASS_MAPPINGS = {
    "Comfly_Prompt_Expand_From_Image": Comfly_Prompt_Expand_From_Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfly_Prompt_Expand_From_Image": "扩写高质量提示词 (Comfly/T8)",
}