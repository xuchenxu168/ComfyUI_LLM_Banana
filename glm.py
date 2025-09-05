import os
import json
import base64
import random

# --- 全局常量和配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_CONFIG_FILE_NAME = 'Glm_Config.json'

# 提示词文件名称
TEXT_PROMPTS_FILE_NAME = 'text_prompts.txt'
IMAGE_PROMPTS_FILE_NAME = 'image_prompts.txt'

# 支持的语言代码列表
SUPPORTED_TRANSLATION_LANGS = ['zh', 'en']

# --- 辅助函数 ---
def _log_info(message):
    print(f"[LLM Prompt] 信息：{message}")

def _log_warning(message):
    print(f"[LLM Prompt] 警告：{message}")

def _log_error(message):
    print(f"[LLM Prompt] 错误：{message}")

def get_glm_api_key():
    env_api_key = os.getenv("ZHIPUAI_API_KEY")
    if env_api_key:
        _log_info("使用环境变量 API Key。")
        return env_api_key

    config_path = os.path.join(CURRENT_DIR, GLM_CONFIG_FILE_NAME)
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            api_key = config.get("ZHIPUAI_API_KEY")
            if api_key:
                _log_info(f"从 {GLM_CONFIG_FILE_NAME} 读取 API Key。")
                return api_key
            else:
                _log_warning(f"在 {GLM_CONFIG_FILE_NAME} 中未找到 ZHIPUAI_API_KEY。")
                return ""
        else:
            _log_warning(f"未找到 API Key 配置文件 {GLM_CONFIG_FILE_NAME}。")
            return ""
    except Exception as e:
        _log_error(f"读取配置文件时发生错误: {e}")
        return ""

def load_prompts_from_txt(file_path, default_built_in_prompts):
    prompts = {}
    current_prompt_name = None
    current_prompt_content = []

    if not os.path.exists(file_path):
        _log_warning(f"提示词文件 '{os.path.basename(file_path)}' 不存在，使用内置默认提示词。")
        return default_built_in_prompts

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('[') and line.endswith(']'):
                    if current_prompt_name and current_prompt_content:
                        prompts[current_prompt_name] = "\n".join(current_prompt_content).strip()
                    current_prompt_name = line[1:-1].strip()
                    current_prompt_content = []
                elif current_prompt_name is not None:
                    current_prompt_content.append(line)

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

# 延迟导入zhipuai
def get_zhipuai_client():
    try:
        from zhipuai import ZhipuAI
        return ZhipuAI
    except ImportError:
        _log_error("zhipuai包未安装，请运行: pip install zhipuai")
        return None







# --- GLM专用节点 ---
class GLM_Prompt_Expand_From_Image:
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("expanded_prompt",)
    FUNCTION = "expand_prompt"

    _BUILT_IN_IMAGE_PROMPTS = {
        "通用高质量英文描述 (内置)": "你是一个专业的图像描述专家，能够将图片内容转化为高质量的英文提示词，用于文本到图像的生成模型。请仔细观察提供的图片，并生成一段详细、具体、富有创造性的英文短语，描述图片中的主体对象、场景、动作、光线、材质、色彩、构图和艺术风格。要求：语言：严格使用英文。细节：尽可能多地描绘图片细节，包括但不限于物体、人物、背景、前景、纹理、表情、动作、服装、道具等。角度：尽可能从多个角度丰富描述，例如特写、广角、俯视、仰视等，但不要直接写\"角度\"。连接：使用逗号（,）连接不同的短语，形成一个连贯的提示词。人物：描绘人物时，使用第三人称（如 'a woman', 'the man'）。质量词：在生成的提示词末尾，务必添加以下质量增强词：', best quality, high resolution, 4k, high quality, masterpiece, photorealistic'"
    }

    @classmethod
    def get_image_prompts(cls):
        return load_prompts_from_txt(
            os.path.join(CURRENT_DIR, IMAGE_PROMPTS_FILE_NAME),
            cls._BUILT_IN_IMAGE_PROMPTS
        )

    @classmethod
    def get_glm_api_key(cls, api_key_input):
        api_key = api_key_input.strip()
        if api_key:
            return api_key
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Glm_Config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get("ZHIPUAI_API_KEY", "")
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        available_prompts = cls.get_image_prompts()
        prompt_keys = list(available_prompts.keys())
        default_selection = prompt_keys[0] if prompt_keys else "无可用提示词"
        
        # 完整的模型列表（包含文本和视觉模型）
        glm_model_list = [
            "glm-4", "glm-4v", "glm-4v-plus", "glm-4v-flash", "glm-4-air", "glm-3-turbo",
            "glm-4-flash-250414", "glm-4-0520", "glm-4v-plus-0111", "glm-4v-flash-250414",
            "glm-4-flash", "glm-4-plus", "glm-4-airx", "glm-4-flashx-250414", "glm-4-long",
            "glm-4v-plus-latest", "glm-4v-latest", "glm-4-latest", "glm-4-air-latest", "glm-3-turbo-latest"
        ]
        
        return {
            "required": {
                "image_prompt_preset": (prompt_keys, {"default": default_selection}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "智谱AI API Key（留空自动从Glm_Config.json读取）"}),
                "model_name": (glm_model_list, {"default": "glm-4v-flash"}),
                "image_base64": ("STRING", {"multiline": True, "default": "", "placeholder": "Base64编码图片，或由上游节点传入"}),
                "system_prompt": ("STRING", {"multiline": True, "default": available_prompts.get(default_selection, ""), "placeholder": "系统提示词（可自定义专家角色，支持中文）"}),
                "user_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入你的原始提示词（支持中文）"}),
                "user_requirement": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入你的额外要求（可选，支持中文）"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            "optional": {
                "image_url": ("STRING", {"default": "", "placeholder": "请输入图片URL（与Base64二选一）"}),
            }
        }

    def expand_prompt(self, image_prompt_preset, api_key, model_name, image_base64, system_prompt, user_prompt, user_requirement, temperature, seed, image_url="",):
        import requests
        final_api_key = self.get_glm_api_key(api_key)
        if not final_api_key:
            return ("未检测到API Key，请在节点输入框填写API Key，或在Glm_Config.json中配置ZHIPUAI_API_KEY。",)
        
        available_prompts = self.get_image_prompts()
        final_system_prompt = system_prompt.strip() or available_prompts.get(image_prompt_preset, "")
        
        # 检查是否有图像输入
        has_image = bool(image_base64.strip() or image_url.strip())
        
        # 根据是否有图像构建不同的内容
        if has_image:
            # 有图像时使用视觉模型
            content_parts = []
            if final_system_prompt:
                content_parts.append({"type": "text", "text": final_system_prompt})
            if user_prompt.strip():
                content_parts.append({"type": "text", "text": user_prompt.strip()})
            if user_requirement.strip():
                content_parts.append({"type": "text", "text": user_requirement.strip()})
            if image_base64.strip():
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64.strip()}"}})
            elif image_url.strip():
                content_parts.append({"type": "image_url", "image_url": {"url": image_url.strip()}})
        else:
            # 无图像时使用文本模型进行提示词扩写
            # 构建扩写提示词
            expand_prompt = f"""请根据以下系统提示词的要求，对用户输入的提示词进行高质量扩写：

系统提示词要求：
{final_system_prompt}

用户原始提示词：{user_prompt.strip()}

用户额外要求：{user_requirement.strip() if user_requirement.strip() else "无"}

请严格按照系统提示词的要求和格式，对用户输入的提示词进行扩写，输出高质量的扩写结果。"""
            
            content_parts = [{"type": "text", "text": expand_prompt}]
        
        messages = [{"role": "user", "content": content_parts}]
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {final_api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            expanded_prompt = data["choices"][0]["message"]["content"]
            return (expanded_prompt,)
        except Exception as e:
            return (f"GLM API 调用失败: {e}",)

# --- 注册GLM节点 ---
NODE_CLASS_MAPPINGS = {
    "GLM_Prompt_Expand_From_Image": GLM_Prompt_Expand_From_Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLM_Prompt_Expand_From_Image": "GLM扩写高质量提示词",
}