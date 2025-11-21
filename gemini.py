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
import time

# 视频处理相关导入
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ opencv-python 未安装，视频处理功能将受限。请运行: pip install opencv-python")

# Gemini SDK 导入
try:
    from google import genai
    from google.genai import types
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False
    print("⚠️ google-genai SDK 未安装，将使用 REST API。建议运行: pip install google-genai")

# 延迟导入google.genai (保留兼容性)
def get_google_genai():
    try:
        import google.genai
        return google.genai
    except ImportError:
        return None

def _log_info(message):
    print(f"[LLM Prompt] {message}")

def _log_warning(message):
    print(f"[LLM Prompt] WARNING: {message}")

def _log_error(message):
    print(f"[LLM Prompt] ERROR: {message}")

def load_gemini_config():
    """加载Gemini配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gemini_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            _log_info(f"成功加载Gemini配置文件: {config_path}")
            return config
        else:
            _log_warning(f"Gemini配置文件不存在: {config_path}")
            return {}
    except Exception as e:
        _log_error(f"加载Gemini配置文件失败: {e}")
        return {}

def get_gemini_config():
    """获取Gemini配置，优先从配置文件读取"""
    config = load_gemini_config()
    return config

def save_video_tensor_to_mp4(video_tensor, output_path, fps=30):
    """
    将 ComfyUI 的 VIDEO tensor 保存为 MP4 文件

    Args:
        video_tensor: torch.Tensor, shape [frames, channels, height, width]
        output_path: str, 输出 MP4 文件路径
        fps: int, 帧率（默认 30）

    Returns:
        str: 输出文件路径，如果失败返回 None
    """
    if not CV2_AVAILABLE:
        _log_error("opencv-python 未安装，无法保存视频为 MP4")
        return None

    try:
        frames_count = video_tensor.shape[0]
        height = video_tensor.shape[2]
        width = video_tensor.shape[3]

        _log_info(f"📹 保存视频: {frames_count} 帧, {width}x{height}, {fps} FPS")

        # 创建视频写入器 (使用 mp4v 编码器)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            _log_error(f"无法创建视频写入器: {output_path}")
            return None

        for i in range(frames_count):
            frame = video_tensor[i]

            # 转换为 numpy 数组 (HWC 格式)
            if frame.shape[0] in [1, 3, 4]:  # CHW 格式
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
            else:  # HWC 格式
                frame_np = frame.cpu().numpy()

            # 转换为 uint8
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

            # 处理通道数
            if frame_np.shape[2] == 1:  # 灰度
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
            elif frame_np.shape[2] == 3:  # RGB -> BGR
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            elif frame_np.shape[2] == 4:  # RGBA -> BGR
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR)

            out.write(frame_np)

        out.release()
        _log_info(f"✅ 视频保存成功: {output_path}")
        return output_path

    except Exception as e:
        _log_error(f"保存视频失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def upload_video_to_gemini_file_api(video_path, api_key, display_name="video"):
    """
    使用 Gemini File API 上传视频文件

    Args:
        video_path: str, 视频文件路径
        api_key: str, Gemini API 密钥
        display_name: str, 显示名称

    Returns:
        dict: 包含 file_uri 和 mime_type 的字典，如果失败返回 None
    """
    if not GENAI_SDK_AVAILABLE:
        _log_error("google-genai SDK 未安装，无法使用 File API")
        return None

    try:
        _log_info(f"📤 上传视频到 Gemini File API: {video_path}")

        # 创建客户端
        client = genai.Client(api_key=api_key)

        # 上传文件
        uploaded_file = client.files.upload(file=video_path)

        _log_info(f"⏳ 等待文件处理... (文件名: {uploaded_file.name})")

        # 等待文件处理完成
        max_wait = 60  # 最多等待 60 秒
        wait_time = 0
        while uploaded_file.state.name == "PROCESSING":
            if wait_time >= max_wait:
                _log_error(f"文件处理超时 ({max_wait}秒)")
                return None

            time.sleep(1)
            wait_time += 1
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            _log_error(f"文件处理失败: {uploaded_file.name}")
            return None

        _log_info(f"✅ 视频上传成功! URI: {uploaded_file.uri}")

        return {
            "file_uri": uploaded_file.uri,
            "mime_type": uploaded_file.mime_type,
            "file_name": uploaded_file.name
        }

    except Exception as e:
        _log_error(f"上传视频到 File API 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

class KenChenLLMGeminiTextNode:
    @classmethod
    def INPUT_TYPES(s):
        config = get_gemini_config()
        default_params = config.get('default_params', {})
        
        # 添加默认模型列表，防止配置加载失败
        default_models = [
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b",
        ]
        
        models = config.get('models', {}).get('text_models', default_models)
        if not models:
            models = default_models
            print("[LLM Prompt] 警告: 配置文件中的文本模型列表为空，使用默认模型列表")
        
        default_model = config.get('default_model', {}).get('text_gen', "gemini-2.0-flash-lite")
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
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
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_text"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
    
    def generate_text(
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
    ):
        try:
            genai = get_google_genai()
            if genai is None:
                return (f"错误: 未安装google-genai库，请运行: pip install google-genai",)
            
            config = get_gemini_config()
            final_api_key = api_key.strip() if api_key.strip() else config.get('api_key', '')
            final_proxy = proxy.strip() if proxy.strip() else config.get('proxy', '')
            if not final_api_key:
                return (f"错误: 请提供Gemini API Key",)
            if final_proxy and final_proxy.strip() and "None" not in final_proxy:
                os.environ['HTTPS_PROXY'] = final_proxy
                os.environ['HTTP_PROXY'] = final_proxy
                _log_info(f"已设置代理: {final_proxy}")
            elif final_proxy and "None" in final_proxy:
                _log_info("跳过无效代理设置")
            
            # 使用新的API结构
            client = genai.Client(api_key=final_api_key)
            
            # 准备生成配置
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }
            if seed > 0:
                generation_config["seed"] = seed
            
            # 生成文本
            response = client.models.generate_content(
                model=model,
                contents=[{"parts": [{"text": prompt}]}],
                config=generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text
                _log_info(f"KenChen LLM Gemini文本生成成功，模型: {model}")
                return (text,)
            else:
                return (f"错误: 未获得有效响应",)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                _log_error(f"API配额限制: {error_msg}")
                return (f"错误: API配额限制，请尝试以下解决方案：\n1. 使用免费模型 (如 gemini-2.0-flash-lite)\n2. 升级到付费账户\n3. 等待配额重置\n\n详细错误: {error_msg}",)
            else:
                error_msg = f"KenChen LLM Gemini文本生成失败: {e}"
                _log_error(error_msg)
                return (error_msg,)

class KenChenLLMGeminiMultimodalNode:
    @classmethod
    def INPUT_TYPES(s):
        config = get_gemini_config()
        default_params = config.get('default_params', {})
        
        # 添加默认模型列表，防止配置加载失败
        default_models = [
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp-image-generation",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.5-pro-exp-03-25",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-05-06",
        ]
        
        models = config.get('models', {}).get('multimodal_models', default_models)
        if not models:
            models = default_models
            print("[LLM Prompt] 警告: 配置文件中的多模态模型列表为空，使用默认模型列表")
        
        default_model = config.get('default_model', {}).get('multimodal', "gemini-1.5-flash")
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
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
                "video": ("VIDEO",),
                "max_video_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "视频处理时最多提取的帧数。增加此值会提高视频分析质量，但会增加API成本和处理时间。建议值: 10-30"
                }),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("multimodal_text",)  # 改为唯一的名称
    FUNCTION = "generate_multimodal"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
    
    def generate_multimodal(
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
        video=None,
        max_video_frames=10,
    ):
        try:
            genai = get_google_genai()
            if genai is None:
                return (f"错误: 未安装google-genai库，请运行: pip install google-genai",)
            config = get_gemini_config()
            final_api_key = api_key.strip() if api_key.strip() else config.get('api_key', '')
            final_proxy = proxy.strip() if proxy.strip() else config.get('proxy', '')
            if not final_api_key:
                return (f"错误: 请提供Gemini API Key",)
            if final_proxy and final_proxy.strip() and "None" not in final_proxy:
                os.environ['HTTPS_PROXY'] = final_proxy
                os.environ['HTTP_PROXY'] = final_proxy
                _log_info(f"已设置代理: {final_proxy}")
            elif final_proxy and "None" in final_proxy:
                _log_info("跳过无效代理设置")
            
            # 使用新的API结构
            client = genai.Client(api_key=final_api_key)
            
            # 检查模型能力
            _log_info(f"使用模型: {model}")
            if "image-generation" in model.lower():
                _log_warning(f"警告: {model} 是图像生成模型，可能不支持音频处理")
            elif "flash" in model.lower() and "exp" not in model.lower():
                _log_info(f"使用 Flash 模型: {model}，支持多模态输入")
            else:
                _log_info(f"使用模型: {model}")
            
            # 准备生成配置
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }
            if seed > 0:
                generation_config["seed"] = seed
            
            # 准备内容
            # 如果有视频输入，需要在提示词中添加视频上下文
            video_context_added = False
            if video is not None:
                # 检查提示词中是否已经包含视频相关的说明
                video_keywords = ['视频', 'video', '帧', 'frame', '动态', '连续', '序列']
                has_video_context = any(keyword in prompt.lower() for keyword in video_keywords)

                if not has_video_context:
                    # 自动添加视频上下文说明（更强的提示）
                    video_prompt_prefix = (
                        "【重要提示 - 这是视频分析任务】\n"
                        "接下来你会看到多张图片，这些图片是从同一个视频中按时间顺序提取的连续帧。\n"
                        "请务必：\n"
                        "1. 将所有图片作为一个完整的视频序列来分析\n"
                        "2. 分析视频中的动作、场景变化、时间流程\n"
                        "3. 不要只描述第一张图片，要综合所有帧的信息\n"
                        "4. 如果看到场景或人物的变化，请描述这个变化过程\n\n"
                        "用户的问题：\n"
                    )
                    enhanced_prompt = video_prompt_prefix + prompt
                    content_parts = [{"text": enhanced_prompt}]
                    video_context_added = True
                    _log_info("🎬 自动添加视频分析上下文到提示词")
                else:
                    content_parts = [{"text": prompt}]
            else:
                content_parts = [{"text": prompt}]

            # 处理图像
            if image is not None:
                if isinstance(image, torch.Tensor):
                    try:
                        _log_info(f"开始处理输入图像，原始形状: {image.shape}, 数据类型: {image.dtype}")
                        
                        # 检查图像维度
                        if image.dim() == 4:
                            image = image[0]  # 取第一张图片
                            _log_info(f"4D图像，取第一张，新形状: {image.shape}")
                        
                        # 检查图像形状
                        if image.dim() != 3:
                            _log_warning(f"图像维度不正确: {image.dim()}, 期望3维")
                            return (f"错误: 图像维度不正确，期望3维，实际{image.dim()}维",)
                        
                        # 检查图像尺寸和通道数
                        if len(image.shape) == 3:
                            # 判断是CHW还是HWC格式
                            if image.shape[0] in [1, 3, 4]:  # CHW格式
                                channels, height, width = image.shape
                                _log_info(f"CHW格式图像 - 通道: {channels}, 高度: {height}, 宽度: {width}")
                                
                                # 检查是否是正常的图像格式
                                if channels not in [1, 3, 4]:
                                    _log_warning(f"通道数异常: {channels}, 期望1,3,4")
                                    return (f"错误: 通道数异常: {channels}, 期望1,3,4",)
                                
                                if height < 10 or width < 10:
                                    _log_warning(f"图像尺寸太小: {height}x{width}")
                                    return (f"错误: 图像尺寸太小: {height}x{width}",)
                                    
                            elif image.shape[2] in [1, 3, 4]:  # HWC格式
                                height, width, channels = image.shape
                                _log_info(f"HWC格式图像 - 高度: {height}, 宽度: {width}, 通道: {channels}")
                                
                                # 检查是否是正常的图像格式
                                if channels not in [1, 3, 4]:
                                    _log_warning(f"通道数异常: {channels}, 期望1,3,4")
                                    return (f"错误: 通道数异常: {channels}, 期望1,3,4",)
                                
                                if height < 10 or width < 10:
                                    _log_warning(f"图像尺寸太小: {height}x{width}")
                                    return (f"错误: 图像尺寸太小: {height}x{width}",)
                                    
                            else:
                                _log_warning(f"无法识别的图像格式: {image.shape}")
                                return (f"错误: 无法识别的图像格式: {image.shape}",)
                        
                        # 转换为PIL图像
                        try:
                            # 确保是CHW格式
                            if image.shape[0] in [1, 3, 4]:  # 通道在第一个维度
                                image_np = image.permute(1, 2, 0).cpu().numpy()
                            else:
                                # 如果已经是HWC格式，直接转换
                                image_np = image.cpu().numpy()
                            
                            _log_info(f"转换为numpy数组，形状: {image_np.shape}, 数据类型: {image_np.dtype}")
                            
                            # 检查数据类型和范围
                            if image_np.dtype != np.float32 and image_np.dtype != np.float64:
                                image_np = image_np.astype(np.float32)
                            
                            # 确保值在0-1范围内
                            if image_np.max() > 1.0:
                                image_np = image_np / 255.0
                            
                            # 转换为uint8
                            image_np = (image_np * 255).astype(np.uint8)
                            
                            # 处理单通道图像
                            if image_np.shape[2] == 1:
                                image_np = np.repeat(image_np, 3, axis=2)
                                _log_info("单通道图像转换为RGB")
                            
                            # 处理RGBA图像
                            if image_np.shape[2] == 4:
                                image_np = image_np[:, :, :3]
                                _log_info("RGBA图像转换为RGB")
                            
                            pil_image = Image.fromarray(image_np)
                            _log_info(f"成功创建PIL图像，尺寸: {pil_image.size}, 模式: {pil_image.mode}")
                            
                        except Exception as permute_error:
                            _log_error(f"图像转换失败: {permute_error}")
                            return (f"错误: 图像转换失败: {permute_error}",)
                        
                        # 转换为base64
                        buffer = BytesIO()
                        pil_image.save(buffer, format='PNG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode()
                        content_parts.append({
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        })
                        _log_info(f"成功处理输入图像，最终尺寸: {pil_image.size}")
                        
                    except Exception as e:
                        _log_error(f"处理输入图像时出错: {e}")
                        return (f"错误: 处理输入图像失败: {e}",)
            
            # 处理音频
            if audio is not None:
                _log_info(f"检测到音频输入，类型: {type(audio)}")
                
                # 保存原始音频字典以便获取采样率
                original_audio_dict = None
                
                # 处理字典格式的音频输入（ComfyUI音频节点输出）
                if isinstance(audio, dict):
                    _log_info(f"音频输入为字典格式，键: {list(audio.keys())}")
                    original_audio_dict = audio  # 保存原始字典
                    
                    # 尝试从字典中提取音频数据
                    audio_tensor = None
                    if 'waveform' in audio:
                        audio_tensor = audio['waveform']
                        _log_info("从 'waveform' 键提取音频数据")
                    elif 'samples' in audio:
                        audio_tensor = audio['samples']
                        _log_info("从 'samples' 键提取音频数据")
                    elif 'audio' in audio:
                        audio_tensor = audio['audio']
                        _log_info("从 'audio' 键提取音频数据")
                    elif 'data' in audio:
                        audio_tensor = audio['data']
                        _log_info("从 'data' 键提取音频数据")
                    else:
                        _log_warning(f"无法从音频字典中找到音频数据，可用键: {list(audio.keys())}")
                        return (f"错误: 无法从音频字典中找到音频数据，可用键: {list(audio.keys())}",)
                    
                    # 如果提取的是numpy数组，转换为tensor
                    if isinstance(audio_tensor, np.ndarray):
                        audio_tensor = torch.from_numpy(audio_tensor)
                        _log_info("将numpy数组转换为torch.Tensor")
                    
                    if not isinstance(audio_tensor, torch.Tensor):
                        _log_warning(f"提取的音频数据类型不正确: {type(audio_tensor)}, 期望 torch.Tensor")
                        return (f"错误: 提取的音频数据类型不正确: {type(audio_tensor)}, 期望 torch.Tensor",)
                    
                    audio = audio_tensor  # 替换为提取的tensor
                
                # 处理torch.Tensor格式的音频
                if isinstance(audio, torch.Tensor):
                    try:
                        _log_info(f"开始处理输入音频，原始形状: {audio.shape}, 数据类型: {audio.dtype}")
                        
                        # 检查音频维度
                        if audio.dim() == 1:
                            # 单声道音频
                            audio_np = audio.cpu().numpy()
                            _log_info(f"单声道音频，长度: {len(audio_np)}")
                        elif audio.dim() == 2:
                            # 立体声音频，取第一个通道
                            audio_np = audio[0].cpu().numpy()
                            _log_info(f"立体声音频，取第一通道，长度: {len(audio_np)}")
                        elif audio.dim() == 3:
                            # ComfyUI 3维音频格式: [batch, channels, samples]
                            _log_info(f"3维音频格式，形状: {audio.shape}")
                            if audio.shape[0] == 1 and audio.shape[1] == 1:
                                # 单声道音频，取第一个样本
                                audio_np = audio[0, 0].cpu().numpy()
                                _log_info(f"提取单声道音频，长度: {len(audio_np)}")
                            elif audio.shape[0] == 1 and audio.shape[1] > 1:
                                # 立体声音频，取第一个通道
                                audio_np = audio[0, 0].cpu().numpy()
                                _log_info(f"提取立体声第一通道，长度: {len(audio_np)}")
                            else:
                                _log_warning(f"无法处理的3维音频格式: {audio.shape}")
                                return (f"错误: 无法处理的3维音频格式: {audio.shape}",)
                        else:
                            _log_warning(f"音频维度不正确: {audio.dim()}, 期望1、2或3维")
                            return (f"错误: 音频维度不正确，期望1、2或3维，实际{audio.dim()}维",)
                        
                        # 检查音频长度
                        if len(audio_np) < 1000:
                            _log_warning(f"音频太短: {len(audio_np)} 采样点")
                            return (f"错误: 音频太短: {len(audio_np)} 采样点",)
                        
                        # 确保数据类型正确
                        if audio_np.dtype != np.float32 and audio_np.dtype != np.float64:
                            audio_np = audio_np.astype(np.float32)
                        
                        # 确保值在合理范围内
                        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                            audio_np = np.clip(audio_np, -1.0, 1.0)
                            _log_info("音频值已裁剪到 [-1, 1] 范围")
                        
                        # 获取采样率（如果原始音频字典中有的话）
                        sample_rate = 44100  # 默认采样率
                        if original_audio_dict and 'sample_rate' in original_audio_dict:
                            sample_rate = original_audio_dict['sample_rate']
                            _log_info(f"使用原始采样率: {sample_rate} Hz")
                        
                        # 转换为WAV字节
                        buffer = BytesIO()
                        try:
                            import scipy.io.wavfile as wavfile
                            wavfile.write(buffer, sample_rate, audio_np)
                            audio_bytes = buffer.getvalue()
                            audio_base64 = base64.b64encode(audio_bytes).decode()
                            content_parts.append({
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": audio_base64
                                }
                            })
                            _log_info(f"成功处理输入音频，长度: {len(audio_np)} 采样点，采样率: {sample_rate} Hz")
                        except ImportError:
                            _log_error("scipy 库未安装，无法处理音频")
                            return (f"错误: scipy 库未安装，请运行: pip install scipy",)
                        
                    except Exception as e:
                        _log_error(f"处理输入音频时出错: {e}")
                        return (f"错误: 处理输入音频失败: {e}",)
                else:
                    _log_warning(f"音频输入类型不正确: {type(audio)}, 期望 dict 或 torch.Tensor")
                    return (f"错误: 音频输入类型不正确: {type(audio)}, 期望 dict 或 torch.Tensor",)
            else:
                _log_info("没有音频输入")
            
            # 🎬 处理视频 - 使用 File API
            uploaded_video_file = None
            temp_video_path = None

            if video is not None:
                try:
                    # 检查视频类型
                    if isinstance(video, str):
                        # Load_AF_Video 返回视频文件路径 - 直接使用
                        _log_info(f"🎬 检测到视频文件路径: {video}")
                        temp_video_path = video
                        use_existing_file = True
                    elif isinstance(video, (dict, torch.Tensor)):
                        # 需要转换为 MP4
                        video_frames = None

                        # 支持多种 VIDEO 类型格式
                        if isinstance(video, dict):
                            if "video" in video:
                                video_frames = video["video"]
                            elif "frames" in video:
                                video_frames = video["frames"]
                            else:
                                for key, value in video.items():
                                    if isinstance(value, torch.Tensor):
                                        video_frames = value
                                        break
                        elif isinstance(video, torch.Tensor):
                            video_frames = video

                        if video_frames is not None and isinstance(video_frames, torch.Tensor):
                            _log_info(f"🎬 处理视频，形状: {video_frames.shape}, 数据类型: {video_frames.dtype}")

                            if video_frames.dim() == 4:
                                frames_count = video_frames.shape[0]
                                _log_info(f"📊 视频帧数: {frames_count}")

                                # 保存为临时 MP4 文件
                                temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                                temp_video_path = temp_file.name
                                temp_file.close()

                                fps = 30
                                saved_path = save_video_tensor_to_mp4(video_frames, temp_video_path, fps=fps)

                                if not saved_path:
                                    _log_error("保存视频失败，无法使用 File API")
                                    temp_video_path = None
                                else:
                                    use_existing_file = False
                            else:
                                _log_warning(f"视频维度不正确: {video_frames.dim()}, 期望4维")
                                temp_video_path = None
                        else:
                            _log_warning("无法从 dict 中提取视频 tensor")
                            temp_video_path = None
                    else:
                        _log_warning(f"视频输入类型不支持: {type(video)}")
                        temp_video_path = None

                    # 如果有视频文件，上传到 File API
                    if temp_video_path:
                        _log_info("🎬 检测到视频输入，使用 Gemini File API 上传视频")

                        uploaded_video_file = upload_video_to_gemini_file_api(
                            temp_video_path,
                            final_api_key,
                            display_name="comfyui_video"
                        )

                        if not uploaded_video_file:
                            _log_error("视频上传失败，将跳过视频分析")
                            temp_video_path = None

                except Exception as e:
                    _log_error(f"处理输入视频时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    temp_video_path = None
            
            # 🎬 如果有视频文件，使用 SDK 调用
            if uploaded_video_file and GENAI_SDK_AVAILABLE:
                _log_info(f"🔍 使用 Gemini SDK 进行视频分析...")

                try:
                    # 使用 SDK 调用
                    client_sdk = genai.Client(api_key=final_api_key)

                    # 构建内容
                    sdk_contents = [
                        types.Part(text=prompt),
                        types.Part(
                            file_data=types.FileData(
                                file_uri=uploaded_video_file['file_uri'],
                                mime_type=uploaded_video_file['mime_type']
                            )
                        )
                    ]

                    # 添加其他媒体（如果有）
                    for part in content_parts[1:]:  # 跳过第一个文本部分
                        if "inline_data" in part:
                            sdk_contents.append(
                                types.Part(
                                    inline_data=types.Blob(
                                        data=base64.b64decode(part["inline_data"]["data"]),
                                        mime_type=part["inline_data"]["mime_type"]
                                    )
                                )
                            )

                    # 调用 API
                    response = client_sdk.models.generate_content(
                        model=model,
                        contents=types.Content(parts=sdk_contents),
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            max_output_tokens=max_output_tokens
                        )
                    )

                    text = response.text

                    # 清理临时文件
                    if temp_video_path and not use_existing_file and os.path.exists(temp_video_path):
                        try:
                            os.unlink(temp_video_path)
                            _log_info("🗑️ 临时视频文件已删除")
                        except:
                            pass

                    _log_info("✅ 视频分析成功完成")
                    return (text,)

                except Exception as sdk_error:
                    _log_error(f"SDK 调用失败: {sdk_error}")
                    # 清理临时文件
                    if temp_video_path and not use_existing_file and os.path.exists(temp_video_path):
                        try:
                            os.unlink(temp_video_path)
                        except:
                            pass
                    raise

            # 如果没有视频或 SDK 不可用，使用原来的方式
            _log_info(f"准备发送到API的内容部分数量: {len(content_parts)}")
            for i, part in enumerate(content_parts):
                if "text" in part:
                    _log_info(f"第 {i+1} 部分: 文本内容 (长度: {len(part['text'])})")
                elif "inline_data" in part:
                    mime_type = part["inline_data"]["mime_type"]
                    data_length = len(part["inline_data"]["data"])
                    _log_info(f"第 {i+1} 部分: {mime_type} 数据 (长度: {data_length})")

            # 根据内容类型选择调用方式
            has_audio = any("audio" in part.get("inline_data", {}).get("mime_type", "") for part in content_parts if "inline_data" in part)
            has_image = any("image" in part.get("inline_data", {}).get("mime_type", "") for part in content_parts if "inline_data" in part)

            if has_audio:
                _log_info("检测到音频内容，使用多模态调用")
                response = client.models.generate_content(
                    model=model,
                    contents=[{"parts": content_parts}],
                    config=generation_config
                )
            elif has_image:
                _log_info("检测到图像内容，使用多模态调用")
                response = client.models.generate_content(
                    model=model,
                    contents=[{"parts": content_parts}],
                    config=generation_config
                )
            else:
                _log_info("使用标准文本调用")
                response = client.models.generate_content(
                    model=model,
                    contents=[{"parts": content_parts}],
                    config=generation_config
                )

            # 清理临时文件
            if temp_video_path and not use_existing_file and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                    _log_info("🗑️ 临时视频文件已删除")
                except:
                    pass
            
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text
                _log_info(f"KenChen LLM Gemini多模态生成成功，模型: {model}")
                return (text,)
            else:
                return (f"错误: 未获得有效响应",)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                _log_error(f"API配额限制: {error_msg}")
                return (f"错误: API配额限制，请尝试以下解决方案：\n1. 使用免费模型 (如 gemini-1.5-flash)\n2. 升级到付费账户\n3. 等待配额重置\n\n详细错误: {error_msg}",)
            else:
                error_msg = f"KenChen LLM Gemini多模态生成失败: {e}"
                _log_error(error_msg)
                return (error_msg,)

class KenChenLLMGeminiImageGenerationNode:
    @classmethod
    def INPUT_TYPES(s):
        config = get_gemini_config()
        default_params = config.get('default_params', {})
        
        # 图像生成专用模型
        default_models = [
            "gemini-2.0-flash-preview-image-generation",
            "gemini-2.0-flash-exp-image-generation",
        ]
        
        models = config.get('models', {}).get('image_gen_models', default_models)
        if not models:
            models = default_models
        
        default_model = "gemini-2.0-flash-preview-image-generation"
        default_proxy = "http://127.0.0.1:None"
        
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (models, {"default": default_model}),
                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 0, "max": 32768}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("generation_text", "generated_image",)
    FUNCTION = "generate_image"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
    
    def generate_image(self, api_key, prompt, model, proxy, temperature, top_p, top_k, max_output_tokens, seed, image=None):
        try:
            genai = get_google_genai()
            if genai is None:
                dummy_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                return (f"错误: 未安装google-genai库，请运行: pip install google-genai", dummy_image)
            
            config = get_gemini_config()
            final_api_key = api_key.strip() if api_key.strip() else config.get('api_key', '')
            if not final_api_key:
                dummy_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                return (f"错误: 请提供Gemini API Key", dummy_image)
            
            client = genai.Client(api_key=final_api_key)
            
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
                "response_modalities": ["TEXT", "IMAGE"],  # 图像生成必须指定
            }
            if seed > 0:
                generation_config["seed"] = seed
            
            content_parts = [{"text": prompt}]
            
            # 处理输入图像（图生图功能）
            if image is not None:
                if isinstance(image, torch.Tensor):
                    if image.dim() == 4:
                        image = image[0]
                    
                    if image.shape[0] in [1, 3, 4]:
                        image_np = image.permute(1, 2, 0).cpu().numpy()
                    else:
                        image_np = image.cpu().numpy()
                    
                    if image_np.dtype != np.float32:
                        image_np = image_np.astype(np.float32)
                    
                    if image_np.max() > 1.0:
                        image_np = image_np / 255.0
                    
                    image_np = (image_np * 255).astype(np.uint8)
                    
                    if image_np.shape[2] == 1:
                        image_np = np.repeat(image_np, 3, axis=2)
                    elif image_np.shape[2] == 4:
                        image_np = image_np[:, :, :3]
                    
                    pil_image = Image.fromarray(image_np)
                    buffer = BytesIO()
                    pil_image.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    content_parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_base64
                        }
                    })
            
            response = client.models.generate_content(
                model=model,
                contents=[{"parts": content_parts}],
                config=generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts[0].text else "图像生成完成"
                
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.mime_type.startswith('image/'):
                        try:
                            if isinstance(part.inline_data.data, bytes):
                                img_data = part.inline_data.data
                            else:
                                img_data = base64.b64decode(part.inline_data.data)
                            
                            pil_image = Image.open(BytesIO(img_data))
                            img_array = np.array(pil_image)
                            
                            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                                img_array = img_array[:, :, :3]
                            
                            img_tensor = torch.from_numpy(img_array).float() / 255.0
                            if len(img_tensor.shape) == 3 and img_tensor.shape[2] == 3:
                                img_tensor = img_tensor.unsqueeze(0)
                            else:
                                img_tensor = img_tensor.permute(1, 2, 0).unsqueeze(0)
                            
                            return (text, img_tensor)
                        except Exception as img_error:
                            _log_error(f"处理生成的图像时出错: {img_error}")
                            continue
                
                dummy_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                return (text, dummy_image)
            
            dummy_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
            return (f"错误: 未生成有效图像", dummy_image)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                dummy_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                return (f"错误: API配额限制: {error_msg}", dummy_image)
            else:
                dummy_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                return (f"错误: {error_msg}", dummy_image)

class KenChenLLMGeminiImageAnalysisNode:
    @classmethod
    def INPUT_TYPES(s):
        config = get_gemini_config()
        default_params = config.get('default_params', {})
        
        # 图像分析专用模型
        default_models = [
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ]
        
        models = config.get('models', {}).get('multimodal_models', default_models)
        if not models:
            models = default_models
        
        default_model = "gemini-1.5-flash"
        default_proxy = "http://127.0.0.1:None"
        
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "请详细分析这张图片的内容", "multiline": True}),
                "model": (models, {"default": default_model}),
                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 0, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_text",)
    FUNCTION = "analyze_image"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
    
    def analyze_image(self, api_key, prompt, model, proxy, temperature, top_p, top_k, max_output_tokens, seed, image=None):
        try:
            genai = get_google_genai()
            if genai is None:
                return (f"错误: 未安装google-genai库，请运行: pip install google-genai",)
            
            config = get_gemini_config()
            final_api_key = api_key.strip() if api_key.strip() else config.get('api_key', '')
            if not final_api_key:
                return (f"错误: 请提供Gemini API Key",)
            
            if image is None:
                return (f"错误: 请提供要分析的图像",)
            
            client = genai.Client(api_key=final_api_key)
            
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }
            if seed > 0:
                generation_config["seed"] = seed
            
            content_parts = [{"text": prompt}]
            
            # 处理输入图像
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]
                
                if image.shape[0] in [1, 3, 4]:
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                else:
                    image_np = image.cpu().numpy()
                
                if image_np.dtype != np.float32:
                    image_np = image_np.astype(np.float32)
                
                if image_np.max() > 1.0:
                    image_np = image_np / 255.0
                
                image_np = (image_np * 255).astype(np.uint8)
                
                if image_np.shape[2] == 1:
                    image_np = np.repeat(image_np, 3, axis=2)
                elif image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3]
                
                pil_image = Image.fromarray(image_np)
                buffer = BytesIO()
                pil_image.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                content_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })
            
            response = client.models.generate_content(
                model=model,
                contents=[{"parts": content_parts}],
                config=generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text
                _log_info(f"KenChen LLM Gemini图像分析成功，模型: {model}")
                return (text,)
            else:
                return (f"错误: 未获得有效响应",)
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                _log_error(f"API配额限制: {error_msg}")
                return (f"错误: API配额限制: {error_msg}",)
            else:
                error_msg = f"KenChen LLM Gemini图像分析失败: {e}"
                _log_error(error_msg)
                return (error_msg,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "KenChenLLMPromptGeminiText": KenChenLLMGeminiTextNode,
    "KenChenLLMPromptGeminiMultimodal": KenChenLLMGeminiMultimodalNode,
    "KenChenLLMPromptGeminiImageGeneration": KenChenLLMGeminiImageGenerationNode,
    "KenChenLLMPromptGeminiImageAnalysis": KenChenLLMGeminiImageAnalysisNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KenChenLLMPromptGeminiText": "Gemini-Text",
    "KenChenLLMPromptGeminiMultimodal": "Gemini-Multimodal",
    "KenChenLLMPromptGeminiImageGeneration": "Gemini-Image-Generation",
    "KenChenLLMPromptGeminiImageAnalysis": "Gemini-Image-Analysis",
}
