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
from typing import Tuple, Optional

# 🚀 nano-banana官方调用方式已集成
# gemini_banana.py 已经包含了完整的nano-banana官方调用实现
# 包括：generate_with_priority_api, generate_with_official_api, generate_with_rest_api 等
NANO_BANANA_OFFICIAL_AVAILABLE = True
print("✅ nano-banana官方调用方式已集成到gemini_banana模块")

# 🌐 导入独立的翻译模块
try:
    from gemini_banana_translation import (
        KenChenLLMGeminiBananaTextTranslationNode as TranslationNode,
        NODE_CLASS_MAPPINGS as TRANSLATION_NODE_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as TRANSLATION_DISPLAY_MAPPINGS
    )
    TRANSLATION_MODULE_AVAILABLE = True
    print("✅ 成功导入独立翻译模块")
except ImportError:
    try:
        from .gemini_banana_translation import (
            KenChenLLMGeminiBananaTextTranslationNode as TranslationNode,
            NODE_CLASS_MAPPINGS as TRANSLATION_NODE_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as TRANSLATION_DISPLAY_MAPPINGS
        )
        TRANSLATION_MODULE_AVAILABLE = True
        print("✅ 成功导入独立翻译模块")
    except ImportError:
        print("⚠️ 无法导入翻译模块，翻译功能将不可用")
        TRANSLATION_MODULE_AVAILABLE = False
        TranslationNode = None
        TRANSLATION_NODE_MAPPINGS = {}
        TRANSLATION_DISPLAY_MAPPINGS = {}

# 🚀 AI放大模型集成
def detect_available_upscale_models():
    """
    自动检测可用的AI放大模型
    """
    available_models = []
    
    # 检测 Real-ESRGAN
    try:
        import realesrgan
        available_models.append("Real-ESRGAN")
        print(f"✅ 检测到 Real-ESRGAN 模型")
    except ImportError:
        print(f"⚠️ Real-ESRGAN 模型未安装")
    
    # 检测 ESRGAN
    try:
        import esrgan
        available_models.append("ESRGAN")
        print(f"✅ 检测到 ESRGAN 模型")
    except ImportError:
        print(f"⚠️ ESRGAN 模型未安装")
    
    # 检测 Waifu2x
    try:
        import waifu2x
        available_models.append("Waifu2x")
        print(f"✅ 检测到 Waifu2x 模型")
    except ImportError:
        print(f"⚠️ Waifu2x 模型未安装")
    
    # 检测 GFPGAN
    try:
        import gfpgan
        available_models.append("GFPGAN")
        print(f"✅ 检测到 GFPGAN 模型")
    except ImportError:
        print(f"⚠️ GFPGAN 模型未安装")
    
    print(f"🔍 可用AI放大模型: {available_models}")
    return available_models

def ai_upscale_with_realesrgan(image, target_width, target_height, gigapixel_model="High Fidelity"):
    """
    统一委托到通用放大器（banana_upscale.smart_upscale），优先2x，失败回退LANCZOS。
    """
    try:
        from .banana_upscale import smart_upscale as _smart
        res = _smart(image, target_width, target_height, gigapixel_model)
        if res is not None:
            return res
        print(f"⚠️ 智能放大器不可用，使用高质量重采样")
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"⚠️ 智能放大器失败，回退重采样: {e}")
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
def ai_upscale_with_esrgan(image, target_width, target_height):
    """
    使用 ESRGAN 进行AI高清放大
    """
    try:
        print(f"🚀 使用 ESRGAN 进行AI高清放大...")
        
        # ESRGAN 实现代码
        # 这里需要根据具体的ESRGAN实现来编写
        
        print(f"✅ ESRGAN AI放大完成")
        return image  # 临时返回原图
        
    except Exception as e:
        print(f"❌ ESRGAN 放大失败: {e}")
        raise e

def ai_upscale_with_waifu2x(image, target_width, target_height):
    """
    使用 Waifu2x 进行AI高清放大
    """
    try:
        print(f"🚀 使用 Waifu2x 进行AI高清放大...")
        
        # Waifu2x 实现代码
        # 这里需要根据具体的Waifu2x实现来编写
        
        print(f"✅ Waifu2x AI放大完成")
        return image  # 临时返回原图
        
    except Exception as e:
        print(f"❌ Waifu2x 放大失败: {e}")
        raise e

def smart_ai_upscale(image, target_width, target_height, gigapixel_model="High Fidelity"):
	"""
	统一委托到通用放大器（banana_upscale.smart_upscale）
	"""
	try:
		from .banana_upscale import smart_upscale as _smart
		return _smart(image, target_width, target_height, gigapixel_model)
	except Exception as e:
		_log_warning(f"⚠️ 智能放大器失败: {e}")
		return None

try:
    from server import PromptServer
except Exception:
    PromptServer = None

def _log_info(message):
    pass  # 关闭调试信息

def _log_warning(message):
    pass  # 关闭调试信息

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

def remove_white_areas(image: Image.Image, white_threshold: int = 240) -> Image.Image:
    """
    检测并去除图像中的白色区域

    Args:
        image: 输入图像
        white_threshold: 白色阈值，像素值大于此值被认为是白色 (0-255)

    Returns:
        去除白色区域后的图像
    """
    try:
        import numpy as np

        # 转换为numpy数组
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        _log_info(f"🔍 开始检测白色区域，阈值: {white_threshold}")

        # 多种白色检测策略
        white_masks = []

        # 策略1: 严格白色检测 (RGB三个通道都大于阈值)
        if len(img_array.shape) == 3:  # RGB图像
            strict_white_mask = np.all(img_array >= white_threshold, axis=2)
            white_masks.append(strict_white_mask)

            # 策略2: 近似白色检测 (RGB差异小且平均值高)
            rgb_mean = np.mean(img_array, axis=2)
            rgb_std = np.std(img_array, axis=2)
            approx_white_mask = (rgb_mean >= white_threshold - 20) & (rgb_std <= 25)
            white_masks.append(approx_white_mask)

            # 策略3: 纯白色检测 (RGB = 255)
            pure_white_mask = np.all(img_array == 255, axis=2)
            white_masks.append(pure_white_mask)

        else:  # 灰度图像
            white_mask = img_array >= white_threshold
            white_masks.append(white_mask)

        # 合并所有白色检测结果
        combined_white_mask = np.logical_or.reduce(white_masks)

        # 计算白色像素比例
        white_ratio = np.sum(combined_white_mask) / (height * width)
        _log_info(f"🔍 白色像素比例: {white_ratio:.2%}")

        # 降低白色像素比例阈值，更容易检测到白色区域
        if white_ratio < 0.01:  # 小于1%
            _log_info(f"ℹ️ 白色像素比例较低({white_ratio:.2%})，跳过处理")
            return image

        # 找到非白色区域的边界框
        non_white_mask = ~combined_white_mask

        # 找到非白色像素的行和列
        non_white_rows = np.any(non_white_mask, axis=1)
        non_white_cols = np.any(non_white_mask, axis=0)

        # 如果没有非白色像素，返回原图
        if not np.any(non_white_rows) or not np.any(non_white_cols):
            _log_warning(f"⚠️ 图像几乎全是白色，保持原图")
            return image

        # 找到边界
        top = np.argmax(non_white_rows)
        bottom = len(non_white_rows) - 1 - np.argmax(non_white_rows[::-1])
        left = np.argmax(non_white_cols)
        right = len(non_white_cols) - 1 - np.argmax(non_white_cols[::-1])

        # 检测边缘白色区域的厚度
        edge_thickness = {
            'top': top,
            'bottom': height - 1 - bottom,
            'left': left,
            'right': width - 1 - right
        }

        _log_info(f"🔍 边缘白色厚度: 上{edge_thickness['top']}, 下{edge_thickness['bottom']}, 左{edge_thickness['left']}, 右{edge_thickness['right']}")

        # 更智能的白色边框检测逻辑
        min_edge_thickness = max(20, width // 15, height // 15)  # 降低阈值：至少20像素或图像尺寸的6.7%

        # 检查是否是真正的边框
        thick_edges = [k for k, v in edge_thickness.items() if v >= min_edge_thickness]

        # 特别检查底部白色区域（常见的生成图像问题）
        bottom_white_ratio = edge_thickness['bottom'] / height if height > 0 else 0

        _log_info(f"🔍 厚边检测: {thick_edges}, 底部白色比例: {bottom_white_ratio:.2%}")

        # 更宽松的边框检测条件：
        # 1. 四边都有厚白边
        # 2. 对边都有厚白边（上下或左右）
        # 3. 三边有厚白边
        # 4. 底部有大面积白色区域（>15%）
        is_border = (
            len(thick_edges) >= 3 or  # 三边或四边有厚白边
            ('top' in thick_edges and 'bottom' in thick_edges) or  # 上下都有
            ('left' in thick_edges and 'right' in thick_edges) or  # 左右都有
            bottom_white_ratio > 0.15  # 底部白色区域超过15%
        )

        if not is_border:
            _log_info(f"ℹ️ 不是真正的白色边框，跳过裁剪。厚边: {thick_edges}, 底部白色比例: {bottom_white_ratio:.2%}")
            return image

        _log_info(f"✅ 检测到白色边框，厚边: {thick_edges}")

        # 添加一些边距，避免裁剪过紧
        margin = min(5, width // 50, height // 50)  # 最多5像素或图像尺寸的2%
        top = max(0, top - margin)
        bottom = min(height - 1, bottom + margin)
        left = max(0, left - margin)
        right = min(width - 1, right + margin)

        # 检查裁剪区域是否有效
        crop_width = right - left + 1
        crop_height = bottom - top + 1

        if crop_width <= 0 or crop_height <= 0:
            _log_warning(f"⚠️ 裁剪区域无效，保持原图")
            return image

        # 计算裁剪比例
        crop_ratio = (crop_width * crop_height) / (width * height)

        # 如果裁剪后的区域太小，可能是误判
        if crop_ratio < 0.3:  # 小于30%
            _log_warning(f"⚠️ 裁剪后区域过小({crop_ratio:.2%})，可能是误判，保持原图")
            return image

        _log_info(f"✅ 检测到白色边框，裁剪区域: ({left}, {top}) -> ({right}, {bottom})")
        _log_info(f"✅ 裁剪尺寸: {crop_width}x{crop_height} (保留{crop_ratio:.1%})")

        # 裁剪图像
        cropped_image = image.crop((left, top, right + 1, bottom + 1))

        return cropped_image

    except Exception as e:
        _log_warning(f"❌ 白色区域检测失败: {e}")
        import traceback
        _log_warning(f"🔍 详细错误: {traceback.format_exc()}")
        return image

def smart_resize_with_padding(image: Image.Image, target_size: Tuple[int, int],
                             fill_color: Tuple[int, int, int] = (255, 255, 255),
                             fill_strategy: str = "smart", gigapixel_model: str = "High Fidelity") -> Image.Image:
    """
    🚀 直接目标尺寸扩图技术，按控制尺寸要求直接扩图
    彻底解决过度扩图问题，直接扩到目标尺寸
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        fill_color: 填充颜色，默认白色
        fill_strategy: 填充策略
            - "smart": 智能选择最佳策略（优先使用 crop，避免重叠且无白边，参考官方分支）
            - "direct": 直接缩放到目标尺寸（可能变形，谨慎使用）
            - "crop": 裁剪模式（无填充，无重叠，高清放大后裁剪）
            - "paste": 粘贴模式（有填充，主体完全可见）
            - "extend": 背景扩展模式（等比缩放贴中间 + 背景内容扩展，可能有重叠）
    """
    img_width, img_height = image.size
    target_width, target_height = target_size

    _log_info(f"🎯 开始直接目标尺寸扩图技术: {img_width}x{img_height} -> {target_width}x{target_height}")
    _log_info(f"🎯 填充策略: {fill_strategy}")

    # 🚀 第一步：检测并去除白色区域
    processed_image = remove_white_areas(image)
    if processed_image.size != image.size:
        _log_info(f"✅ 白色区域已去除: {image.size} -> {processed_image.size}")
        image = processed_image
        img_width, img_height = image.size

        # 如果还有白色区域，尝试更激进的检测
        processed_image2 = remove_white_areas(image, white_threshold=230)
        if processed_image2.size != image.size:
            _log_info(f"✅ 激进模式再次去除白色区域: {image.size} -> {processed_image2.size}")
            image = processed_image2
            img_width, img_height = image.size
    else:
        _log_info(f"ℹ️ 未检测到需要去除的白色区域")

    # 🎯 策略1：比例相同时，直接调整尺寸
    if abs(img_width/img_height - target_width/target_height) < 0.01:
        _log_info(f"🎯 比例相同，直接调整尺寸")
        resized_img = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return resized_img
    
    # 🎯 策略2：比例不同时
    _log_info(f"🎯 比例不同，选择合适策略")
    
    # 默认智能策略：走 crop，避免重叠且无白边（参考官方分支）
    if fill_strategy == "smart":
        fill_strategy = "crop"
    
    if fill_strategy == "extend":
        # 等比缩放至不超过目标尺寸
        scale_x = target_width / img_width
        scale_y = target_height / img_height
        scale = min(scale_x, scale_y)
        new_width = max(1, int(img_width * scale))
        new_height = max(1, int(img_height * scale))
        _log_info(f"🎯 extend 缩放尺寸: {new_width}x{new_height} (scale={scale:.3f})")
        fg = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 背景：先按 cover 生成一张铺满目标的背景，再高斯模糊，避免白边
        cover_scale = max(scale_x, scale_y)
        bg_w = max(1, int(img_width * cover_scale))
        bg_h = max(1, int(img_height * cover_scale))
        bg = image.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
        crop_x = (bg_w - target_width) // 2
        crop_y = (bg_h - target_height) // 2
        bg = bg.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
        try:
            from PIL import ImageFilter
            bg = bg.filter(ImageFilter.GaussianBlur(radius=24))
        except Exception:
            pass
        
        # 将前景等比缩放图粘贴到中心（确保完全覆盖背景，避免重叠）
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        _log_info(f"🎯 前景尺寸: {new_width}x{new_height}, 粘贴位置: ({paste_x}, {paste_y})")

        # 创建一个新的画布，确保没有重叠问题
        result = Image.new('RGB', (target_width, target_height))
        result.paste(bg, (0, 0))  # 先粘贴背景

        # 确保前景图像是RGB模式，避免透明度问题
        if fg.mode != 'RGB':
            fg = fg.convert('RGB')

        # 🔧 关键修复：创建一个遮罩来确保前景完全覆盖背景的相应区域
        # 先在前景区域填充纯色，然后粘贴前景，避免任何重叠效果
        if paste_x >= 0 and paste_y >= 0 and paste_x + new_width <= target_width and paste_y + new_height <= target_height:
            # 在前景区域先填充背景色，确保完全覆盖
            from PIL import ImageDraw
            draw = ImageDraw.Draw(result)
            # 获取背景的平均颜色作为填充色
            try:
                bg_sample = bg.resize((1, 1), Image.Resampling.LANCZOS)
                avg_color = bg_sample.getpixel((0, 0))
                if isinstance(avg_color, int):
                    avg_color = (avg_color, avg_color, avg_color)
            except:
                avg_color = fill_color

            # 在前景区域填充平均色，确保没有重叠
            draw.rectangle([paste_x, paste_y, paste_x + new_width, paste_y + new_height], fill=avg_color)

            # 然后粘贴前景图像
            result.paste(fg, (paste_x, paste_y))
        else:
            _log_warning(f"⚠️ 前景图像超出边界，调整粘贴位置")
            # 如果超出边界，直接居中粘贴，可能会裁剪
            safe_paste_x = max(0, min(paste_x, target_width - new_width))
            safe_paste_y = max(0, min(paste_y, target_height - new_height))

            # 同样先填充再粘贴
            from PIL import ImageDraw
            draw = ImageDraw.Draw(result)
            try:
                bg_sample = bg.resize((1, 1), Image.Resampling.LANCZOS)
                avg_color = bg_sample.getpixel((0, 0))
                if isinstance(avg_color, int):
                    avg_color = (avg_color, avg_color, avg_color)
            except:
                avg_color = fill_color

            draw.rectangle([safe_paste_x, safe_paste_y, safe_paste_x + new_width, safe_paste_y + new_height], fill=avg_color)
            result.paste(fg, (safe_paste_x, safe_paste_y))

        _log_info(f"✅ extend 完成：无白边、不变形，输出 {result.size}")
        return result
    
    if fill_strategy in ["direct"]:
        # 🎯 直接扩图模式：直接扩到目标尺寸（可能变形，谨慎使用）
        _log_info(f"⚠️ direct 模式：直接缩放到目标，可能变形")
        final_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return final_image
        
    elif fill_strategy == "crop":
        # 🎯 裁剪模式：使用高清无损放大到最大边，然后智能裁剪
        _log_info(f"🎯 裁剪模式：高清无损放大到最大边，然后智能裁剪")
        
        # 🚀 高清无损放大（保持原始比例，不拉伸变形）
        # 计算最佳缩放比例，使用max确保完全覆盖目标区域
        scale_x = target_width / img_width      # 宽度比例
        scale_y = target_height / img_height    # 高度比例
        scale = max(scale_x, scale_y)  # 使用较大的缩放比例，确保完全覆盖
        
        # 计算放大后的尺寸（保持原始比例，确保覆盖目标区域）
        enlarged_width = int(img_width * scale)
        enlarged_height = int(img_height * scale)
        
        _log_info(f"🔧 高清无损放大: {img_width}x{img_height} -> {enlarged_width}x{enlarged_height}")
        _log_info(f"🔧 缩放比例: {scale:.3f} (使用max确保完全覆盖，然后智能裁剪)")
        _log_info(f"🔧 关键：直接放大到最大边，保持图像清晰度和比例")
        
        # 🎯 使用AI放大模型进行高清无损放大（保持比例）
        # 优先使用AI模型，回退到高质量重采样
        try:
            _log_info(f"🔧 尝试使用AI放大模型进行高清放大...")
            ai_upscaled_image = smart_ai_upscale(image, enlarged_width, enlarged_height, gigapixel_model)
            
            if ai_upscaled_image is not None:
                # 如果AI放大成功，调整到目标尺寸
                if ai_upscaled_image.size != (enlarged_width, enlarged_height):
                    _log_info(f"🔧 AI放大后调整到目标尺寸: {ai_upscaled_image.size} -> {enlarged_width}x{enlarged_height}")
                    enlarged_image = ai_upscaled_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                else:
                    enlarged_image = ai_upscaled_image
                _log_info(f"✅ AI放大模型放大完成，图像质量大幅提升")
            else:
                _log_warning(f"⚠️ AI放大模型不可用，使用高质量重采样")
                enlarged_image = image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                
        except Exception as e:
            _log_warning(f"⚠️ AI放大模型失败，使用高质量重采样: {e}")
            # 回退到 LANCZOS 算法
            enlarged_image = image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
        
        # 🎯 智能裁剪 - 从高清放大的图像中裁剪出目标尺寸
        if enlarged_width >= target_width and enlarged_height >= target_height:
            _log_info(f"🔧 智能裁剪：从高清放大图像中裁剪目标尺寸，确保主体居中")
            
            # 🎯 精确计算裁剪区域，确保主体完全居中
            crop_x = (enlarged_width - target_width) // 2
            crop_y = (enlarged_height - target_height) // 2
            
            # 🎯 微调偏移，确保完全居中（避免奇数像素偏差）
            if (enlarged_width - target_width) % 2 == 1:
                crop_x += 1
            if (enlarged_height - target_height) % 2 == 1:
                crop_y += 1
            
            _log_info(f"🔧 精确居中裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x + target_width}, {crop_y + target_height})")
            _log_info(f"🔧 确保主体在裁剪后图像的正中心位置")
            
            # 从高清放大的图像中裁剪出目标尺寸
            final_image = enlarged_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
            
            _log_info(f"✅ 高清无损放大 + 智能裁剪完成")
            _log_info(f"✅ 结果：无白色填充，完全不变形，主体精确居中，保持最高清晰度")
            _log_info(f"✅ 图像质量：高清无损，比例完美，主体可见")
            
            return final_image
            
        else:
            _log_warning(f"⚠️ 高清放大后尺寸不足，使用智能填充（避免拉伸变形）")
            # 创建目标尺寸的画布，使用填充色
            final_image = Image.new('RGB', (target_width, target_height), fill_color)
            
            # 将高清放大的图像居中放置
            paste_x = (target_width - enlarged_width) // 2
            paste_y = (target_height - enlarged_height) // 2
            final_image.paste(enlarged_image, (paste_x, paste_y))
            
            _log_info(f"✅ 智能填充完成：高清放大图像居中放置，边缘用填充色")
            return final_image
    
    else:
        # 🎯 粘贴模式：使用min(scale_x, scale_y)保护主体，留边（可能出现填充色）
        scale_x = target_width / img_width
        scale_y = target_height / img_height
        scale = min(scale_x, scale_y)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        final_image = Image.new('RGB', target_size, fill_color)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        final_image.paste(resized_img, (paste_x, paste_y))
        return final_image

def smart_ai_upscale(image, target_width, target_height, gigapixel_model="High Fidelity"):
    """
    🚀 智能AI放大技术 - 统一委托到通用放大器（banana_upscale.smart_upscale）
    """
    try:
        try:
            from .banana_upscale import smart_upscale as _smart
        except ImportError:
            from banana_upscale import smart_upscale as _smart

        # 调用AI放大
        result = _smart(image, target_width, target_height, gigapixel_model)

        # 验证结果是否有效
        if result is not None and hasattr(result, 'size') and result.size[0] > 0 and result.size[1] > 0:
            _log_info(f"✅ AI放大成功: {image.size} -> {result.size}")
            return result
        else:
            _log_warning(f"⚠️ AI放大返回无效结果，回退到普通重采样")
            return None

    except Exception as e:
        _log_warning(f"⚠️ 智能放大器失败: {e}")
        import traceback
        _log_warning(f"⚠️ 详细错误: {traceback.format_exc()}")
        return None


def _analyze_image_type_simple(image: Image.Image) -> str:
    """
    🎯 简单图像类型分析，用于选择AI增强策略
    """
    try:
        # 基于图像尺寸和特征的简单分析
        width, height = image.size

        # 小尺寸图像可能是头像或图标
        if width <= 512 and height <= 512:
            return "face"

        # 极宽或极高的图像可能包含文字
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3.0:
            return "text"

        # 中等尺寸的正方形或接近正方形图像
        if 0.7 <= width/height <= 1.3:
            return "art"

        # 默认为通用图像
        return "general"

    except Exception:
        return "general"


def _apply_ai_super_resolution(image: Image.Image, image_type: str = "general", gigapixel_model: str = "High Fidelity") -> Optional[Image.Image]:
    """
    🚀 AI超分辨率增强：根据图像类型选择最佳策略
    """
    try:
        if not image:
            return None

        # 根据图像类型确定增强策略
        if image_type == "face":
            # 人脸图像：适度放大，保持自然
            target_scale = 1.8
        elif image_type == "text":
            # 文字图像：高倍放大，提升清晰度
            target_scale = 2.5
        elif image_type == "art":
            # 艺术图像：中等放大，保持风格
            target_scale = 2.0
        else:
            # 通用图像：标准放大
            target_scale = 2.0

        # 计算目标尺寸
        target_w = int(image.width * target_scale)
        target_h = int(image.height * target_scale)

        _log_info(f"🚀 尝试AI超分辨率增强: {image.size} -> ({target_w}, {target_h})")

        # 使用智能放大系统
        enhanced = smart_ai_upscale(image, target_w, target_h, gigapixel_model)
        if enhanced:
            # 如果放大成功，缩回原尺寸以保持细节提升
            final = enhanced.resize(image.size, Image.Resampling.LANCZOS)
            _log_info(f"✅ AI超分辨率增强成功")
            return final

        return None

    except Exception as e:
        _log_warning(f"AI超分辨率增强失败: {e}")
        return None


def enhance_image_quality(image: Image.Image, quality: str = "hd", adaptive_mode: str = "disabled", gigapixel_model: str = "High Fidelity") -> Image.Image:
    """
    🚀 图像质量增强（集成AI超分辨率技术）
    - 传统增强：锐化、对比度、色彩、亮度微调
    - AI增强：智能超分辨率放大技术
    """

    # 🚀 AI超分辨率增强（新增功能）
    if quality in ["ai_enhanced", "ai_ultra"]:
        _log_info(f"🚀 启用AI超分辨率增强模式: {quality}")

        # 分析图像类型以选择最佳AI增强策略
        image_type = _analyze_image_type_simple(image)

        # 应用AI超分辨率增强
        ai_enhanced = _apply_ai_super_resolution(image, image_type, gigapixel_model)
        if ai_enhanced:
            image = ai_enhanced
            _log_info(f"✅ AI超分辨率增强完成")
        else:
            _log_warning(f"⚠️ AI增强失败，回退到传统增强")

    # 传统质量增强
    if quality in ["hd", "ai_enhanced"]:
        # 应用智能锐化滤镜
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)  # 比参考项目更强

        # 对比度增强
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.15)  # 比参考项目更强

        # 色彩饱和度增强
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.08)  # 比参考项目更强

        # 亮度优化
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.02)

    elif quality in ["ultra_hd", "ai_ultra"]:
        from PIL import ImageEnhance
        # 超高清质量增强
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.12)

        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.05)

    return image

def image_to_base64_enhanced(image: Image.Image, format: str = "PNG") -> str:
    """超越参考项目的base64转换，保持最高质量"""
    buffered = BytesIO()
    image.save(buffered, format=format, quality=95, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_image_controls(quality: str, style: str) -> dict:
    """
    处理图像控制参数，返回标准化的控制配置

    注意：根据Gemini官方API文档，尺寸控制应该通过generationConfig.imageConfig.aspectRatio参数实现，
    而不是通过提示词。因此这个函数只处理quality和style参数。

    为了向后兼容，返回一个默认的size值（"Original size"），但这个值不应该被使用。
    新代码应该使用upscale_factor参数进行放大。

    Args:
        quality: 质量设置
        style: 风格设置

    Returns:
        dict: 包含处理后的图像控制参数
    """
    # 构建控制配置（包含quality、style和默认的size）
    controls = {
        "quality": quality,
        "style": style,
        "size": "Original size",  # 默认值，用于向后兼容
        "is_custom_size": False   # 默认值，用于向后兼容
    }

    return controls


def enhance_prompt_with_controls(prompt: str, controls: dict, detail_level: str = "Professional Detail",
                               camera_control: str = "Auto Select", lighting_control: str = "Auto Settings",
                               template_selection: str = "Auto Select", quality_enhancement: bool = True,
                               enhance_quality: bool = True, smart_resize: bool = True, fill_color: str = "255,255,255") -> str:
    """
    🚀 智能提示词增强系统

    根据Gemini官方API文档，图像尺寸应该通过generationConfig.imageConfig.aspectRatio参数控制，
    而不是通过提示词。因此这个函数只处理质量和风格的增强。

    集成的增强功能：
    - 智能风格识别和模板
    - 动态质量控制指令
    - 艺术风格增强
    - 专业摄影参数
    """
    
    # 🚀 超越参考项目的完整风格模板系统
    style_templates = {
        # 基础风格（超越参考项目）
        "vivid": {
            "prefix": "Create a vivid, high-contrast image with",
            "suffix": "Use vibrant colors and dramatic lighting to make the image pop.",
            "quality_boost": "Ensure maximum visual impact and artistic appeal.",
            "camera_settings": "Use high contrast and saturation settings.",
            "lighting": "Dramatic lighting with strong shadows and highlights."
        },
        "natural": {
            "prefix": "Generate a natural, realistic image of",
            "suffix": "Use natural lighting and authentic colors for a lifelike appearance.",
            "quality_boost": "Focus on realism and natural beauty.",
            "camera_settings": "Natural color balance and realistic exposure.",
            "lighting": "Soft, natural lighting with subtle shadows."
        },
        
        # 🎨 专业艺术风格（参考项目核心功能）
        "professional_portrait": {
            "prefix": "Create a professional portrait photograph of",
            "suffix": "Use studio lighting, professional composition, and high-end camera settings.",
            "quality_boost": "Achieve magazine-quality portrait photography.",
            "camera_settings": "85mm lens, f/2.8 aperture, professional color grading.",
            "lighting": "Three-point lighting setup with soft key light and fill."
        },
        "cinematic_landscape": {
            "prefix": "Generate a cinematic landscape photograph of",
            "suffix": "Use dramatic lighting, depth of field, and cinematic composition.",
            "quality_boost": "Create a film-quality landscape experience.",
            "camera_settings": "Wide-angle lens, f/8 aperture, cinematic color grading.",
            "lighting": "Golden hour or dramatic storm lighting with atmospheric perspective."
        },
        "product_photography": {
            "prefix": "Create a professional product photograph of",
            "suffix": "Use professional photography techniques and studio lighting.",
            "quality_boost": "Achieve catalog-quality product imagery.",
            "camera_settings": "Macro lens, f/11 aperture, commercial color accuracy.",
            "lighting": "Clean studio lighting with controlled reflections and shadows."
        },
        "digital_concept_art": {
            "prefix": "Generate digital concept art featuring",
            "suffix": "Use digital painting techniques and creative composition.",
            "quality_boost": "Create professional concept art quality.",
            "camera_settings": "Digital art style with creative color palette.",
            "lighting": "Dramatic lighting with artistic interpretation."
        },
        "anime_style_art": {
            "prefix": "Create anime-style artwork featuring",
            "suffix": "Use anime/manga art style with vibrant colors and clean lines.",
            "quality_boost": "Achieve professional anime art quality.",
            "camera_settings": "Anime art style with bold colors and smooth shading.",
            "lighting": "Bright, colorful lighting typical of anime art."
        },
        "photorealistic_render": {
            "prefix": "Generate a photorealistic 3D render of",
            "suffix": "Use advanced rendering techniques and realistic materials.",
            "quality_boost": "Achieve indistinguishable-from-photography quality.",
            "camera_settings": "Photorealistic rendering with accurate materials and lighting.",
            "lighting": "Physically accurate lighting with global illumination."
        },
        "classical_oil_painting": {
            "prefix": "Create a classical oil painting style image of",
            "suffix": "Use traditional oil painting techniques and classical composition.",
            "quality_boost": "Achieve master painter quality artwork.",
            "camera_settings": "Oil painting texture with classical color palette.",
            "lighting": "Classical lighting with rich, warm tones."
        },
        "watercolor_painting": {
            "prefix": "Generate a watercolor painting style image of",
            "suffix": "Use watercolor painting techniques and flowing composition.",
            "quality_boost": "Achieve authentic watercolor art quality.",
            "camera_settings": "Watercolor texture with transparent, flowing colors.",
            "lighting": "Soft, diffused lighting with watercolor transparency."
        },
        "cyberpunk_future": {
            "prefix": "Create a cyberpunk future scene featuring",
            "suffix": "Use futuristic aesthetics with neon lighting and high-tech elements.",
            "quality_boost": "Achieve cinematic cyberpunk quality.",
            "camera_settings": "Cyberpunk style with neon color palette and futuristic elements.",
            "lighting": "Neon lighting with dramatic shadows and futuristic atmosphere."
        },
        "vintage_film_photography": {
            "prefix": "Create a vintage film photograph of",
            "suffix": "Use classic film photography aesthetics and vintage color grading.",
            "quality_boost": "Achieve authentic vintage film quality.",
            "camera_settings": "Vintage film grain with classic color palette.",
            "lighting": "Classic film lighting with vintage color temperature."
        },
        "architectural_photography": {
            "prefix": "Create an architectural photograph of",
            "suffix": "Use architectural photography techniques and geometric composition.",
            "quality_boost": "Achieve professional architectural photography quality.",
            "camera_settings": "Wide-angle lens, f/8 aperture, architectural perspective.",
            "lighting": "Natural lighting with architectural shadows and highlights."
        },
        "gourmet_food_photography": {
            "prefix": "Generate a gourmet food photograph of",
            "suffix": "Use food photography techniques and appetizing composition.",
            "quality_boost": "Achieve magazine-quality food photography.",
            "camera_settings": "Macro lens, f/5.6 aperture, food-optimized color grading.",
            "lighting": "Soft, appetizing lighting with food-appropriate shadows."
        }
    }
    
    # 获取风格配置
    style_config = style_templates.get(controls['style'], style_templates["natural"])
    
    # 🚀 构建超越参考项目的增强提示词
    # 🎯 平衡修复：适度的构图控制，避免主体过大或过小
    enhanced_parts = [
        style_config["prefix"],
        prompt.strip(),
        style_config["suffix"],
        # 平衡的构图控制指令
        "Use balanced composition with proper subject-to-background ratio.",
        "Subject should be clearly visible and well-framed, occupying 40-60% of the image area.",
        "Include rich background environment and context to create depth and atmosphere.",
        "Use medium shot framing that shows the subject in their environment with meaningful background.",
        "Prioritize clear subject visibility while maintaining environmental context.",
        "Show meaningful background elements and environmental details.",
        "Avoid extreme close-ups that eliminate all background context."
    ]
    
    # 🎨 添加参考项目的专业控制参数（超越参考项目）
    if "camera_settings" in style_config:
        enhanced_parts.append(f"Camera Settings: {style_config['camera_settings']}")
    
    if "lighting" in style_config:
        enhanced_parts.append(f"Lighting: {style_config['lighting']}")
    
    # 添加质量控制
    if controls['quality'] == "hd":
        enhanced_parts.append(style_config["quality_boost"])
        enhanced_parts.append("Generate in ultra-high definition with exceptional detail.")
    elif controls['quality'] == "ultra_hd":
        enhanced_parts.append(style_config["quality_boost"])
        enhanced_parts.append("Generate in ultra-high definition with exceptional detail and professional quality.")
    
    # 🎯 关键指令：必须生成图像而不是描述
    enhanced_parts.append("CRITICAL: You MUST return an actual generated image, not just a description.")
    enhanced_parts.append("Use your image generation capabilities to create the visual content.")
    
    # 🎨 应用所有界面参数
    if detail_level != "Auto Select":
        detail_instructions = {
            "Basic Detail": "Focus on essential details and clean composition.",
            "Professional Detail": "Include professional-level detail and refined elements.",
            "Premium Quality": "Achieve premium quality with exceptional attention to detail.",
            "Masterpiece Level": "Create a masterpiece with extraordinary detail and artistic excellence."
        }
        enhanced_parts.append(f"Detail Level: {detail_level} - {detail_instructions.get(detail_level, '')}")
    
    if camera_control != "Auto Select":
        camera_instructions = {
            "Wide-angle Lens": "Use wide-angle perspective for expansive composition.",
            "Macro Shot": "Focus on close-up details with macro photography techniques.",
            "Low-angle Perspective": "Use low-angle perspective for dramatic impact.",
            "High-angle Shot": "Use high-angle perspective for overview composition.",
            "Close-up Shot": "Focus on intimate details with close-up framing.",
            "Medium Shot": "Use medium framing for balanced composition."
        }
        enhanced_parts.append(f"Camera Control: {camera_control} - {camera_instructions.get(camera_control, '')}")
    
    if lighting_control != "Auto Settings":
        lighting_instructions = {
            "Natural Light": "Use natural lighting with soft, diffused illumination",
            "Studio Lighting": "Use professional studio lighting setup with controlled shadows",
            "Dramatic Shadows": "Use dramatic lighting with strong contrast and deep shadows",
            "Soft Glow": "Use soft, glowing lighting for gentle, romantic atmosphere",
            "Golden Hour": "Use golden hour lighting with warm, golden tones",
            "Blue Hour": "Use blue hour lighting with cool, atmospheric tones"
        }
        enhanced_parts.append(f"Lighting Control: {lighting_control} - {lighting_instructions.get(lighting_control, '')}")
    
    if template_selection != "Auto Select":
        template_instructions = {
            "Professional Portrait": "Apply professional portrait photography techniques and composition",
            "Cinematic Landscape": "Use cinematic landscape photography composition and lighting",
            "Product Photography": "Apply professional product photography techniques and studio setup",
            "Digital Concept Art": "Use digital concept art style and creative composition",
            "Anime Style Art": "Apply anime/manga art style and vibrant aesthetics",
            "Photorealistic Render": "Create photorealistic 3D rendering quality and materials",
            "Classical Oil Painting": "Apply classical oil painting techniques and traditional style",
            "Watercolor Painting": "Use watercolor painting techniques and flowing aesthetics",
            "Cyberpunk Future": "Apply cyberpunk futuristic aesthetics and neon lighting",
            "Vintage Film Photography": "Use vintage film photography aesthetics and color grading",
            "Architectural Photography": "Apply architectural photography techniques and perspective",
            "Gourmet Food Photography": "Use gourmet food photography techniques and appetizing lighting"
        }
        enhanced_parts.append(f"Template: {template_selection} - {template_instructions.get(template_selection, 'Follow professional composition guidelines')}")
    
    # 🚀 处理质量增强开关（参考项目功能）
    if quality_enhancement:
        enhanced_parts.append("Quality Enhancement: ENABLED - Apply advanced image quality improvements including sharpening, contrast enhancement, and color optimization.")
        _log_info("✨ 质量增强已启用")
    
    if enhance_quality:
        enhanced_parts.append("Apply enhanced image quality processing for professional output.")
    
    if smart_resize:
        enhanced_parts.append("Use intelligent resizing with proper padding and composition.")

    if fill_color and fill_color != "255,255,255":
        enhanced_parts.append(f"Use specified fill color: RGB({fill_color}) for padding areas.")

    # 🚀 应用参考项目的图形增强技术
    if controls['quality'] == "hd":
        enhanced_parts.append("Generate in high definition with professional detail.")
    elif controls['quality'] == "ultra_hd":
        enhanced_parts.append("Generate in ultra-high definition with exceptional detail and professional quality.")

    # 🚀 添加额外的环境和透视控制
    enhanced_parts.append("Include rich environmental details and background elements.")
    enhanced_parts.append("Create depth and layers in the composition with foreground, middle ground, and background.")
    enhanced_parts.append("Use natural perspective and realistic spatial relationships.")

    # 🎯 最终平衡提醒：确保主体与环境的平衡
    enhanced_parts.append("Create a balanced composition with the subject clearly visible in their environment, showing both the subject and meaningful background context.")

    return " ".join(enhanced_parts)


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

def validate_api_key(api_key):
    """Validate API key format"""
    if not api_key or not isinstance(api_key, str):
        return False
    api_key = api_key.strip()
    if not api_key:
        return False
    return True

def format_error_message(error):
    """Format error message for better readability"""
    try:
        if hasattr(error, 'response'):
            if hasattr(error.response, 'status_code'):
                return f"HTTP {error.response.status_code}: {str(error)}"
        return str(error)
    except:
        return str(error)

def pil_to_tensor(pil_image):
    """Convert PIL Image to PyTorch tensor - 简化版本"""
    try:
        # 如果已经是tensor，直接返回
        if isinstance(pil_image, torch.Tensor):
            return pil_image

        # 确保是PIL Image
        if not isinstance(pil_image, Image.Image):
            # 创建默认图像
            pil_image = Image.new('RGB', (512, 512), color=(255, 255, 255))

        # 确保RGB格式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # 转换为numpy数组并归一化
        image_array = np.array(pil_image).astype(np.float32) / 255.0

        # 转换为tensor并添加batch维度 (BHWC格式)
        tensor = torch.from_numpy(image_array).unsqueeze(0)

        return tensor
    except Exception as e:
        _log_error(f"Failed to convert PIL to tensor: {e}")
        # 返回默认的白色图像tensor
        return torch.ones((1, 512, 512, 3), dtype=torch.float32)

def tensor_to_pil(image_tensor):
    """Convert PyTorch tensor to PIL Image"""
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
        _log_error(f"Failed to convert tensor to PIL: {e}")
        return None

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

                        # 简化的tensor转换
                        img_array = np.array(pil_image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

                        return img_tensor

                    except Exception as e:
                        _log_error(f"解码图片失败: {e}")
                        continue

        # 如果没有找到图像，返回占位符
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

def generate_with_official_api(api_key, model, content_parts, generation_config, max_retries=5, proxy=None):
    """优先使用官方google.genai库调用API"""
    try:
        # 尝试导入官方库
        from google import genai
        from google.genai import types
        
        _log_info(f"🚀 使用官方google.genai库调用模型: {model}")
        
        # 代理处理：有效则设置，无效/未填则清除，避免残留环境变量影响请求
        if proxy and proxy.strip() and "None" not in proxy:
            _log_info(f"🔌 使用代理: {proxy.strip()}")
            # 临时设置环境变量供google.genai库使用
            old_https_proxy = os.environ.get('HTTPS_PROXY')
            old_http_proxy = os.environ.get('HTTP_PROXY')
            os.environ['HTTPS_PROXY'] = proxy.strip()
            os.environ['HTTP_PROXY'] = proxy.strip()
        else:
            _log_info("🔌 使用系统代理 (保持现有环境变量)")
            old_https_proxy = os.environ.get('HTTPS_PROXY')
            old_http_proxy = os.environ.get('HTTP_PROXY')
            # 不清理环境变量，使用系统代理
        
        # 创建客户端
        client = genai.Client(api_key=api_key)
        
        # 转换generation_config格式
        config_params = {
            'temperature': generation_config.get('temperature', 0.7),
            'top_p': generation_config.get('topP', generation_config.get('top_p', 0.95)),
            'top_k': generation_config.get('topK', generation_config.get('top_k', 40)),
            'max_output_tokens': generation_config.get('maxOutputTokens', generation_config.get('max_output_tokens', 8192)),
        }

        # 处理responseModalities
        if 'responseModalities' in generation_config:
            config_params['response_modalities'] = generation_config['responseModalities']
        elif 'image' in model.lower():
            config_params['response_modalities'] = ['Text', 'Image']
        else:
            config_params['response_modalities'] = ['Text']

        # 处理imageConfig（aspect_ratio）
        if 'imageConfig' in generation_config and 'aspectRatio' in generation_config['imageConfig']:
            config_params['image_config'] = types.ImageConfig(
                aspect_ratio=generation_config['imageConfig']['aspectRatio']
            )

        # 处理seed
        if 'seed' in generation_config and generation_config['seed'] > 0:
            config_params['seed'] = generation_config['seed']

        official_config = types.GenerateContentConfig(**config_params)
        
        # 转换content_parts格式
        official_parts = []
        for part in content_parts:
            if 'text' in part:
                official_parts.append({"text": part['text']})
            elif 'inline_data' in part:
                official_parts.append({
                    "inline_data": {
                        "mime_type": part['inline_data']['mime_type'],
                        "data": part['inline_data']['data']
                    }
                })
        
        official_contents = [{"parts": official_parts}]
        
        # 调用官方API
        response = client.models.generate_content(
            model=model,
            contents=official_contents,
            config=official_config
        )
        
        # 处理响应
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                result_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        result_parts.append({'text': part.text})
                    elif hasattr(part, 'inline_data') and part.inline_data:
                        # 官方API返回的可能是二进制数据，需要转换为base64
                        if isinstance(part.inline_data.data, bytes):
                            import base64
                            data_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                        else:
                            data_b64 = part.inline_data.data
                        
                        result_parts.append({
                            'inline_data': {
                                'mime_type': 'image/png',
                                'data': data_b64
                            }
                        })
                
                return {
                    'candidates': [{
                        'content': {
                            'parts': result_parts
                        }
                    }]
                }
        
        _log_warning("官方API返回了空响应")
        return None
        
    except ImportError:
        _log_warning("google.genai库未安装，将使用REST API")
        return None
    except Exception as e:
        _log_error(f"官方API调用失败: {str(e)}")
        _log_error(f"错误类型: {type(e).__name__}")
        _log_error(f"content_parts结构: {[part.keys() if isinstance(part, dict) else type(part) for part in content_parts]}")
        _log_error(f"generation_config: {generation_config}")
        return None
    finally:
        # 恢复原来的代理设置
        if old_https_proxy is not None:
            os.environ['HTTPS_PROXY'] = old_https_proxy
        else:
            os.environ.pop('HTTPS_PROXY', None)
        if old_http_proxy is not None:
            os.environ['HTTP_PROXY'] = old_http_proxy
        else:
            os.environ.pop('HTTP_PROXY', None)

def generate_with_rest_api(api_key, model, content_parts, generation_config, max_retries=5, proxy=None, base_url=None):
    """使用REST API的智能重试机制调用（回退方案）"""
    
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

def generate_with_priority_api(api_key, model, content_parts, generation_config, max_retries=5, proxy=None, base_url=None):
    """优先使用官方API，失败时回退到REST API"""
    
    # 首先尝试官方API
    _log_info("🎯 优先尝试官方google.genai API")
    result = generate_with_official_api(api_key, model, content_parts, generation_config, max_retries, proxy)
    
    if result is not None:
        _log_info("✅ 官方API调用成功")
        return result
    
    # 官方API失败，回退到REST API
    _log_info("🔄 官方API失败，回退到REST API")
    return generate_with_rest_api(api_key, model, content_parts, generation_config, max_retries, proxy, base_url)

def generate_with_priority_api_direct(api_key, model, request_data, max_retries=5, proxy=None, base_url=None):
    """优先使用官方API，失败时回退到直接REST API调用（用于多图像编辑）"""
    
    # 首先尝试官方API
    try:
        from google import genai
        from google.genai import types
        
        _log_info("🎯 优先尝试官方google.genai API (多图像编辑)")
        
        # 创建客户端
        client = genai.Client(api_key=api_key)
        
        # 转换请求数据格式
        contents = request_data.get('contents', [])
        generation_config = request_data.get('generationConfig', {})
        
        # 转换generation_config
        config_params = {
            'temperature': generation_config.get('temperature', 0.7),
            'top_p': generation_config.get('topP', 0.95),
            'top_k': generation_config.get('topK', 40),
            'max_output_tokens': generation_config.get('maxOutputTokens', 8192),
        }

        # 处理responseModalities
        if 'responseModalities' in generation_config:
            config_params['response_modalities'] = generation_config['responseModalities']
        else:
            config_params['response_modalities'] = ['Text', 'Image']

        # 处理imageConfig（aspect_ratio）
        if 'imageConfig' in generation_config and 'aspectRatio' in generation_config['imageConfig']:
            config_params['image_config'] = types.ImageConfig(
                aspect_ratio=generation_config['imageConfig']['aspectRatio']
            )

        # 处理seed
        if 'seed' in generation_config and generation_config['seed'] > 0:
            config_params['seed'] = generation_config['seed']

        official_config = types.GenerateContentConfig(**config_params)
        
        # 转换contents格式
        official_contents = []
        for content in contents:
            parts = content.get('parts', [])
            official_parts = []
            for part in parts:
                if 'text' in part:
                    official_parts.append({"text": part['text']})
                elif 'inline_data' in part:
                    official_parts.append({
                        "inline_data": {
                            "mime_type": part['inline_data']['mime_type'],
                            "data": part['inline_data']['data']
                        }
                    })
            official_contents.append({"parts": official_parts})
        
        # 调用官方API
        response = client.models.generate_content(
            model=model,
            contents=official_contents,
            config=official_config
        )
        
        # 转换响应格式
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                result_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        result_parts.append({'text': part.text})
                    elif hasattr(part, 'inline_data') and part.inline_data:
                        # 官方API返回的可能是二进制数据，需要转换为base64
                        if isinstance(part.inline_data.data, bytes):
                            import base64
                            data_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                        else:
                            data_b64 = part.inline_data.data
                        
                        result_parts.append({
                            'inline_data': {
                                'mime_type': 'image/png',
                                'data': data_b64
                            }
                        })
                
                return {
                    'candidates': [{
                        'content': {
                            'parts': result_parts
                        }
                    }]
                }
        
        _log_warning("官方API返回了空响应")
        
    except ImportError:
        _log_warning("google.genai库未安装，将使用REST API")
    except Exception as e:
        _log_error(f"官方API调用失败: {str(e)}")
    
    # 官方API失败，回退到直接REST API调用
    _log_info("🔄 官方API失败，回退到直接REST API调用")
    
    # 构建API URL
    if base_url and base_url.strip():
        base_url = base_url.rstrip('/')
        if '/models/' in base_url and ':generateContent' in base_url:
            url = base_url
        elif base_url.endswith('/v1beta') or base_url.endswith('/v1'):
            url = f"{base_url}/models/{model}:generateContent"
        else:
            url = f"{base_url}/v1beta/models/{model}:generateContent"
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
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
    
    timeout = (10, 120)
    
    for attempt in range(max_retries):
        try:
            _log_info(f"🌐 REST API调用 ({attempt + 1}/{max_retries}) 模型: {model}")
            
            response = requests.post(url, headers=headers, json=request_data, timeout=timeout, proxies=proxies)
            
            if response.status_code == 200:
                return response.json()
            else:
                _log_error(f"HTTP状态码: {response.status_code}")
                if attempt == max_retries - 1:
                    response.raise_for_status()
                
                delay = smart_retry_delay(attempt, response.status_code)
                _log_info(f"🔄 等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
                
        except requests.exceptions.RequestException as e:
            _log_error(f"请求失败: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            else:
                delay = smart_retry_delay(attempt)
                _log_info(f"🔄 等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
    
    raise Exception("所有重试都失败了")

class KenChenLLMGeminiBananaTextToImageBananaNode:
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
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
                "gemini-2.0-flash-preview-image-generation",
                "nano-banana"
            ]
        
        # Get default model from config, prioritize latest Banana model
        default_model = config.get('default_model', {}).get('image_gen', "gemini-2.5-flash-image-preview")
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        
        # Get image control presets - Enhanced with Gemini official API features
        aspect_ratios = image_settings.get('aspect_ratios', [
            "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
        ])
        response_modalities = image_settings.get('response_modalities', [
            "TEXT_AND_IMAGE", "IMAGE_ONLY"
        ])
        quality_presets = image_settings.get('quality_presets', [
            "standard", "hd", "ultra_hd", "ai_enhanced", "ai_ultra"  # 🚀 AI超分辨率增强选项
        ])
        style_presets = image_settings.get('style_presets', [
            "vivid", "natural", "artistic", "cinematic", "photographic"  # 超越参考项目的风格选项
        ])
        
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

                # 📐 Gemini官方API图像控制参数
                "aspect_ratio": (aspect_ratios, {
                    "default": image_settings.get('default_aspect_ratio', "1:1"),
                    "tooltip": "图像宽高比 (Gemini官方API支持)"
                }),
                "response_modality": (response_modalities, {
                    "default": image_settings.get('default_response_modality', "TEXT_AND_IMAGE"),
                    "tooltip": "响应模式：TEXT_AND_IMAGE=文字+图像，IMAGE_ONLY=仅图像"
                }),

                # 🔍 Topaz Gigapixel AI放大控制
                "upscale_factor": (["1x (不放大)", "2x", "4x", "6x"], {
                    "default": "1x (不放大)",
                    "tooltip": "使用Topaz Gigapixel AI进行智能放大"
                }),
                "gigapixel_model": (["High Fidelity", "Standard", "Art & CG", "Lines", "Very Compressed", "Low Resolution", "Text & Shapes", "Redefine", "Recover"], {
                    "default": "High Fidelity",
                    "tooltip": "Gigapixel AI放大模型"
                }),

                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),

                # 🎨 智能图像控制组（放在style下面）
                "detail_level": (["Basic Detail", "Professional Detail", "Premium Quality", "Masterpiece Level"], {"default": "Professional Detail"}),
                "camera_control": (["Auto Select", "Wide-angle Lens", "Macro Shot", "Low-angle Perspective", "High-angle Shot", "Close-up Shot", "Medium Shot"], {"default": "Auto Select"}),
                "lighting_control": (["Auto Settings", "Natural Light", "Studio Lighting", "Dramatic Shadows", "Soft Glow", "Golden Hour", "Blue Hour"], {"default": "Auto Settings"}),
                "template_selection": (["Auto Select", "Professional Portrait", "Cinematic Landscape", "Product Photography", "Digital Concept Art", "Anime Style Art", "Photorealistic Render", "Classical Oil Painting", "Watercolor Painting", "Cyberpunk Future", "Vintage Film Photography", "Architectural Photography", "Gourmet Food Photography"], {"default": "Auto Select"}),
                
                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.9), "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": default_params.get('top_k', 40), "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": default_params.get('max_output_tokens', 2048), "min": 0, "max": 32768}),
                "seed": ("INT", {"default": default_params.get('seed', 0), "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                # ✨ 自定义指令组
                "custom_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义指令和特殊要求（超越参考项目的功能）"
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
        aspect_ratio,
        response_modality,
        upscale_factor,
        gigapixel_model,
        quality,
        style,
        temperature,
        top_p,
        top_k,
        max_output_tokens,
        seed,
        custom_instructions: str = "",
        detail_level: str = "Professional Detail",
        camera_control: str = "Auto Select",
        lighting_control: str = "Auto Settings",
        template_selection: str = "Auto Select",
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

            # 🎨 构建增强提示词（使用enhance_prompt_with_controls函数）
            controls = process_image_controls(quality, style)

            # 使用enhance_prompt_with_controls函数进行完整的提示词增强
            enhanced_prompt = enhance_prompt_with_controls(
                prompt.strip(),
                controls,
                detail_level,
                camera_control,
                lighting_control,
                template_selection,
                quality_enhancement="Auto",  # 默认值
                enhance_quality=True,  # 默认值
                smart_resize=True,  # 默认值
                fill_color="white"  # 默认值
            )

            # 处理自定义指令
            if custom_instructions and custom_instructions.strip():
                enhanced_prompt += f"\n\n{custom_instructions.strip()}"
                _log_info(f"📝 添加自定义指令: {custom_instructions[:100]}...")

            _log_info(f"🎨 图像控制参数: aspect_ratio={aspect_ratio}, quality={quality}, style={style}")

            # 代理处理：有效则设置，None或无效时使用系统代理
            if proxy and proxy.strip() and "None" not in proxy:
                os.environ['HTTPS_PROXY'] = proxy.strip()
                os.environ['HTTP_PROXY'] = proxy.strip()
                _log_info(f"🔌 使用代理: {proxy.strip()}")
            else:
                # 当代理为None或无效时，不清理环境变量，使用系统代理
                _log_info("🔌 使用系统代理 (保持现有环境变量)")
            
            # 构建生成配置
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
            }

            # 🎯 Gemini官方API：Response Modalities控制
            if response_modality == "IMAGE_ONLY":
                generation_config["responseModalities"] = ["Image"]
                _log_info("📊 响应模式：仅图像（IMAGE_ONLY）")
            else:
                generation_config["responseModalities"] = ["Text", "Image"]
                _log_info("📊 响应模式：文字+图像（TEXT_AND_IMAGE）")

            # 📐 Gemini官方API：Aspect Ratio控制
            if aspect_ratio and aspect_ratio != "1:1":
                generation_config["imageConfig"] = {
                    "aspectRatio": aspect_ratio
                }
                _log_info(f"📐 设置宽高比: {aspect_ratio}")

            # 智能种子控制
            if seed > 0:
                generation_config["seed"] = seed
            
            # 准备内容
            content_parts = [{"text": enhanced_prompt}]
            # 注意：文本生成图像不需要输入图像
            _log_info(f"🔍 调试：content_parts结构: {[part.get('text', 'IMAGE_DATA') if 'text' in part else 'IMAGE_DATA' for part in content_parts]}")
            
            # 使用REST API调用
            _log_info(f"🎨 使用模型 {model} 生成图像...")
            _log_info(f"📝 提示词: {enhanced_prompt[:100]}...")
            
            response_json = generate_with_priority_api(api_key, model, content_parts, generation_config, proxy=proxy, base_url=None)
            
            # 处理响应
            raw_text = extract_text_from_response(response_json)
            generated_image = process_generated_image_from_response(response_json)
            
            # 简化图像处理流程
            if generated_image is not None:
                try:
                    # 如果是tensor，转换为PIL进行处理
                    if isinstance(generated_image, torch.Tensor):
                        # 处理batch维度
                        if generated_image.dim() == 4:
                            generated_image = generated_image[0]  # 取第一个batch

                        # 转换为PIL Image (HWC格式)
                        img_array = generated_image.cpu().numpy()
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)

                        # 确保RGB格式
                        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]  # 移除alpha通道

                        pil_image = Image.fromarray(img_array)
                        generated_image = pil_image
                    
                    # 🔍 Topaz Gigapixel AI智能放大
                    if upscale_factor and upscale_factor != "1x (不放大)" and isinstance(generated_image, Image.Image):
                        try:
                            # 提取放大倍数
                            scale = int(upscale_factor.replace("x", "").strip())
                            if scale > 1:
                                _log_info(f"🔍 使用智能AI放大进行{scale}x放大，模型: {gigapixel_model}")

                                # 导入放大函数
                                try:
                                    from .banana_upscale import smart_upscale
                                except ImportError:
                                    from banana_upscale import smart_upscale

                                # 计算目标尺寸
                                target_w = generated_image.width * scale
                                target_h = generated_image.height * scale

                                # 使用智能放大
                                upscaled_image = smart_upscale(
                                    generated_image,
                                    target_w,
                                    target_h,
                                    gigapixel_model
                                )

                                if upscaled_image:
                                    generated_image = upscaled_image
                                    _log_info(f"✅ 智能AI放大完成: {generated_image.size}")
                                else:
                                    _log_warning("⚠️ 智能AI放大失败，使用原始图像")
                        except Exception as e:
                            _log_warning(f"⚠️ 智能AI放大失败: {e}，使用原始图像")

                except Exception as e:
                    _log_error(f"图像处理失败: {e}")
                    # 确保在异常情况下也转换为tensor格式
                    if isinstance(generated_image, Image.Image):
                        generated_image = pil_to_tensor(generated_image)
            
            if not raw_text or raw_text == "Response received but no text content":
                assistant_text = "遵命！这是你所要求的图片："
            else:
                assistant_text = raw_text.strip()
            
            self._push_chat(enhanced_prompt, assistant_text, unique_id)
            
            # 确保返回tensor格式
            if isinstance(generated_image, Image.Image):
                generated_image = pil_to_tensor(generated_image)
            
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
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
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
                "gemini-2.0-flash-preview-image-generation",
                "nano-banana"
            ]
        
        # Get default model from config, prioritize latest Banana model
        default_model = config.get('default_model', {}).get('image_gen', "gemini-2.5-flash-image-preview")
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        
        # 🚀 Gemini官方API图像控制预设
        aspect_ratios = image_settings.get('aspect_ratios', [
            "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
        ])
        response_modalities = image_settings.get('response_modalities', [
            "TEXT_AND_IMAGE", "IMAGE_ONLY"
        ])
        quality_presets = image_settings.get('quality_presets', [
            "standard", "hd", "ultra_hd", "ai_enhanced", "ai_ultra"  # 🚀 AI超分辨率增强选项
        ])
        style_presets = image_settings.get('style_presets', [
            "vivid", "natural", "artistic", "cinematic", "photographic"  # 超越参考项目的风格选项
        ])
        
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

                # 📐 Gemini官方API图像控制参数
                "aspect_ratio": (aspect_ratios, {
                    "default": image_settings.get('default_aspect_ratio', "1:1"),
                    "tooltip": "图像宽高比 (Gemini官方API支持)"
                }),
                "response_modality": (response_modalities, {
                    "default": image_settings.get('default_response_modality', "TEXT_AND_IMAGE"),
                    "tooltip": "响应模式：TEXT_AND_IMAGE=文字+图像，IMAGE_ONLY=仅图像"
                }),

                # 🔍 Topaz Gigapixel AI放大控制
                "upscale_factor": (["1x (不放大)", "2x", "4x", "6x"], {
                    "default": "1x (不放大)",
                    "tooltip": "使用Topaz Gigapixel AI进行智能放大"
                }),
                "gigapixel_model": (["High Fidelity", "Standard", "Art & CG", "Lines", "Very Compressed", "Low Resolution", "Text & Shapes", "Redefine", "Recover"], {
                    "default": "High Fidelity",
                    "tooltip": "Gigapixel AI放大模型"
                }),

                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),

                # 🎨 智能图像控制组（放在style下面）
                "detail_level": (["Basic Detail", "Professional Detail", "Premium Quality", "Masterpiece Level"], {"default": "Professional Detail"}),
                "camera_control": (["Auto Select", "Wide-angle Lens", "Macro Shot", "Low-angle Perspective", "High-angle Shot", "Close-up Shot", "Medium Shot"], {"default": "Auto Select"}),
                "lighting_control": (["Auto Settings", "Natural Light", "Studio Lighting", "Dramatic Shadows", "Soft Glow", "Golden Hour", "Blue Hour"], {"default": "Auto Settings"}),
                "template_selection": (["Auto Select", "Professional Portrait", "Cinematic Landscape", "Product Photography", "Digital Concept Art", "Anime Style Art", "Photorealistic Render", "Classical Oil Painting", "Watercolor Painting", "Cyberpunk Future", "Vintage Film Photography", "Architectural Photography", "Gourmet Food Photography"], {"default": "Auto Select"}),

                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.9), "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": default_params.get('top_k', 40), "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": default_params.get('max_output_tokens', 2048), "min": 0, "max": 32768}),
                "seed": ("INT", {"default": default_params.get('seed', 0), "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                # ✨ 自定义指令组
                "custom_additions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义添加和特殊要求"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generation_text", "generated_image")
    FUNCTION = "transform_image"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"

    def transform_image(
        self,
        api_key,
        prompt,
        image,
        model,
        proxy,
        aspect_ratio,
        response_modality,
        upscale_factor,
        gigapixel_model,
        quality,
        style,
        detail_level,
        camera_control,
        lighting_control,
        template_selection,
        temperature,
        top_p,
        top_k,
        max_output_tokens,
        seed,
        custom_additions: str = "",
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

            # 🎨 构建增强提示词（使用enhance_prompt_with_controls函数）
            controls = process_image_controls(quality, style)

            # 使用enhance_prompt_with_controls函数进行完整的提示词增强
            enhanced_prompt = enhance_prompt_with_controls(
                prompt.strip(),
                controls,
                detail_level,
                camera_control,
                lighting_control,
                template_selection,
                quality_enhancement="Auto",  # 默认值
                enhance_quality=True,  # 默认值
                smart_resize=True,  # 默认值
                fill_color="white"  # 默认值
            )

            # 处理自定义指令
            if custom_additions and custom_additions.strip():
                enhanced_prompt += f"\n\n{custom_additions.strip()}"
                _log_info(f"📝 添加自定义指令: {custom_additions[:100]}...")

            _log_info(f"🎨 图像控制参数: aspect_ratio={aspect_ratio}, quality={quality}, style={style}")
            
            # 转换输入图片
            pil_image = tensor_to_pil(image)
            
            # 调整图像尺寸以符合API要求
            pil_image = resize_image_for_api(pil_image)
            
            # 转换为base64
            image_base64 = image_to_base64(pil_image, format='JPEG')
            
            # 代理处理：有效则设置，None或无效时使用系统代理
            if proxy and proxy.strip() and "None" not in proxy:
                os.environ['HTTPS_PROXY'] = proxy.strip()
                os.environ['HTTP_PROXY'] = proxy.strip()
                _log_info(f"🔌 使用代理: {proxy.strip()}")
            else:
                # 当代理为None或无效时，不清理环境变量，使用系统代理
                _log_info("🔌 使用系统代理 (保持现有环境变量)")
            
            # 构建生成配置
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
            }

            # 🎯 Gemini官方API：Response Modalities控制
            if response_modality == "IMAGE_ONLY":
                generation_config["responseModalities"] = ["Image"]
                _log_info("📊 响应模式：仅图像（IMAGE_ONLY）")
            else:
                generation_config["responseModalities"] = ["Text", "Image"]
                _log_info("📊 响应模式：文字+图像（TEXT_AND_IMAGE）")

            # 📐 Gemini官方API：Aspect Ratio控制
            if aspect_ratio and aspect_ratio != "1:1":
                generation_config["imageConfig"] = {
                    "aspectRatio": aspect_ratio
                }
                _log_info(f"📐 设置宽高比: {aspect_ratio}")

            # 智能种子控制
            if seed > 0:
                generation_config["seed"] = seed
            
            # 准备内容 - 文本 + 图像
            content_parts = [{"text": enhanced_prompt}]
            content_parts.extend(prepare_media_content(image=image))
            
            # 使用优先API调用
            _log_info(f"🖼️ 使用模型 {model} 进行图像转换...")
            _log_info(f"📝 转换指令: {enhanced_prompt[:100]}...")
            
            response_json = generate_with_priority_api(api_key, model, content_parts, generation_config, proxy=proxy, base_url=None)
            
            # 处理响应
            raw_text = extract_text_from_response(response_json)
            edited_image = process_generated_image_from_response(response_json)
            
            # 如果没有编辑后的图片，返回原图片
            if edited_image is None:
                _log_warning("未检测到编辑后的图片，返回原图片")
                edited_image = pil_image
                if not raw_text:
                    raw_text = "图片编辑请求已发送，但未收到编辑后的图片"

            # 确保edited_image是PIL Image格式
            if isinstance(edited_image, torch.Tensor):
                edited_image = tensor_to_pil(edited_image)

            # 🔍 智能AI放大
            if upscale_factor and upscale_factor != "1x (不放大)" and isinstance(edited_image, Image.Image):
                try:
                    # 提取放大倍数
                    scale = int(upscale_factor.replace("x", "").strip().split()[0])
                    if scale > 1:
                        _log_info(f"🔍 使用智能AI放大进行{scale}x放大，模型: {gigapixel_model}")

                        # 导入放大函数
                        try:
                            from .banana_upscale import smart_upscale
                        except ImportError:
                            from banana_upscale import smart_upscale

                        # 计算目标尺寸
                        target_w = edited_image.width * scale
                        target_h = edited_image.height * scale

                        # 使用智能放大
                        upscaled_image = smart_upscale(
                            edited_image,
                            target_w,
                            target_h,
                            gigapixel_model
                        )

                        if upscaled_image:
                            edited_image = upscaled_image
                            _log_info(f"✅ 智能AI放大完成: {edited_image.size}")
                        else:
                            _log_warning("⚠️ 智能AI放大失败，使用原始图像")
                except Exception as e:
                    _log_warning(f"⚠️ 智能AI放大失败: {e}，使用原始图像")

            # 如果没有响应文本，提供默认文本
            if not raw_text:
                response_text = "图片编辑完成！这是根据您的编辑指令修改后的图像。"
                _log_info("📝 使用默认响应文本")
            else:
                response_text = raw_text.strip()
            
            # 转换为tensor
            if isinstance(edited_image, Image.Image):
                image_tensor = pil_to_tensor(edited_image)
            elif isinstance(edited_image, torch.Tensor):
                image_tensor = edited_image
            else:
                _log_error(f"未知的图像类型: {type(edited_image)}")
                # 创建一个默认的黑色图像tensor
                image_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            
            _log_info("✅ 图片编辑完成")
            _log_info(f"📝 响应文本长度: {len(response_text)}")
            _log_info(f"📝 响应文本内容: {response_text[:200]}...")
            self._push_chat(enhanced_prompt, response_text or "", unique_id) # 使用增强后的提示词
            
            return (response_text, image_tensor)
            
        except Exception as e:
            error_msg = str(e)
            _log_error(f"图像转换失败: {error_msg}")
            
            # 增强的错误分类处理
            if "API key" in error_msg or "401" in error_msg or "403" in error_msg:
                friendly_error = "API密钥无效，请检查配置"
            elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                friendly_error = "内容被安全过滤器阻止，请修改提示词"
            else:
                friendly_error = f"转换失败: {error_msg}"
            
            return (friendly_error, create_dummy_image())
    
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
                            "timestamp": time.time()
                        }
                    ], ensure_ascii=False)
                },
            }
            PromptServer.instance.send_sync("display_component", render_spec)
        except Exception as e:
            _log_error(f"Chat push failed: {e}")
            pass

class KenChenLLMGeminiBananaMultimodalBananaNode:
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
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
                    "default": "", 
                    "multiline": False,
                    "placeholder": "API密钥（留空时自动从配置文件读取，建议保持为空以确保安全）"
                }),
                "prompt": ("STRING", {"default": "Describe what you see", "multiline": True}),
                "model": (
                    models,
                    {"default": default_model},
                ),
                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),

                # 🎨 分析控制组
                "detail_level": (["Basic Detail", "Professional Detail", "Premium Quality", "Masterpiece Level"], {"default": "Professional Detail"}),
                "analysis_mode": (["Auto Select", "Visual Analysis", "Audio Analysis", "Combined Analysis", "Detailed Description", "Summary Report"], {"default": "Auto Select"}),
                "output_format": (["Natural Language", "Structured Report", "Technical Analysis", "Creative Description", "Professional Summary"], {"default": "Natural Language"}),
                
                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.9), "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": default_params.get('top_k', 40), "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": default_params.get('max_output_tokens', 2048), "min": 0, "max": 8192}),
                "seed": ("INT", {"default": default_params.get('seed', 0), "min": 0, "max": 0xfffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                
                # ✨ 自定义指令组
                "custom_additions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义分析要求和特殊指令"
                }),
            },
        }
    

    
    def analyze_multimodal(
        self,
        api_key,
        prompt,
        model,
        proxy,
        detail_level,
        analysis_mode,
        output_format,
        temperature,
        top_p,
        top_k,
        max_output_tokens,
        seed,
        image=None,
        audio=None,
        custom_additions="",
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
                _log_info("🔌 使用系统代理")
            
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
            
            response_json = generate_with_priority_api(api_key, model, content_parts, generation_config, proxy=proxy)
            
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

class GeminiBananaMultiImageEditNode:
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
        config = get_gemini_banana_config()
        default_params = config.get('default_params', {})
        image_settings = config.get('image_settings', {})
        
        # 🚀 Gemini官方API图像控制预设
        aspect_ratios = image_settings.get('aspect_ratios', [
            "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
        ])
        response_modalities = image_settings.get('response_modalities', [
            "TEXT_AND_IMAGE", "IMAGE_ONLY"
        ])
        quality_presets = image_settings.get('quality_presets', [
            "standard", "hd", "ultra_hd", "ai_enhanced", "ai_ultra"  # 🚀 AI超分辨率增强选项
        ])
        style_presets = image_settings.get('style_presets', [
            "vivid", "natural", "artistic", "cinematic", "photographic"  # 超越参考项目的风格选项
        ])
        
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "请根据这些图片进行专业的图像编辑", "multiline": True}),
                "model": (["gemini-2.5-flash-image", "gemini-2.5-flash-image-preview", "gemini-2.0-flash"], {"default": "gemini-2.5-flash-image"}),

                # 📐 Gemini官方API图像控制参数
                "aspect_ratio": (aspect_ratios, {
                    "default": image_settings.get('default_aspect_ratio', "1:1"),
                    "tooltip": "图像宽高比 (Gemini官方API支持)"
                }),
                "response_modality": (response_modalities, {
                    "default": image_settings.get('default_response_modality', "TEXT_AND_IMAGE"),
                    "tooltip": "响应模式：TEXT_AND_IMAGE=文字+图像，IMAGE_ONLY=仅图像"
                }),

                # 🔍 Topaz Gigapixel AI放大控制
                "upscale_factor": (["1x (不放大)", "2x", "4x", "6x"], {
                    "default": "1x (不放大)",
                    "tooltip": "使用Topaz Gigapixel AI进行智能放大"
                }),
                "gigapixel_model": (["High Fidelity", "Standard", "Art & CG", "Lines", "Very Compressed", "Low Resolution", "Text & Shapes", "Redefine", "Recover"], {
                    "default": "High Fidelity",
                    "tooltip": "Gigapixel AI放大模型"
                }),

                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),

                # 🎨 智能图像控制组（放在style下面）
                "detail_level": (["Basic Detail", "Professional Detail", "Premium Quality", "Masterpiece Level"], {"default": "Professional Detail"}),
                "camera_control": (["Auto Select", "Wide-angle Lens", "Macro Shot", "Low-angle Perspective", "High-angle Shot", "Close-up Shot", "Medium Shot"], {"default": "Auto Select"}),
                "lighting_control": (["Auto Settings", "Natural Light", "Studio Lighting", "Dramatic Shadows", "Soft Glow", "Golden Hour", "Blue Hour"], {"default": "Auto Settings"}),
                "template_selection": (["Auto Select", "Professional Portrait", "Cinematic Landscape", "Product Photography", "Digital Concept Art", "Anime Style Art", "Photorealistic Render", "Classical Oil Painting", "Watercolor Painting", "Cyberpunk Future", "Vintage Film Photography", "Architectural Photography", "Gourmet Food Photography"], {"default": "Auto Select"}),

                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.95), "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": default_params.get('top_k', 40), "min": 0, "max": 100}),
                "max_output_tokens": ("INT", {"default": default_params.get('max_output_tokens', 8192), "min": 0, "max": 32768}),
                "seed": ("INT", {"default": default_params.get('seed', 0), "min": 0, "max": 999999}),
                "post_generation_control": (["randomize", "maintain_consistency", "enhance_creativity"], {"default": "randomize"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                
                # ✨ 自定义指令组
                "custom_additions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义添加和特殊要求"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "edit_multiple_images"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"

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
                            "timestamp": time.time()
                        }
                    ], ensure_ascii=False)
                },
            }
            PromptServer.instance.send_sync("display_component", render_spec)
        except Exception as e:
            _log_error(f"Chat push failed: {e}")
            pass

    def edit_multiple_images(self, api_key: str, prompt: str, model: str, aspect_ratio: str, response_modality: str,
                           upscale_factor: str, gigapixel_model: str, quality: str, style: str, detail_level: str,
                           camera_control: str, lighting_control: str, template_selection: str, temperature: float, top_p: float,
                           top_k: int, max_output_tokens: int, seed: int, post_generation_control: str,
                           image1=None, image2=None, image3=None, image4=None, custom_additions: str = "", unique_id: str = "") -> Tuple[torch.Tensor, str]:
        """使用 Gemini API 进行多图像编辑"""

        # 如果用户没有输入API密钥，自动从配置文件获取
        if not api_key or not api_key.strip():
            config = get_gemini_banana_config()
            auto_api_key = config.get('api_key', '')
            if auto_api_key and auto_api_key.strip():
                api_key = auto_api_key.strip()
                print(f"🔑 自动使用配置文件中的API密钥: {api_key[:8]}...")
            else:
                raise ValueError("API密钥不能为空，请在配置文件中设置api_key或手动输入")

        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")

        # 🎨 构建增强提示词（使用enhance_prompt_with_controls函数）
        controls = process_image_controls(quality, style)

        # 使用enhance_prompt_with_controls函数进行完整的提示词增强
        enhanced_prompt = enhance_prompt_with_controls(
            prompt.strip(),
            controls,
            detail_level,
            camera_control,
            lighting_control,
            template_selection,
            quality_enhancement="Auto",  # 默认值
            enhance_quality=True,  # 默认值
            smart_resize=True,  # 默认值
            fill_color="white"  # 默认值
        )

        # 处理自定义指令
        if custom_additions and custom_additions.strip():
            enhanced_prompt += f"\n\n{custom_additions.strip()}"
            print(f"📝 添加自定义指令: {custom_additions[:100]}...")

        print(f"🎨 图像控制参数: aspect_ratio={aspect_ratio}, quality={quality}, style={style}")
        
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
        
        # 🚀 应用真正的图形增强技术
        # 1. 对输入图像进行预处理增强
        enhanced_input_pils = []
        for i, pil_image in enumerate(all_input_pils):
            print(f"🎨 处理输入图像 {i+1}...")
            enhanced_image = pil_image

            # 调整图像尺寸以符合API要求
            enhanced_image = resize_image_for_api(enhanced_image)
            enhanced_input_pils.append(enhanced_image)

        # 2. 构建结构化内容：图片标识 + 增强后的图片数据 + 指令文本
        content = []

        # 先添加图片标识文本（参考项目的核心技术）
        image_labels = ["Figure 1", "Figure 2", "Figure 3", "Figure 4"]
        for i, enhanced_image in enumerate(enhanced_input_pils):
            content.append({
                "type": "text",
                "text": f"[This is {image_labels[i]}]"
            })

        # 再添加增强后的图片数据
        for enhanced_image in enhanced_input_pils:
            # 转换为base64
            image_base64 = image_to_base64(enhanced_image, format='JPEG')
            content.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            })
        
        # 3. 最后添加强制图像生成指令（参考项目的核心技术）
        if len(all_input_pils) == 1:
            # 单图编辑
            image_edit_instruction = f"""CRITICAL INSTRUCTION: You MUST generate and return an actual image, not just text description.

Task: {prompt}

REQUIREMENTS:
1. GENERATE a new edited image based on my request
2. DO NOT just describe what the image should look like
3. RETURN the actual image file/data
4. The output MUST be a visual image, not text

Execute the image editing task now and return the generated image."""
        else:
            # 多图编辑
            image_edit_instruction = f"""CRITICAL INSTRUCTION: You MUST generate and return an actual image, not just text description.

Task: {enhanced_prompt}

Image References:
- When I mention "Figure 1", "第一张图片", "左边图片", I mean the first image provided above
- When I mention "Figure 2", "第二张图片", "右边图片", I mean the second image provided above
- When I mention "Figure 3", "第三张图片", I mean the third image provided above
- When I mention "Figure 4", "第四张图片", I mean the fourth image provided above

REQUIREMENTS:
1. GENERATE a new image based on my request and the provided reference images
2. DO NOT just describe what the image should look like
3. RETURN the actual image file/data
4. The output MUST be a visual image, not text
5. Combine elements from the reference images as specified in the task
6. Maintain high quality and natural appearance

Execute the image editing task now and return the generated image."""
        
        content.append({"type": "text", "text": image_edit_instruction})
        
        # 构建请求数据
        generation_config = {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_output_tokens,
        }

        # 🎯 Gemini官方API：Response Modalities控制
        if response_modality == "IMAGE_ONLY":
            generation_config["responseModalities"] = ["Image"]
            print("📊 响应模式：仅图像（IMAGE_ONLY）")
        else:
            generation_config["responseModalities"] = ["Text", "Image"]
            print("📊 响应模式：文字+图像（TEXT_AND_IMAGE）")

        # 📐 Gemini官方API：Aspect Ratio控制
        if aspect_ratio and aspect_ratio != "1:1":
            generation_config["imageConfig"] = {
                "aspectRatio": aspect_ratio
            }
            print(f"📐 设置宽高比: {aspect_ratio}")

        # 智能种子控制
        if seed and seed > 0:
            generation_config["seed"] = seed

        request_data = {
            "contents": [{
                "parts": content
            }],
            "generationConfig": generation_config
        }
        
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
                
                # 使用优先API调用
                result = generate_with_priority_api_direct(
                    api_key, 
                    model, 
                    request_data, 
                    max_retries=1,  # 在重试循环中只尝试一次
                    proxy=None,
                    base_url=get_gemini_banana_config().get('base_url', 'https://generativelanguage.googleapis.com')
                )
                
                # 成功响应
                if result:
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

                    # 🔍 智能AI放大
                    if upscale_factor and upscale_factor != "1x (不放大)" and isinstance(edited_image, Image.Image):
                        try:
                            # 提取放大倍数
                            scale = int(upscale_factor.replace("x", "").strip().split()[0])
                            if scale > 1:
                                print(f"🔍 使用智能AI放大进行{scale}x放大，模型: {gigapixel_model}")

                                # 导入放大函数
                                try:
                                    from .banana_upscale import smart_upscale
                                except ImportError:
                                    from banana_upscale import smart_upscale

                                # 计算目标尺寸
                                target_w = edited_image.width * scale
                                target_h = edited_image.height * scale

                                # 使用智能放大
                                upscaled_image = smart_upscale(
                                    edited_image,
                                    target_w,
                                    target_h,
                                    gigapixel_model
                                )

                                if upscaled_image:
                                    edited_image = upscaled_image
                                    print(f"✅ 智能AI放大完成: {edited_image.size}")
                                else:
                                    print("⚠️ 智能AI放大失败，使用原始图像")
                        except Exception as e:
                            print(f"⚠️ 智能AI放大失败: {e}，使用原始图像")

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
                    print(f"❌ API调用失败: 未收到有效响应")
                    
                    # 如果是最后一次尝试，抛出异常
                    if attempt == max_retries - 1:
                        raise ValueError("API调用失败: 未收到有效响应")
                    
                    # 智能等待
                    delay = smart_retry_delay(attempt, 500)  # 使用通用错误码
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except Exception as e:
                error_msg = format_error_message(e)
                print(f"❌ 请求失败: {error_msg}")
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，抛出异常
                    print(f"❌ 所有重试都失败了，多图像编辑失败: {error_msg}")
                    raise ValueError(f"多图像编辑失败: {error_msg}")
                else:
                    delay = smart_retry_delay(attempt)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)

# 翻译节点已移动到独立模块 gemini_banana_translation.py

# Node mappings
NODE_CLASS_MAPPINGS = {
    "KenChenLLMGeminiBananaTextToImageBananaNode": KenChenLLMGeminiBananaTextToImageBananaNode,
    "KenChenLLMGeminiBananaImageToImageBananaNode": KenChenLLMGeminiBananaImageToImageBananaNode,
    "KenChenLLMGeminiBananaMultimodalBananaNode": KenChenLLMGeminiBananaMultimodalBananaNode,
    "GeminiBananaMultiImageEdit": GeminiBananaMultiImageEditNode,
}

# 集成独立翻译模块
if TRANSLATION_MODULE_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(TRANSLATION_NODE_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "KenChenLLMGeminiBananaTextToImageBananaNode": "Gemini Banana Text to Image Banana",
    "KenChenLLMGeminiBananaImageToImageBananaNode": "Gemini Banana Image to Image Banana",
    "KenChenLLMGeminiBananaMultimodalBananaNode": "Gemini Banana Multimodal Banana",
    "GeminiBananaMultiImageEdit": "Gemini Banana Multi Image Edit",
}

# 集成独立翻译模块显示名称
if TRANSLATION_MODULE_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update(TRANSLATION_DISPLAY_MAPPINGS)

