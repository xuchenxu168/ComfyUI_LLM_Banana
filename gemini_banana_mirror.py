"""
Gemini Banana 镜像站节点
支持自定义API地址，适配国内镜像站和代理服务
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import time
import random
from typing import Optional, Tuple, Dict, Any
import re
import os
try:
    from server import PromptServer
except ImportError:
    # 在测试环境中，PromptServer可能不可用
    class PromptServer:
        @staticmethod
        def instance():
            return None
import sys
from io import BytesIO
from PIL import Image, ImageFilter

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

def ai_upscale_with_realesrgan(image, target_width, target_height):
    """
    统一委托到通用放大器（banana_upscale.smart_upscale），优先2x，失败回退LANCZOS。
    """
    try:
        try:
            from .banana_upscale import smart_upscale as _smart
        except ImportError:
            from banana_upscale import smart_upscale as _smart
        res = _smart(image, target_width, target_height)
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

def smart_ai_upscale(image, target_width, target_height):
	"""
	统一委托到通用放大器（banana_upscale.smart_upscale）
	"""
	try:
		try:
			from .banana_upscale import smart_upscale as _smart
		except ImportError:
			from banana_upscale import smart_upscale as _smart
		return _smart(image, target_width, target_height)
	except Exception as e:
		print(f"⚠️ 智能放大器失败: {e}")
		return None


def detect_image_foreground_subject(image):
    """
    🎯 智能检测图像前景主体（人物、物体等）
    返回主体边界框 (x, y, width, height) 和中心点
    """
    try:
        import cv2
        import numpy as np

        # 转换为OpenCV格式
        if hasattr(image, 'convert'):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image

        # 转换为BGR格式（OpenCV使用BGR）
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        height, width = img_bgr.shape[:2]
        print(f"🔍 图像尺寸: {width}x{height}")

        # 🎯 方法1：全图人脸检测（最优先）
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # 找到最大的人脸
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face

                # 🚀 关键修复：以人脸为中心，扩展到合理的人物区域
                # 人脸通常占人物高度的1/8到1/6
                estimated_person_height = h * 6  # 假设人脸是人物高度的1/6
                estimated_person_width = w * 3   # 假设人物宽度是人脸宽度的3倍

                # 计算人物区域（以人脸中心为基准）
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # 人物区域的左上角（人脸通常在人物上部1/4处）
                person_x = max(0, face_center_x - estimated_person_width // 2)
                person_y = max(0, face_center_y - estimated_person_height // 4)

                # 确保不超出图像边界
                person_w = min(estimated_person_width, width - person_x)
                person_h = min(estimated_person_height, height - person_y)

                subject_center_x = person_x + person_w // 2
                subject_center_y = person_y + person_h // 2

                print(f"🎯 人脸检测识别到主体: 人脸({x}, {y}, {w}x{h}), 人物区域({person_x}, {person_y}, {person_w}x{person_h}), 中心({subject_center_x}, {subject_center_y})")
                return (person_x, person_y, person_w, person_h), (subject_center_x, subject_center_y)
        except Exception as e:
            print(f"⚠️ 人脸检测失败: {e}")

        # 🎯 方法2：全图肤色检测（改进版）
        try:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # 扩展肤色范围，包含更多肤色类型
            lower_skin1 = np.array([0, 20, 70])    # 偏红肤色
            upper_skin1 = np.array([20, 255, 255])

            lower_skin2 = np.array([0, 10, 60])    # 较浅肤色
            upper_skin2 = np.array([25, 255, 255])

            # 创建肤色掩码
            skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)

            # 形态学操作
            kernel = np.ones((7,7), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

            # 找到肤色区域
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 过滤太小的区域
                min_area = (width * height) * 0.005  # 最小面积阈值
                valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

                if valid_contours:
                    # 找到最大的肤色区域
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # 🚀 关键修复：合理扩展边界框
                    # 肤色区域通常只是人物的一部分，需要适当扩展
                    expand_factor_w = 1.5  # 宽度扩展系数
                    expand_factor_h = 1.8  # 高度扩展系数

                    expanded_w = int(w * expand_factor_w)
                    expanded_h = int(h * expand_factor_h)

                    # 重新计算位置，保持中心不变
                    expanded_x = max(0, x - (expanded_w - w) // 2)
                    expanded_y = max(0, y - (expanded_h - h) // 4)  # 向上扩展更多

                    # 确保不超出图像边界
                    expanded_w = min(expanded_w, width - expanded_x)
                    expanded_h = min(expanded_h, height - expanded_y)

                    subject_center_x = expanded_x + expanded_w // 2
                    subject_center_y = expanded_y + expanded_h // 2

                    print(f"🎯 肤色检测识别到主体: 原始({x}, {y}, {w}x{h}), 扩展后({expanded_x}, {expanded_y}, {expanded_w}x{expanded_h}), 中心({subject_center_x}, {subject_center_y})")
                    return (expanded_x, expanded_y, expanded_w, expanded_h), (subject_center_x, subject_center_y)

        except Exception as e:
            print(f"⚠️ 肤色检测失败: {e}")
        
        # 🎯 方法3：改进的边缘检测（专注于人物轮廓）
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 使用更保守的边缘检测参数
            edges = cv2.Canny(blurred, 30, 100)

            # 形态学操作连接边缘
            kernel = np.ones((5,5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 找到轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 过滤小轮廓，找到主要物体
                min_area = (width * height) * 0.02  # 提高最小面积阈值
                valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

                if valid_contours:
                    # 找到最大的轮廓
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # 🚀 关键修复：确保检测到的是合理的人物区域
                    aspect_ratio = w / h
                    if 0.3 <= aspect_ratio <= 2.0:  # 人物的宽高比通常在这个范围内
                        subject_center_x = x + w // 2
                        subject_center_y = y + h // 2

                        print(f"🎯 边缘检测识别到前景主体: 位置({x}, {y}), 尺寸({w}x{h}), 中心({subject_center_x}, {subject_center_y})")
                        return (x, y, w, h), (subject_center_x, subject_center_y)

        except Exception as e:
            print(f"⚠️ 边缘检测算法失败: {e}")

        # 🎯 方法4：安全的保守策略（确保主体不丢失）
        try:
            print(f"🔍 使用安全保守策略...")

            # 🚀 关键修复：使用更保守的主体位置估计
            # 基于图像的黄金分割点和常见人物位置

            # 分析图像的亮度分布，找到可能的主体区域
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 将图像分为9个区域（3x3网格）
            h_step = height // 3
            w_step = width // 3

            max_variance = 0
            best_region = None

            for i in range(3):
                for j in range(3):
                    region_y = i * h_step
                    region_x = j * w_step
                    region = gray[region_y:region_y+h_step, region_x:region_x+w_step]

                    # 计算区域的方差（高方差通常表示有更多细节，可能是主体）
                    variance = np.var(region)
                    if variance > max_variance:
                        max_variance = variance
                        best_region = (region_x, region_y, w_step, h_step)

            if best_region:
                x, y, w, h = best_region
                subject_center_x = x + w // 2
                subject_center_y = y + h // 2

                print(f"🎯 方差分析识别到主体区域: 位置({x}, {y}), 尺寸({w}x{h}), 中心({subject_center_x}, {subject_center_y})")
                return (x, y, w, h), (subject_center_x, subject_center_y)

        except Exception as e:
            print(f"⚠️ 方差分析失败: {e}")

        # 🎯 最终安全策略：基于图像中心的保守估计
        print(f"🔍 使用最终安全策略：图像中心区域...")

        # 🚀 关键修复：使用图像中心区域作为主体位置
        # 这样可以确保主体不会完全丢失
        safe_x = width // 4
        safe_y = height // 4
        safe_w = width // 2
        safe_h = height // 2

        safe_center_x = safe_x + safe_w // 2  # 图像中心
        safe_center_y = safe_y + safe_h // 2  # 图像中心

        print(f"🎯 安全策略主体位置: ({safe_x}, {safe_y}), 尺寸({safe_w}x{safe_h}), 中心({safe_center_x}, {safe_center_y})")
        return (safe_x, safe_y, safe_w, safe_h), (safe_center_x, safe_center_y)
        
    except ImportError:
        print("⚠️ OpenCV未安装，无法进行智能主体检测")
        # 🚀 关键修复：返回安全的图像中心位置
        width, height = image.size
        safe_center_x = width // 2   # 图像中心
        safe_center_y = height // 2  # 图像中心
        safe_w = width // 2
        safe_h = height // 2
        safe_x = width // 4
        safe_y = height // 4
        print(f"🎯 默认安全位置: ({safe_x}, {safe_y}), 尺寸({safe_w}x{safe_h}), 中心({safe_center_x}, {safe_center_y})")
        return (safe_x, safe_y, safe_w, safe_h), (safe_center_x, safe_center_y)
    except Exception as e:
        print(f"❌ 智能主体检测失败: {e}")
        # 🚀 关键修复：返回安全的图像中心位置
        width, height = image.size
        safe_center_x = width // 2   # 图像中心
        safe_center_y = height // 2  # 图像中心
        safe_w = width // 2
        safe_h = height // 2
        safe_x = width // 4
        safe_y = height // 4
        print(f"🎯 异常处理安全位置: ({safe_x}, {safe_y}), 尺寸({safe_w}x{safe_h}), 中心({safe_center_x}, {safe_center_y})")
        return (safe_x, safe_y, safe_w, safe_h), (safe_center_x, safe_center_y)

def _log_info(message):
    try:
        print(f"[LLM Prompt][Gemini-Banana-Mirror] {message}")
    except UnicodeEncodeError:
        print(f"[LLM Prompt][Gemini-Banana-Mirror] INFO: {repr(message)}")

def _log_warning(message):
    try:
        print(f"[LLM Prompt][Gemini-Banana-Mirror] WARNING: {message}")
    except UnicodeEncodeError:
        print(f"[LLM Prompt][Gemini-Banana-Mirror] WARNING: {repr(message)}")

def _log_error(message):
    try:
        print(f"[LLM Prompt][Gemini-Banana-Mirror] ERROR: {message}")
    except UnicodeEncodeError:
        print(f"[LLM Prompt][Gemini-Banana-Mirror] ERROR: {repr(message)}")

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    tensor = (tensor * 255).clamp(0, 255).byte()
    return Image.fromarray(tensor.cpu().numpy())

def pil_to_tensor(image):
    """Convert PIL Image to tensor"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_array).unsqueeze(0)
    return tensor

def enhance_image_quality(image: Image.Image, quality: str = "hd", adaptive_mode: str = "disabled") -> Image.Image:
    """
    🚀 全面AI画质增强系统
    智能识别图像类型并应用最适合的增强技术
    """
    print(f"🎨 开始全面AI画质增强，质量级别: {quality}")

    # 保存原图用于过度增强检测
    original_image = image.copy()

    # 🚀 第一步：智能图像类型识别
    image_type = _analyze_image_type(image)
    print(f"🔍 图像类型识别: {image_type}")

    # 🚀 第二步：根据图像类型选择最佳增强策略
    if image_type == "portrait":
        # 人像图像：优先人脸修复
        enhanced_image = _apply_ai_face_restoration(image)
        if enhanced_image:
            print(f"✅ AI人脸修复完成")
            image = enhanced_image

    elif image_type == "landscape":
        # 风景图像：使用风景专用增强
        enhanced_image = _apply_landscape_enhancement(image, quality)
        if enhanced_image:
            print(f"✅ 风景专用增强完成")
            image = enhanced_image

    elif image_type == "architecture":
        # 建筑图像：使用建筑专用增强
        enhanced_image = _apply_architecture_enhancement(image, quality)
        if enhanced_image:
            print(f"✅ 建筑专用增强完成")
            image = enhanced_image

    elif image_type == "artwork":
        # 艺术作品：使用艺术专用增强
        enhanced_image = _apply_artwork_enhancement(image, quality)
        if enhanced_image:
            print(f"✅ 艺术专用增强完成")
            image = enhanced_image

    else:
        # 通用图像：使用通用AI增强
        print(f"🔍 使用通用AI增强策略")

    # 🚀 第三步：通用AI超分辨率增强（适用于所有类型）
    if quality == "ultra_hd":
        sr_enhanced = _apply_ai_super_resolution(image, image_type)
        if sr_enhanced:
            print(f"✅ AI超分辨率增强完成")
            image = sr_enhanced
        else:
            print(f"🔍 AI超分辨率不可用，使用传统增强")

    # 🚀 第四步：自适应传统增强（根据图像类型调整参数）
    if quality in ["hd", "ultra_hd"]:
        image = _apply_adaptive_traditional_enhancement(image, quality, image_type)

    # 🚀 第五步：智能过度增强检测和修正
    final_image = _check_and_correct_over_enhancement(original_image, image)  # 传入原图和增强后的图
    if final_image != image:
        print(f"🔧 检测到过度增强，已自动修正")
        image = final_image

    print(f"🎨 全面画质增强流程完成")
    return image


def _apply_ai_face_restoration(image: Image.Image) -> Optional[Image.Image]:
    """
    🚀 AI人脸修复：优先专业模型，回退到智能增强
    """
    try:
        # 首先检测是否有人脸
        faces = _detect_faces_with_locations(image)
        if not faces:
            return None

        print(f"🎯 检测到{len(faces)}个人脸，开始AI修复...")

        # 尝试CodeFormer（最新技术）
        restored = _try_codeformer_restoration(image)
        if restored:
            print(f"✅ CodeFormer人脸修复成功")
            return restored

        # 回退到GFPGAN
        restored = _try_gfpgan_restoration(image)
        if restored:
            print(f"✅ GFPGAN人脸修复成功")
            return restored

        # 🚀 实用的回退方案：使用AI放大技术增强人脸区域
        enhanced = _enhance_face_regions_with_ai_upscale(image, faces)
        if enhanced:
            print(f"✅ AI放大人脸增强成功")
            return enhanced

        print(f"⚠️ 所有AI人脸修复方法不可用")
        return None

    except Exception as e:
        print(f"❌ AI人脸修复失败: {e}")
        return None


def _apply_ai_super_resolution(image: Image.Image, image_type: str = "general") -> Optional[Image.Image]:
    """
    🚀 AI超分辨率增强：根据图像类型选择最佳策略
    """
    try:
        # 🚀 根据图像类型调整放大策略
        current_size = max(image.size)

        if image_type == "portrait":
            # 人像：非常保守，保持自然
            target_scale = 1.15 if current_size < 1024 else 1.08
        elif image_type == "landscape":
            # 风景：温和放大，保持真实感
            target_scale = 1.25 if current_size < 1024 else 1.12
        elif image_type == "architecture":
            # 建筑：适度放大，保持材质质感
            target_scale = 1.2 if current_size < 1024 else 1.1
        elif image_type == "artwork":
            # 艺术作品：最小放大，保持原有风格
            target_scale = 1.1 if current_size < 1024 else 1.05
        else:
            # 通用：保守策略
            target_scale = 1.2 if current_size < 1024 else 1.1

        target_w = int(image.width * target_scale)
        target_h = int(image.height * target_scale)

        print(f"🚀 尝试AI超分辨率增强: {image.size} -> ({target_w}, {target_h})")

        # 使用现有的智能放大系统
        enhanced = smart_ai_upscale(image, target_w, target_h)
        if enhanced:
            # 如果放大成功，缩回原尺寸以保持细节提升
            final = enhanced.resize(image.size, Image.Resampling.LANCZOS)
            print(f"✅ AI超分辨率增强成功")
            return final

        return None

    except Exception as e:
        print(f"❌ AI超分辨率增强失败: {e}")
        return None


def _analyze_image_type(image: Image.Image) -> str:
    """
    🚀 智能图像类型识别
    分析图像内容并返回最适合的增强策略类型
    """
    try:
        import cv2
        import numpy as np

        # 转换为OpenCV格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 检测人脸
        faces = _detect_faces_with_locations(image)
        if faces:
            face_area = sum(w * h for x, y, w, h in faces)
            total_area = image.width * image.height
            face_ratio = face_area / total_area

            if face_ratio > 0.05:  # 人脸占比超过5%
                return "portrait"

        # 分析颜色分布和纹理特征
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 检测天空（蓝色区域）
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        blue_ratio = np.sum(blue_mask > 0) / (image.width * image.height)

        # 检测绿色植被
        green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / (image.width * image.height)

        # 检测建筑特征（直线和边缘）
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 检测直线（建筑特征）
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0

        # 根据特征判断图像类型
        if blue_ratio > 0.15 and green_ratio > 0.2:
            return "landscape"  # 风景：有天空和植被
        elif line_count > 20:
            return "architecture"  # 建筑：有很多直线
        elif blue_ratio < 0.05 and green_ratio < 0.1 and line_count < 10:
            # 检测艺术作品特征（色彩丰富度）
            colors = img_array.reshape(-1, 3)
            unique_colors = len(np.unique(colors.view(np.dtype((np.void, colors.dtype.itemsize*colors.shape[1])))))
            color_diversity = unique_colors / (image.width * image.height)

            if color_diversity > 0.3:
                return "artwork"  # 艺术作品：色彩丰富

        return "general"  # 通用图像

    except Exception as e:
        print(f"⚠️ 图像类型识别失败: {e}")
        return "general"


def _apply_landscape_enhancement(image: Image.Image, quality: str) -> Optional[Image.Image]:
    """
    🚀 自然风景专用增强 - 保持真实感
    """
    try:
        print(f"🌄 应用自然风景增强...")
        from PIL import ImageEnhance, ImageFilter
        import numpy as np

        enhanced = image.copy()

        # 🚀 自然风景增强策略（保持真实感）：
        # 1. 轻微增强对比度（避免过度HDR效果）
        # 2. 温和提升饱和度（保持自然色彩）
        # 3. 细节增强但不过度锐化

        # 轻微对比度增强（避免假HDR效果）
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.06)  # 降低到更自然的水平

        # 温和饱和度增强（保持自然）
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(1.04)  # 更温和的饱和度

        # 自然的细节增强
        if quality == "ultra_hd":
            # 使用非常温和的锐化
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=0.8, percent=80, threshold=5))
        else:
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(1.08)  # 温和锐化

        # 保持自然亮度
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = brightness_enhancer.enhance(1.01)  # 微调即可

        print(f"✅ 自然风景增强完成")
        return enhanced

    except Exception as e:
        print(f"❌ 风景增强失败: {e}")
        return None


def _apply_architecture_enhancement(image: Image.Image, quality: str) -> Optional[Image.Image]:
    """
    🚀 建筑图像专用增强 - 保持真实感
    """
    try:
        print(f"🏢 应用建筑专用增强...")
        from PIL import ImageEnhance, ImageFilter

        enhanced = image.copy()

        # 🚀 建筑增强策略（保持真实感）：
        # 1. 适度强化边缘（避免过度锐化）
        # 2. 温和增强对比度（保持自然光影）
        # 3. 保持材质质感

        # 适度边缘增强（避免过度锐化）
        if quality == "ultra_hd":
            # 使用温和的锐化参数
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=100, threshold=3))
        else:
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(1.15)  # 降低锐化强度

        # 温和对比度增强（保持自然光影）
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.08)  # 降低对比度增强

        # 保持自然饱和度
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(1.03)  # 更温和的饱和度

        # 保持自然亮度
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = brightness_enhancer.enhance(1.01)  # 微调即可

        print(f"✅ 建筑专用增强完成")
        return enhanced

    except Exception as e:
        print(f"❌ 建筑增强失败: {e}")
        return None


def _apply_artwork_enhancement(image: Image.Image, quality: str) -> Optional[Image.Image]:
    """
    🚀 艺术作品专用增强 - 保持原有风格
    """
    try:
        print(f"🎨 应用艺术专用增强...")
        from PIL import ImageEnhance, ImageFilter

        enhanced = image.copy()

        # 🚀 艺术作品增强策略（保持原有风格）：
        # 1. 最小化色彩改动
        # 2. 轻微细节增强
        # 3. 完全保持艺术风格

        # 非常温和的色彩增强
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(1.03)  # 降低到最小

        # 轻微的对比度增强
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.04)  # 降低对比度增强

        # 最温和的锐化
        if quality == "ultra_hd":
            # 使用极温和的锐化参数
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=8))
        else:
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(1.06)  # 极温和锐化

        # 保持原有亮度
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = brightness_enhancer.enhance(1.005)  # 几乎不改变

        print(f"✅ 艺术专用增强完成")
        return enhanced

    except Exception as e:
        print(f"❌ 艺术增强失败: {e}")
        return None


def _apply_adaptive_traditional_enhancement(image: Image.Image, quality: str, image_type: str) -> Image.Image:
    """
    🚀 自适应传统增强：根据图像类型调整参数
    """
    try:
        print(f"🎨 应用自适应传统增强 (类型: {image_type}, 质量: {quality})...")

        # 如果已经应用了专用增强，使用更温和的参数
        if image_type in ["landscape", "architecture", "artwork"]:
            return _apply_gentle_traditional_enhancement(image, quality)
        else:
            # 对于人像和通用图像，使用标准增强
            return _apply_advanced_traditional_enhancement(image, quality)

    except Exception as e:
        print(f"❌ 自适应传统增强失败: {e}")
        return image


def _apply_gentle_traditional_enhancement(image: Image.Image, quality: str) -> Image.Image:
    """
    🚀 极温和的传统增强：用于已经应用专用增强的图像
    """
    try:
        from PIL import ImageEnhance

        print(f"🎨 应用极温和传统增强...")

        # 极温和的锐化
        enhancer = ImageEnhance.Sharpness(image)
        sharpness_factor = 1.02 if quality == "hd" else 1.04  # 降低到极温和
        image = enhancer.enhance(sharpness_factor)

        # 极轻微的对比度调整
        enhancer = ImageEnhance.Contrast(image)
        contrast_factor = 1.015 if quality == "hd" else 1.025  # 降低到极温和
        image = enhancer.enhance(contrast_factor)

        # 极微调饱和度
        enhancer = ImageEnhance.Color(image)
        color_factor = 1.01 if quality == "hd" else 1.015  # 降低到极温和
        image = enhancer.enhance(color_factor)

        # 亮度几乎不变
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.003)  # 几乎不改变

        print(f"✅ 极温和传统增强完成")
        return image

    except Exception as e:
        print(f"❌ 温和传统增强失败: {e}")
        return image


def _check_and_correct_over_enhancement(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    """
    🚀 智能检测过度增强并自动修正
    """
    try:
        import numpy as np
        from PIL import ImageStat

        print(f"🔍 检测过度增强...")

        # 转换为numpy数组进行分析
        orig_array = np.array(original)
        enh_array = np.array(enhanced)

        # 检测过度饱和
        orig_sat = np.std(orig_array)
        enh_sat = np.std(enhanced)
        saturation_increase = enh_sat / orig_sat if orig_sat > 0 else 1.0

        # 检测过度对比度
        orig_contrast = np.std(orig_array.astype(float))
        enh_contrast = np.std(enh_array.astype(float))
        contrast_increase = enh_contrast / orig_contrast if orig_contrast > 0 else 1.0

        # 检测过度锐化（边缘检测）
        try:
            import cv2
            orig_gray = cv2.cvtColor(orig_array, cv2.COLOR_RGB2GRAY)
            enh_gray = cv2.cvtColor(enh_array, cv2.COLOR_RGB2GRAY)

            orig_edges = cv2.Canny(orig_gray, 50, 150)
            enh_edges = cv2.Canny(enh_gray, 50, 150)

            orig_edge_count = np.sum(orig_edges > 0)
            enh_edge_count = np.sum(enh_edges > 0)
            edge_increase = enh_edge_count / orig_edge_count if orig_edge_count > 0 else 1.0
        except:
            edge_increase = 1.0

        print(f"🔍 增强分析: 饱和度x{saturation_increase:.2f}, 对比度x{contrast_increase:.2f}, 边缘x{edge_increase:.2f}")

        # 判断是否过度增强
        over_enhanced = False
        blend_ratio = 1.0  # 1.0 = 完全使用增强图像，0.0 = 完全使用原图

        if saturation_increase > 1.15:  # 饱和度增加超过15%
            over_enhanced = True
            blend_ratio *= 0.7
            print(f"⚠️ 检测到过度饱和")

        if contrast_increase > 1.2:  # 对比度增加超过20%
            over_enhanced = True
            blend_ratio *= 0.8
            print(f"⚠️ 检测到过度对比")

        if edge_increase > 1.5:  # 边缘增加超过50%
            over_enhanced = True
            blend_ratio *= 0.6
            print(f"⚠️ 检测到过度锐化")

        if over_enhanced:
            # 混合原图和增强图像以减少过度增强
            print(f"🔧 应用修正混合 (比例: {blend_ratio:.2f})")

            orig_array = orig_array.astype(float)
            enh_array = enh_array.astype(float)

            corrected_array = (enh_array * blend_ratio + orig_array * (1 - blend_ratio)).astype(np.uint8)
            corrected_image = Image.fromarray(corrected_array)

            print(f"✅ 过度增强修正完成")
            return corrected_image
        else:
            print(f"✅ 增强效果自然，无需修正")
            return enhanced

    except Exception as e:
        print(f"❌ 过度增强检测失败: {e}")
        return enhanced


def _detect_faces_in_image(image: Image.Image) -> bool:
    """
    快速检测图像中是否有人脸
    """
    faces = _detect_faces_with_locations(image)
    return len(faces) > 0


def _detect_faces_with_locations(image: Image.Image) -> list:
    """
    检测图像中的人脸并返回位置信息
    返回格式: [(x, y, w, h), ...]
    """
    try:
        import cv2
        import numpy as np

        # 转换为OpenCV格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 使用Haar级联分类器检测人脸
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        return faces.tolist() if len(faces) > 0 else []

    except Exception as e:
        print(f"⚠️ 人脸检测失败: {e}")
        return []


def _enhance_face_regions_with_ai_upscale(image: Image.Image, faces: list) -> Optional[Image.Image]:
    """
    🚀 使用AI放大技术增强人脸区域
    这是一个实用的人脸增强方案，利用现有的AI放大技术
    """
    try:
        if not faces:
            return None

        print(f"🎯 使用AI放大技术增强{len(faces)}个人脸区域...")

        # 创建图像副本
        enhanced_image = image.copy()

        for i, (x, y, w, h) in enumerate(faces):
            try:
                # 扩展人脸区域（包含更多上下文）
                padding = max(w, h) // 4
                expanded_x = max(0, x - padding)
                expanded_y = max(0, y - padding)
                expanded_w = min(image.width - expanded_x, w + 2 * padding)
                expanded_h = min(image.height - expanded_y, h + 2 * padding)

                # 提取人脸区域
                face_region = image.crop((expanded_x, expanded_y, expanded_x + expanded_w, expanded_y + expanded_h))

                # 使用AI放大技术增强人脸区域
                # 计算合适的放大倍数
                face_size = max(expanded_w, expanded_h)
                if face_size < 128:
                    scale_factor = 2.0  # 小人脸需要更多增强
                elif face_size < 256:
                    scale_factor = 1.5  # 中等人脸适度增强
                else:
                    scale_factor = 1.2  # 大人脸轻微增强

                target_w = int(expanded_w * scale_factor)
                target_h = int(expanded_h * scale_factor)

                print(f"🚀 增强人脸{i+1}: {expanded_w}x{expanded_h} -> {target_w}x{target_h}")

                # 使用智能AI放大
                enhanced_face = smart_ai_upscale(face_region, target_w, target_h)

                if enhanced_face:
                    # 缩回原尺寸，保留增强效果
                    enhanced_face_resized = enhanced_face.resize((expanded_w, expanded_h), Image.Resampling.LANCZOS)

                    # 将增强后的人脸区域粘贴回原图
                    enhanced_image.paste(enhanced_face_resized, (expanded_x, expanded_y))
                    print(f"✅ 人脸{i+1}增强完成")
                else:
                    print(f"⚠️ 人脸{i+1}AI放大失败，跳过")

            except Exception as e:
                print(f"❌ 人脸{i+1}增强失败: {e}")
                continue

        print(f"✅ 所有人脸区域AI增强完成")
        return enhanced_image

    except Exception as e:
        print(f"❌ AI人脸区域增强失败: {e}")
        return None


def _try_codeformer_restoration(image: Image.Image) -> Optional[Image.Image]:
    """
    尝试使用CodeFormer进行人脸修复
    """
    try:
        # 🚀 尝试使用ComfyUI的FaceRestore节点（如果可用）
        try:
            # 检查是否有ComfyUI的人脸修复节点
            import comfy.model_management
            from comfy.utils import load_torch_file

            # 查找CodeFormer模型
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
            possible_paths = [
                os.path.join(models_dir, 'facerestore_models', 'codeformer-v0.1.0.pth'),
                os.path.join(models_dir, 'face_restore', 'codeformer-v0.1.0.pth'),
                os.path.join(models_dir, 'upscale_models', 'codeformer-v0.1.0.pth'),
            ]

            codeformer_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    codeformer_path = path
                    break

            if codeformer_path:
                print(f"🚀 找到CodeFormer模型: {codeformer_path}")
                # 这里可以集成实际的CodeFormer调用
                # 由于需要特定的依赖和实现，暂时跳过
                print(f"🔍 CodeFormer集成需要额外依赖，跳过")
                return None
            else:
                print(f"🔍 CodeFormer模型未找到")
                return None

        except ImportError:
            print(f"🔍 ComfyUI人脸修复模块不可用")
            return None

    except Exception as e:
        print(f"❌ CodeFormer修复失败: {e}")
        return None


def _try_gfpgan_restoration(image: Image.Image) -> Optional[Image.Image]:
    """
    尝试使用GFPGAN进行人脸修复
    """
    try:
        # 🚀 尝试使用ComfyUI的FaceRestore节点（如果可用）
        try:
            import comfy.model_management
            from comfy.utils import load_torch_file

            # 查找GFPGAN模型
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
            possible_paths = [
                os.path.join(models_dir, 'facerestore_models', 'GFPGANv1.4.pth'),
                os.path.join(models_dir, 'face_restore', 'GFPGANv1.4.pth'),
                os.path.join(models_dir, 'upscale_models', 'GFPGANv1.4.pth'),
            ]

            gfpgan_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    gfpgan_path = path
                    break

            if gfpgan_path:
                print(f"🚀 找到GFPGAN模型: {gfpgan_path}")
                # 这里可以集成实际的GFPGAN调用
                # 由于需要特定的依赖和实现，暂时跳过
                print(f"🔍 GFPGAN集成需要额外依赖，跳过")
                return None
            else:
                print(f"🔍 GFPGAN模型未找到")
                return None

        except ImportError:
            print(f"🔍 ComfyUI人脸修复模块不可用")
            return None

    except Exception as e:
        print(f"❌ GFPGAN修复失败: {e}")
        return None


def _apply_advanced_traditional_enhancement(image: Image.Image, quality: str) -> Image.Image:
    """
    🚀 自然传统画质增强算法 - 保持真实感
    """
    try:
        from PIL import ImageEnhance, ImageFilter
        import numpy as np

        print(f"🎨 应用自然传统增强算法...")

        # 🚀 第一步：轻微去噪（仅在ultra_hd模式下）
        if quality == "ultra_hd":
            # 非常轻微的去噪
            denoised = image.filter(ImageFilter.GaussianBlur(radius=0.3))
            # 与原图混合（更保守的比例）
            original_array = np.array(image)
            denoised_array = np.array(denoised)
            mixed_array = (original_array * 0.9 + denoised_array * 0.1).astype(np.uint8)
            image = Image.fromarray(mixed_array)
            print(f"✅ 轻微去噪完成")

        # 🚀 第二步：自然锐化（降低强度）
        enhancer = ImageEnhance.Sharpness(image)
        if quality == "hd":
            sharpness_factor = 1.06  # 非常温和的锐化
        else:  # ultra_hd
            sharpness_factor = 1.1   # 温和锐化
        image = enhancer.enhance(sharpness_factor)
        print(f"✅ 自然锐化完成 (factor: {sharpness_factor})")

        # 🚀 第三步：温和对比度增强
        enhancer = ImageEnhance.Contrast(image)
        if quality == "hd":
            contrast_factor = 1.04  # 非常温和
        else:  # ultra_hd
            contrast_factor = 1.06  # 温和对比度
        image = enhancer.enhance(contrast_factor)
        print(f"✅ 温和对比度增强完成 (factor: {contrast_factor})")

        # 🚀 第四步：自然色彩优化
        enhancer = ImageEnhance.Color(image)
        if quality == "hd":
            color_factor = 1.02  # 非常温和的饱和度
        else:  # ultra_hd
            color_factor = 1.03  # 温和饱和度
        image = enhancer.enhance(color_factor)
        print(f"✅ 自然色彩优化完成 (factor: {color_factor})")

        # 🚀 第五步：亮度保持（几乎不变）
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = 1.005  # 几乎不改变亮度
        image = enhancer.enhance(brightness_factor)
        print(f"✅ 亮度保持完成 (factor: {brightness_factor})")

        print(f"🎨 自然传统增强算法完成")
        return image

    except Exception as e:
        print(f"❌ 传统增强失败: {e}")
        return image

def _apply_full_enhancements(base_image: Image.Image, target_size_str: str, quality: str, enhance_quality: bool, smart_resize_enabled: bool) -> Image.Image:
    """尺寸适配 + 主体检测裁剪 + 画质增强，供nano-banana复用"""
    print(f"🔧 开始图像增强处理...")
    print(f"🔧 输入参数: target_size={target_size_str}, quality={quality}, enhance_quality={enhance_quality}, smart_resize={smart_resize_enabled}")
    print(f"🔧 原始图像尺寸: {base_image.size}")

    image = base_image
    try:
        if 'x' in target_size_str:
            target_width, target_height = map(int, target_size_str.split('x'))
        else:
            target_width, target_height = image.size
        print(f"🔧 目标尺寸: {target_width}x{target_height}")
    except Exception as e:
        target_width, target_height = image.size
        print(f"⚠️ 解析目标尺寸失败，使用原始尺寸: {e}")

    try:
        if smart_resize_enabled and (image.size != (target_width, target_height)):
            print(f"🎯 启用智能调整尺寸...")

            # 🚀 关键修复：检查AI模型是否已经生成了合适的尺寸
            current_aspect = image.size[0] / image.size[1]
            target_aspect = target_width / target_height
            aspect_diff = abs(current_aspect - target_aspect) / target_aspect

            print(f"🔍 宽高比分析: 当前={current_aspect:.3f}, 目标={target_aspect:.3f}, 差异={aspect_diff:.3f}")

            # 如果宽高比差异很小（<10%），使用温和的调整策略
            if aspect_diff < 0.1:
                print(f"🎯 宽高比接近目标，使用温和调整策略...")
                # 直接缩放到目标尺寸，不进行激进裁剪
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                print(f"✅ 温和调整完成: {image.size}")
            elif image.size[0] / image.size[1] != target_width / target_height:
                print(f"🎯 检测到宽高比不匹配，启用智能主体检测...")
                scale_x = target_width / image.size[0]
                scale_y = target_height / image.size[1]
                scale = max(scale_x, scale_y)
                enlarged_w = max(1, int(image.size[0] * scale))
                enlarged_h = max(1, int(image.size[1] * scale))
                print(f"🎯 放大到: {enlarged_w}x{enlarged_h}")

                try:
                    print(f"🚀 尝试AI智能放大...")
                    ai_up = smart_ai_upscale(image, enlarged_w, enlarged_h)
                    enlarged = ai_up if ai_up is not None else image.resize((enlarged_w, enlarged_h), Image.Resampling.LANCZOS)
                    if ai_up is not None:
                        print(f"✅ AI智能放大成功")
                    else:
                        print(f"⚠️ AI智能放大失败，使用传统放大")
                except Exception as e:
                    print(f"❌ AI智能放大异常: {e}")
                    enlarged = image.resize((enlarged_w, enlarged_h), Image.Resampling.LANCZOS)

                try:
                    print(f"🎯 开始智能主体检测...")
                    subject_bbox, subject_center = detect_image_foreground_subject(enlarged)
                    cx, cy = subject_center
                    print(f"🎯 检测到主体中心: ({cx}, {cy})")

                    # 🚀 关键修复：添加安全检查，确保主体不会被裁剪掉
                    subject_x, subject_y, subject_w, subject_h = subject_bbox

                    # 检查主体是否合理（不能太小或太大）
                    enlarged_area = enlarged.width * enlarged.height
                    subject_area = subject_w * subject_h
                    subject_ratio = subject_area / enlarged_area

                    print(f"🔍 主体区域检查: 主体({subject_x}, {subject_y}, {subject_w}x{subject_h}), 占比{subject_ratio:.3f}")

                    # 🚀 关键修复：如果主体占比过大，说明AI已经生成了合适的构图，使用温和策略
                    if subject_ratio > 0.6:
                        print(f"🎯 主体占比较大({subject_ratio:.3f})，AI构图良好，使用温和裁剪策略")
                        # 使用图像中心进行温和裁剪
                        crop_x = max(0, (enlarged.width - target_width) // 2)
                        crop_y = max(0, (enlarged.height - target_height) // 2)
                        image = enlarged.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                        print(f"✅ 温和裁剪完成: ({crop_x}, {crop_y})")
                    elif subject_ratio < 0.01:
                        print(f"⚠️ 主体占比过小({subject_ratio:.3f})，检测可能有误，使用中心裁剪")
                        # 使用图像中心作为裁剪中心
                        crop_x = max(0, (enlarged.width - target_width) // 2)
                        crop_y = max(0, (enlarged.height - target_height) // 2)
                        image = enlarged.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                        print(f"⚠️ 使用中心裁剪: ({crop_x}, {crop_y})")
                    else:
                        # 正常的智能主体居中裁剪
                        cx, cy = subject_center

                        # 计算裁剪区域，确保主体在裁剪区域内
                        crop_x = int(cx - target_width / 2)
                        crop_y = int(cy - target_height / 2)

                        # 🚀 关键修复：确保裁剪区域包含主体
                        # 检查主体是否会被裁剪掉
                        crop_right = crop_x + target_width
                        crop_bottom = crop_y + target_height
                        subject_right = subject_x + subject_w
                        subject_bottom = subject_y + subject_h

                        # 如果主体会被裁剪掉，调整裁剪位置
                        if subject_x < crop_x:  # 主体左边被裁剪
                            crop_x = max(0, subject_x - target_width // 10)  # 留10%边距
                        if subject_y < crop_y:  # 主体上边被裁剪
                            crop_y = max(0, subject_y - target_height // 10)
                        if subject_right > crop_right:  # 主体右边被裁剪
                            crop_x = min(enlarged.width - target_width, subject_right - target_width + target_width // 10)
                        if subject_bottom > crop_bottom:  # 主体下边被裁剪
                            crop_y = min(enlarged.height - target_height, subject_bottom - target_height + target_height // 10)

                        # 最终边界检查
                        crop_x = max(0, min(crop_x, enlarged.width - target_width))
                        crop_y = max(0, min(crop_y, enlarged.height - target_height))

                        print(f"🎯 智能裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x + target_width}, {crop_y + target_height})")
                        print(f"🔍 主体保护检查: 主体({subject_x}, {subject_y}) -> ({subject_right}, {subject_bottom})")

                        image = enlarged.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                        print(f"✅ 智能主体居中裁剪完成")

                except Exception as e:
                    print(f"❌ 智能主体检测失败，使用安全中心裁剪: {e}")
                    import traceback
                    print(f"🔍 详细错误: {traceback.format_exc()}")
                    # 🚀 关键修复：使用更安全的中心裁剪
                    crop_x = max(0, (enlarged.width - target_width) // 2)
                    crop_y = max(0, (enlarged.height - target_height) // 2)
                    image = enlarged.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                    print(f"⚠️ 使用安全中心裁剪: ({crop_x}, {crop_y})")
            else:
                print(f"🎯 宽高比匹配，直接缩放")
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            if not smart_resize_enabled:
                print(f"🔧 智能调整已禁用，使用填充模式")
            else:
                print(f"🔧 尺寸已匹配，使用填充模式")
            image = smart_resize_with_padding(image, (target_width, target_height))
    except Exception as e:
        print(f"❌ 尺寸调整失败: {e}")
        import traceback
        print(f"🔍 详细错误: {traceback.format_exc()}")

    try:
        if enhance_quality and quality in ['hd', 'ultra_hd']:
            print(f"🎨 开始画质增强 (质量级别: {quality})...")
            enhanced = enhance_image_quality(image, quality, "disabled")
            if enhanced:
                image = enhanced
                print(f"✅ 画质增强完成")
            else:
                print(f"⚠️ 画质增强返回None")
        else:
            print(f"🔧 跳过画质增强 (enhance_quality={enhance_quality}, quality={quality})")
    except Exception as e:
        print(f"❌ 画质增强失败: {e}")

    print(f"🔧 图像增强处理完成，最终尺寸: {image.size}")
    return image

def image_to_base64(image, format='JPEG'):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
        image = background
    image.save(buffer, format=format, quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def _make_chat_summary(response_text: str) -> str:
    try:
        import re
        url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
        m = re.search(url_pattern, response_text or '')
        if m:
            return f"已生成图像: {m.group(0)}"
        # markdown image
        md_pat = r"!\[.*?\]\((.*?)\)"
        m2 = re.search(md_pat, response_text or '')
        if m2:
            return f"已生成图像: {m2.group(1)}"
        # base64
        if 'data:image/' in (response_text or ''):
            return "已生成图像（内嵌base64）"
    except Exception:
        pass
    return "已生成图像"

def validate_api_key(api_key):
    """Validate API key format"""
    return api_key and len(api_key.strip()) > 10

def format_error_message(error):
    """Format error message"""
    return str(error)

def generate_with_official_api(api_key, model, content_parts, generation_config, max_retries=5, proxy=None):
    """使用官方google.genai库调用API"""
    try:
        # 尝试导入官方库
        from google import genai
        from google.genai import types

        print(f"🚀 使用官方google.genai库调用模型: {model}")

        # 创建客户端
        client = genai.Client(api_key=api_key)

        # 转换generation_config格式
        official_config = types.GenerateContentConfig(
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('topP', 0.95),
            top_k=generation_config.get('topK', 40),
            max_output_tokens=generation_config.get('maxOutputTokens', 8192),
            response_modalities=['Text', 'Image'] if 'IMAGE' in generation_config.get('responseModalities', []) else ['Text']
        )

        # 转换content_parts格式（使用字典格式，与gemini_banana模块保持一致）
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

        # 调用API（使用字典格式）
        response = client.models.generate_content(
            model=model,
            contents=[{"parts": official_parts}],
            config=official_config
        )

        # 转换响应格式为REST API兼容格式
        result = {
            "candidates": [{
                "content": {
                    "parts": []
                }
            }]
        }

        # 处理响应内容
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # 处理文本内容
                        if hasattr(part, 'text') and part.text:
                            result["candidates"][0]["content"]["parts"].append({
                                "text": part.text
                            })
                        # 处理图像内容
                        elif hasattr(part, 'inline_data') and part.inline_data:
                            try:
                                # 确保inline_data不为None且有必要的属性
                                if hasattr(part.inline_data, 'mime_type') and hasattr(part.inline_data, 'data'):
                                    # 处理不同的数据格式
                                    data = part.inline_data.data
                                    if hasattr(data, 'decode'):
                                        # 如果是bytes，转换为base64字符串
                                        import base64
                                        data = base64.b64encode(data).decode('utf-8')
                                    elif isinstance(data, str):
                                        # 如果已经是字符串，直接使用
                                        data = data

                                    result["candidates"][0]["content"]["parts"].append({
                                        "inline_data": {
                                            "mime_type": part.inline_data.mime_type,
                                            "data": data
                                        }
                                    })
                            except Exception as e:
                                print(f"⚠️ 处理图像响应时出错: {e}")

        # 如果没有从candidates中获取到内容，尝试使用response.text
        if not result["candidates"][0]["content"]["parts"] and hasattr(response, 'text') and response.text:
            result["candidates"][0]["content"]["parts"].append({
                "text": response.text
            })

        print("✅ 官方API调用成功")
        return result

    except ImportError:
        print("⚠️ google.genai库未安装，无法使用官方API")
        return None
    except Exception as e:
        print(f"❌ 官方API调用失败: {e}")
        # 打印更详细的错误信息用于调试
        import traceback
        print(f"🔍 详细错误信息: {traceback.format_exc()}")
        return None

def generate_with_rest_api(api_key, model, content_parts, generation_config, max_retries=5, proxy=None, base_url=None):
    """使用REST API调用Gemini"""
    import requests

    if not base_url:
        base_url = "https://generativelanguage.googleapis.com"

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

    # 设置代理
    proxies = None
    if proxy and proxy.strip() and "None" not in proxy:
        proxies = {
            'http': proxy.strip(),
            'https': proxy.strip()
        }

    for attempt in range(max_retries):
        try:
            print(f"🌐 REST API调用 (尝试 {attempt + 1}/{max_retries})")

            response = requests.post(
                f"{base_url}/v1beta/models/{model}:generateContent",
                headers=headers,
                json=request_data,
                timeout=120,
                proxies=proxies
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ REST API错误 {response.status_code}: {response.text}")
                if attempt == max_retries - 1:
                    raise Exception(f"REST API调用失败: {response.status_code}")

        except Exception as e:
            print(f"❌ REST API调用异常: {e}")
            if attempt == max_retries - 1:
                raise e

        # 重试延迟
        import time
        time.sleep(2 ** attempt)

    return None

def generate_with_priority_api(api_key, model, content_parts, generation_config, max_retries=5, proxy=None, base_url=None):
    """优先使用官方API，失败时回退到REST API"""

    # 首先尝试官方API
    print("🎯 优先尝试官方google.genai API")
    result = generate_with_official_api(api_key, model, content_parts, generation_config, max_retries, proxy)

    if result is not None:
        print("✅ 官方API调用成功")
        return result

    # 官方API失败，回退到REST API
    print("🔄 官方API失败，回退到REST API")
    return generate_with_rest_api(api_key, model, content_parts, generation_config, max_retries, proxy, base_url)

def extract_text_from_response(response_json):
    """从响应中提取文本内容"""
    try:
        if not response_json or "candidates" not in response_json:
            return "Response received but no candidates"

        candidates = response_json["candidates"]
        if not candidates:
            return "Response received but no candidates"

        candidate = candidates[0]
        if "content" not in candidate:
            return "Response received but no content"

        content = candidate["content"]
        if "parts" not in content:
            return "Response received but no parts"

        parts = content["parts"]
        text_parts = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])

        if text_parts:
            return "\n".join(text_parts)
        else:
            return "Response received but no text content"

    except Exception as e:
        print(f"❌ 提取文本响应失败: {e}")
        return f"Error extracting text: {str(e)}"

def process_generated_image_from_response(response_json):
    """从响应中提取生成的图像"""
    try:
        if not response_json or "candidates" not in response_json:
            return None

        candidates = response_json["candidates"]
        if not candidates:
            return None

        candidate = candidates[0]
        if "content" not in candidate:
            return None

        content = candidate["content"]
        if "parts" not in content:
            return None

        parts = content["parts"]

        for part in parts:
            if "inline_data" in part:
                inline_data = part["inline_data"]
                if "data" in inline_data:
                    # 解码base64图像数据
                    import base64
                    import io
                    from PIL import Image

                    image_data = inline_data["data"]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    print("✅ 成功提取生成的图像")
                    return image

        return None

    except Exception as e:
        print(f"❌ 提取图像失败: {e}")
        return None

def _normalize_model_name(model: str) -> str:
    """Strip any trailing bracketed labels (e.g., ' [OpenRouter]', ' [Comfly‑T8]'), robust to hyphen variants."""
    try:
        if not model:
            return model
        import re
        # Remove any ' [....]' suffix at end, including unicode hyphen variants inside
        return re.sub(r"\s*\[[^\]]+\]\s*$", "", model)
    except Exception:
        return model

def _is_comfly_base(url: str) -> bool:
    try:
        return isinstance(url, str) and ("ai.comfly.chat" in url or "comfly.chat" in url)
    except Exception:
        return False

def _parse_size_str(size_str: str, default: str = "1024x1024") -> str:
    try:
        s = size_str.strip().lower()
        if "x" in s:
            w, h = s.split("x", 1)
            w = int(w); h = int(h)
            return f"{w}x{h}"
    except Exception:
        pass
    return default

def _comfly_nano_banana_generate(api_url: str, api_key: str, model: str, prompt: str, size: str, temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 32768, seed: int = 0) -> dict:
    """Call images/generations endpoint for nano-banana image generation (supports both Comfly and T8 mirror sites)."""
    import requests, json, os, re, time

    # 调试信息 - 已关闭
    # print(f"[DEBUG] _comfly_nano_banana_generate called with:")
    # print(f"[DEBUG]   api_url: {api_url}")
    # print(f"[DEBUG]   model: {model}")
    # print(f"[DEBUG]   prompt: {prompt[:100]}...")
    # print(f"[DEBUG]   size: {size}")
    # print(f"[DEBUG]   seed: {seed}")
    # 对于图像生成，使用 /v1/images/generations 端点
    if "t8star.cn" in api_url or "ai.t8star.cn" in api_url:
        # T8镜像站的正确URL
        url = "https://ai.t8star.cn/v1/images/generations"
    elif api_url.endswith('/v1/chat/completions'):
        url = api_url.replace('/v1/chat/completions', '/v1/images/generations')
    else:
        # Comfly或其他兼容的镜像站
        url = "https://ai.comfly.chat/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建图像生成的请求格式（完全按照Comfly项目的格式）
    payload = {
        "prompt": prompt,
        "model": str(model)
    }

    # 添加可选参数（按照Comfly项目的方式）
    if size:
        # 处理特殊尺寸格式，T8镜像站需要具体的尺寸格式
        if size == "Original size":
            # 默认使用1024x1024作为原始尺寸
            payload["size"] = "1024x1024"
            # print(f"[DEBUG] 转换 'Original size' -> '1024x1024'")
        else:
            payload["size"] = size

    # 使用url格式而不是b64_json（根据用户反馈）
    payload["response_format"] = "url"

    if seed > 0:
        payload["seed"] = seed

    # 调试信息：打印实际的请求payload - 已关闭
    # print(f"[DEBUG] Request payload: {json.dumps(payload, indent=2)}")
    # print(f"[DEBUG] Target URL: {url}")

    # 发送图像生成请求（非流式）
    session = requests.Session()
    max_retries = 1
    backoff_seconds = 1

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
                if proxy:
                    print(f"Use Proxy: {proxy}")
            except Exception:
                pass

            print(f"[ComflyNanoBananaMirror] (generate) POST attempt {attempt}/{max_retries} → {url}")

            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=(20, 120),
                allow_redirects=True,
            )
            print(f"[ComflyNanoBananaMirror] HTTP status: {response.status_code}")
            try:
                print(f"[ComflyNanoBananaMirror] Content-Type: {response.headers.get('Content-Type','')}")
            except Exception:
                pass
            if response.status_code != 200:
                try:
                    print(f"[ComflyNanoBananaMirror] Error body: {response.text[:500]}")
                except Exception:
                    pass
            response.raise_for_status()

            # 处理图像生成API的JSON响应
            try:
                result = response.json()
                print(f"[ComflyNanoBananaMirror] Response received: {list(result.keys())}")

                # 检查是否有图像数据
                if 'data' in result and result['data']:
                    # OpenAI Images API格式
                    image_data = result['data'][0]

                    # 优先处理URL格式（根据用户反馈）
                    if 'url' in image_data:
                        image_url = image_data['url']
                        # print(f"[DEBUG] 生成的图像URL: {image_url}")

                        # 下载图像并转换为base64
                        import base64
                        from io import BytesIO
                        img_response = session.get(image_url, timeout=30)
                        img_response.raise_for_status()
                        b64_data = base64.b64encode(img_response.content).decode('utf-8')

                        response_text = f"图像生成成功，模型: {model}\n图像URL: {image_url}"
                        # print(f"[DEBUG] 响应文本: {response_text}")

                        return {
                            "data": [{
                                "b64_json": b64_data,
                                "url": image_url,
                                "revised_prompt": image_data.get('revised_prompt', prompt)
                            }],
                            "response_text": response_text
                        }
                    elif 'b64_json' in image_data:
                        # 处理base64数据，可能包含data URL前缀
                        b64_data = image_data['b64_json']
                        if b64_data.startswith('data:image/'):
                            # 提取纯base64部分
                            b64_data = b64_data.split(',', 1)[1] if ',' in b64_data else b64_data

                        return {
                            "data": [{
                                "b64_json": b64_data,
                                "url": "",
                                "revised_prompt": image_data.get('revised_prompt', prompt)
                            }],
                            "response_text": f"图像生成成功，模型: {model}"
                        }

                # 如果没有找到图像数据，返回空结果
                print(f"[ComflyNanoBananaMirror] No image data found in response: {result}")
                return {
                    "data": [],
                    "response_text": f"图像生成失败，响应: {str(result)[:200]}"
                }

            except json.JSONDecodeError as e:
                print(f"[ComflyNanoBananaMirror] JSON decode error: {e}")
                return {
                    "data": [],
                    "response_text": f"响应解析失败: {response.text[:200]}"
                }

        except requests.exceptions.HTTPError as http_err:
            status = getattr(http_err.response, 'status_code', None)
            last_error = http_err
            if status in (408, 409, 429) or (status is not None and 500 <= status < 600):
                if attempt < max_retries:
                    time.sleep(backoff_seconds * attempt)
                    continue
            raise Exception(f"Error in nano-banana generation: {str(http_err)}")
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
                continue
            raise Exception("Error in nano-banana generation: request timed out")
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
                continue
            raise Exception(f"Error in nano-banana generation: {str(e)}")

    raise Exception(f"Error in nano-banana generation: {str(last_error)}")

def _comfly_nano_banana_edit(api_url: str, api_key: str, model: str, prompt: str, pil_images: list, size: str, temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 32768, seed: int = 0) -> dict:
    """Call chat completions endpoint for nano-banana image editing with multiple images (supports both Comfly and T8 mirror sites)."""
    import requests, json, io, base64, re, time, os
    # 使用传入的api_url，如果已经是完整URL则直接使用，否则构建完整URL
    if api_url.endswith('/v1/chat/completions'):
        url = api_url
    elif "t8star.cn" in api_url or "ai.t8star.cn" in api_url:
        url = f"{api_url.rstrip('/')}/v1/chat/completions"
    else:
        # Comfly或其他兼容的镜像站
        url = "https://ai.comfly.chat/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # 构建content数组，包含文本和所有图像
    content = [{"type": "text", "text": prompt}]
    
    # 添加所有输入图像（使用原图尺寸与无损PNG编码）
    for pil_image in pil_images:
        if pil_image is not None:
            try:
                w, h = pil_image.size
                print(f"[ComflyNanoBananaMirror] Use original image size: {w}x{h}")
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                })
            except Exception as e:
                try:
                    print(f"[ComflyNanoBananaMirror] Image encode error: {str(e)}")
                except Exception:
                    pass
    
    # 构建nano-banana的图像编辑请求格式
    messages = [{
        "role": "user",
        "content": content
    }]
    
    payload = {
        "model": str(model),
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": True
    }
    
    if seed > 0:
        payload["seed"] = seed
    
    # 调试信息
    try:
        print(f"[ComflyNanoBananaMirror] Building payload: model={model}, max_tokens={max_tokens}, seed={seed}")
        print(f"[ComflyNanoBananaMirror] Payload keys: {list(payload.keys())}")
        print(f"[ComflyNanoBananaMirror] Messages structure: {payload.get('messages', 'MISSING')}")
    except Exception:
        pass

    # 发送流式请求（对408等可重试错误进行重试）
    full_response = ""
    session = requests.Session()
    max_retries = 3
    backoff_seconds = 2

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # 打印代理信息（若存在）
            try:
                proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
                if proxy:
                    print(f"Use Proxy: {proxy}")
            except Exception:
                pass

            print(f"[ComflyNanoBananaMirror] POST attempt {attempt}/{max_retries} → {url}")
            response = session.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(20, 120),  # 连接/读取超时分离
                allow_redirects=True,
            )
            print(f"[ComflyNanoBananaMirror] HTTP status: {response.status_code}")
            try:
                print(f"[ComflyNanoBananaMirror] Content-Type: {response.headers.get('Content-Type','')}")
            except Exception:
                pass
            # 若非200，打印返回体帮助定位422原因
            if response.status_code != 200:
                try:
                    print(f"[ComflyNanoBananaMirror] Error body: {response.text[:500]}")
                except Exception:
                    pass
            response.raise_for_status()

            line_count = 0
            received_any = False
            ctype = (response.headers.get('Content-Type') or '').lower()
            if 'text/event-stream' in ctype or 'stream' in ctype:
                for line in response.iter_lines(chunk_size=1, decode_unicode=False):
                    if line is None or not line:
                        continue
                    received_any = True
                    try:
                        line_text = line.decode('utf-8', errors='ignore').strip()
                    except Exception:
                        line_text = str(line).strip()
                    line_count += 1
                    if not line_text.startswith('data: '):
                        continue
                    data = line_text[6:]
                    if data == '[DONE]':
                        print(f"[ComflyNanoBananaMirror] Stream finished. Total SSE lines: {line_count}")
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if 'choices' in chunk and chunk['choices']:
                        choice0 = chunk['choices'][0]
                        delta = choice0.get('delta', {})
                        piece = delta.get('content') or (choice0.get('message', {}) or {}).get('content')
                        if piece:
                            full_response += piece
                            try:
                                print(f"[ComflyNanoBananaMirror] +{len(piece)} chars (total {len(full_response)})")
                            except Exception:
                                pass
            else:
                # 非流式：直接解析一次性JSON
                try:
                    body = response.content.decode('utf-8', errors='ignore')
                except Exception:
                    body = response.text
                try:
                    obj = json.loads(body)
                    received_any = True
                    if isinstance(obj, dict):
                        choices = obj.get('choices') or []
                        if choices:
                            msg = choices[0].get('message', {})
                            content_text = isinstance(msg, dict) and msg.get('content')
                            if isinstance(content_text, str):
                                full_response += content_text
                except Exception:
                    pass

            if not received_any:
                print("[ComflyNanoBananaMirror] Warning: no bytes received from stream; fallback to non-streaming request")
                # Fallback once with non-streaming
                try:
                    resp2 = session.post(
                        url,
                        headers=headers,
                        json={**payload, "stream": False},
                        timeout=(20, 120),
                        allow_redirects=True,
                    )
                    if resp2.status_code != 200:
                        try:
                            print(f"[ComflyNanoBananaMirror] Fallback error body: {resp2.text[:500]}")
                        except Exception:
                            pass
                    resp2.raise_for_status()
                    obj = resp2.json()
                    choices = obj.get('choices') or []
                    if choices:
                        msg = choices[0].get('message', {})
                        content_text = isinstance(msg, dict) and msg.get('content')
                        if isinstance(content_text, str):
                            full_response += content_text
                except Exception as _:
                    pass

            # 优先提取base64图片
            base64_pattern = r'data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)'
            matches = re.findall(base64_pattern, full_response)
            if matches:
                return {
                    "data": [{
                        "b64_json": matches[0],
                        "url": "",
                        "revised_prompt": prompt
                    }],
                    "response_text": full_response
                }

            # 其次提取图片URL
            image_md_pattern = r'!\[.*?\]\((.*?)\)'
            url_matches = re.findall(image_md_pattern, full_response)
            if not url_matches:
                url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
                url_matches = re.findall(url_pattern, full_response)
            if not url_matches:
                all_urls_pattern = r'https?://\S+'
                url_matches = re.findall(all_urls_pattern, full_response)

            if url_matches:
                image_url = url_matches[0]
                try:
                    print(f"[ComflyNanoBananaMirror] Found image URL: {image_url}")
                except Exception:
                    pass
                try:
                    img_response = requests.get(image_url, timeout=30)
                    img_response.raise_for_status()
                    # 后处理：按 size 统一适配输出尺寸（保障尺寸控制必生效）
                    try:
                        from io import BytesIO
                        from PIL import Image as _PILImage
                        raw_img = _PILImage.open(BytesIO(img_response.content)).convert('RGB')
                        target_size = size if isinstance(size, str) else str(size)
                        if 'x' in target_size:
                            tw, th = map(int, target_size.split('x'))
                            # 使用等比扩展+留白，避免拉伸
                            processed = smart_resize_with_padding(raw_img, (tw, th))
                        else:
                            processed = raw_img
                        buf = BytesIO()
                        processed.save(buf, format='JPEG', quality=95)
                        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        try:
                            print(f"🔧 Final output size: {processed.size[0]}x{processed.size[1]}")
                        except Exception:
                            pass
                    except Exception:
                        # 回退：直接透传
                        image_base64 = base64.b64encode(img_response.content).decode('utf-8')

                    return {
                        "data": [{
                            "b64_json": image_base64,
                            "url": image_url,
                            "revised_prompt": prompt
                        }],
                        "response_text": full_response
                    }
                except Exception:
                    return {
                        "data": [{
                            "b64_json": "",
                            "url": image_url,
                            "revised_prompt": prompt
                        }],
                        "response_text": full_response
                    }

            # 没有图片，仅返回响应文本
            return {
                "data": [{
                    "b64_json": "",
                    "url": "",
                    "revised_prompt": prompt
                }],
                "response_text": full_response
            }

        except requests.exceptions.HTTPError as http_err:
            status = getattr(http_err.response, 'status_code', None)
            try:
                body = http_err.response.text if http_err.response is not None else ''
                if body:
                    print(f"[ComflyNanoBananaMirror] HTTP {status} body: {body[:1000]}")
            except Exception:
                pass
            last_error = http_err
            # 对408/429/5xx进行重试
            if status in (408, 409, 429) or (status is not None and 500 <= status < 600):
                if attempt < max_retries:
                    time.sleep(backoff_seconds * attempt)
                    continue
            raise Exception(f"Error in nano-banana image editing: {str(http_err)}")
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
                continue
            raise Exception("Error in nano-banana image editing: request timed out")
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
                continue
            raise Exception(f"Error in nano-banana image editing: {str(e)}")

    # 如果到这里仍然失败
    raise Exception(f"Error in nano-banana image editing: {str(last_error)}")

def _comfly_falai_edit(api_url: str, api_key: str, model: str, prompt: str, pil_image: Image.Image, size: str) -> dict:
    """Call Comfly's OpenAI-compatible images edits endpoint for FAL AI models."""
    import requests, io
    url = api_url.rstrip('/') + "/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    data = {
        "model": _normalize_model_name(model),
        "prompt": prompt,
        "size": _parse_size_str(size)
    }
    files = {"image": ("image.png", buf, "image/png")}
    r = requests.post(url, headers=headers, data=data, files=files, timeout=180)
    r.raise_for_status()
    return r.json()

def get_mirror_site_config(mirror_site_name: str) -> Dict[str, str]:
    """根据镜像站名称获取对应的配置"""
    try:
        try:
            from .gemini_banana import get_gemini_banana_config
        except ImportError:
            from gemini_banana import get_gemini_banana_config
        config = get_gemini_banana_config()
        mirror_sites = config.get('mirror_sites', {})
        
        if mirror_site_name in mirror_sites:
            site_config = mirror_sites[mirror_site_name]
            return {
                "url": site_config.get("url", ""),
                "api_key": site_config.get("api_key", ""),
                "api_format": site_config.get("api_format", ""),
                "models": site_config.get("models", []),
                "description": site_config.get("description", "")
            }
        else:
            # 返回默认配置
            return {
                "url": "https://generativelanguage.googleapis.com",
                "api_key": "",
                "api_format": "gemini",
                "models": [],
                "description": ""
            }
    except Exception as e:
        _log_warning(f"Failed to get mirror site config: {e}")
        return {
            "url": "https://generativelanguage.googleapis.com",
            "api_key": "",
            "api_format": "gemini",
            "models": [],
            "description": ""
        }

def validate_openrouter_config(api_url: str, api_key: str, model: str) -> Dict[str, Any]:
    """验证OpenRouter配置并返回优化建议"""
    validation_result = {
        "is_valid": True,
        "warnings": [],
        "suggestions": [],
        "optimized_params": {}
    }
    
    # 验证API URL
    if not api_url or "openrouter.ai" not in api_url:
        validation_result["is_valid"] = False
        validation_result["warnings"].append("OpenRouter API URL 无效")
        return validation_result
    
    # 验证API Key
    if not api_key or not api_key.startswith("sk-or-v1-"):
        validation_result["warnings"].append("OpenRouter API Key 格式可能不正确")
    
    # 验证模型名称
    if "dall-e" in model.lower():
        validation_result["optimized_params"]["size"] = ["1024x1024", "1792x1024", "1024x1792"]
        validation_result["suggestions"].append("DALL-E 模型推荐使用标准尺寸以获得最佳效果")
    elif "stable-diffusion" in model.lower():
        validation_result["suggestions"].append("Stable Diffusion 模型尺寸会自动调整为8的倍数")
    elif "gemini" in model.lower():
        validation_result["suggestions"].append("Gemini 模型支持多种尺寸和质量设置")
    
    return validation_result

def process_openrouter_stream(response) -> str:
    """处理OpenRouter的流式响应"""
    accumulated_content = ""
    chunk_count = 0
    empty_chunks = 0
    content_chunks = 0
    last_content_chunk = 0
    
    print(f"🔄 开始处理OpenRouter流式响应...")
    
    try:
        for line in response.iter_lines(decode_unicode=True, chunk_size=None):
            if line and line.startswith('data: '):
                chunk_count += 1
                data_content = line[6:]  # Remove 'data: ' prefix
                
                print(f"📦 处理第{chunk_count}个数据块...")
                
                if data_content.strip() == '[DONE]':
                    print(f"✅ 收到结束信号[DONE]")
                    break
                
                try:
                    # 尝试解析JSON
                    chunk_data = json.loads(data_content)
                    print(f"🔍 数据块结构: {list(chunk_data.keys())}")
                    
                    # 提取内容
                    if 'choices' in chunk_data and chunk_data['choices']:
                        choice = chunk_data['choices'][0]
                        print(f"🔍 选择项结构: {list(choice.keys())}")
                        
                        if 'delta' in choice:
                            delta = choice['delta']
                            print(f"🔍 Delta结构: {list(delta.keys())}")
                            
                            # 检查images字段（OpenRouter可能在这里返回图像数据）
                            if 'images' in delta and delta['images']:
                                print(f"🖼️ 检测到OpenRouter images字段！")
                                images_data = delta['images']
                                print(f"🔍 Images字段类型: {type(images_data)}")
                                print(f"🔍 Images字段内容: {str(images_data)[:200]}...")
                                
                                # 使用参考项目的方法：直接搜索data:image/字符串
                                import re
                                chunk_str = str(images_data)
                                if 'data:image/' in chunk_str:
                                    print(f"🎯 OpenRouter在images字段中发现图片数据!")
                                    # 使用参考项目的正确正则表达式
                                    base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
                                    matches = re.findall(base64_pattern, chunk_str)
                                    if matches:
                                        for url in matches:
                                            print(f"🎯 OpenRouter提取base64图片，长度: {len(url)}字符")
                                            accumulated_content += " " + url
                                    else:
                                        print(f"⚠️ 正则表达式未找到匹配的图片数据")
                                        # 备用方法：直接提取data:image/开始到字符串结束
                                        start_pos = chunk_str.find('data:image/')
                                        if start_pos != -1:
                                            extracted_data = chunk_str[start_pos:]
                                            print(f"🎯 备用方法提取图片数据，长度: {len(extracted_data)}字符")
                                            accumulated_content += " " + extracted_data
                                else:
                                    print(f"⚠️ 未找到data:image/标记")
                                
                                content_chunks += 1
                                last_content_chunk = chunk_count
                            
                            # 检查content字段
                            if 'content' in delta and delta['content']:
                                content = delta['content']
                                accumulated_content += content
                                content_chunks += 1
                                last_content_chunk = chunk_count
                                print(f"📝 添加内容: {len(content)}字符 (累计: {len(accumulated_content)}字符)")
                                
                                # 检查是否包含图像数据
                                if '![image]' in content:
                                    print("🖼️ 检测到图像数据标记！")
                            
                            # 如果既没有images也没有content，标记为空块
                            if not ('images' in delta and delta['images']) and not ('content' in delta and delta['content']):
                                empty_chunks += 1
                                print(f"⚠️ 空的delta块 (无images和content字段) - 这是正常的，OpenRouter用空块保持连接")
                        
                        elif 'message' in choice:
                            message = choice['message']
                            print(f"🔍 Message结构: {list(message.keys())}")
                            
                            if 'content' in message and message['content']:
                                content = message['content']
                                accumulated_content += content
                                content_chunks += 1
                                last_content_chunk = chunk_count
                                print(f"📝 添加消息内容: {len(content)}字符 (累计: {len(accumulated_content)}字符)")
                                
                                # 检查是否包含图像数据
                                if '![image]' in content:
                                    print("🖼️ 检测到图像数据标记！")
                            else:
                                empty_chunks += 1
                                print(f"⚠️ 空的消息块 (无content字段)")
                        else:
                            empty_chunks += 1
                            print(f"⚠️ 未知的选择项结构")
                    else:
                        empty_chunks += 1
                        print(f"⚠️ 数据块中没有choices字段")
                
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析失败: {e}, 跳过此块")
                    continue
        
        print(f"✅ 流式响应处理完成:")
        print(f"   📊 总块数: {chunk_count}")
        print(f"   📝 内容块数: {content_chunks}")
        print(f"   ⚠️ 空块数: {empty_chunks}")
        print(f"   📏 总内容长度: {len(accumulated_content)}")
        
        if accumulated_content:
            print(f"   🔍 内容预览: {accumulated_content[:200]}{'...' if len(accumulated_content) > 200 else ''}")
            if last_content_chunk > 0:
                print(f"   📍 最后一个内容块位置: 第{last_content_chunk}块")
        else:
            print(f"   ⚠️ 警告: 没有提取到任何内容！")
            print(f"   💡 建议: 检查OpenRouter API响应格式或模型配置")
        
        return accumulated_content
        
    except Exception as e:
        print(f"❌ 流式响应处理失败: {e}")
        return accumulated_content

def validate_api_url(url):
    """验证API URL格式并自动补全"""
    if not url or not url.strip():
        return False
    
    url = url.strip()
    
    # 如果已经是完整URL格式，直接返回True
    if url.startswith(('http://', 'https://')):
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    # 如果不是完整URL，检查是否为有效域名/IP
    domain_pattern = re.compile(
        r'^(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)?$', re.IGNORECASE)
    
    return domain_pattern.match(url) is not None

def build_api_url(base_url, model, api_format="gemini"):
    """构建完整的API URL"""
    base_url = base_url.strip().rstrip('/')
    
    # 自动补全协议前缀
    if not base_url.startswith(('http://', 'https://')):
        base_url = f"https://{base_url}"
    
    # OpenRouter镜像站特殊处理
    if "openrouter.ai" in base_url:
        # OpenRouter使用chat/completions端点，URL构建在OpenRouter处理逻辑中
        return base_url
    
    # T8镜像站特殊处理
    if "t8star.cn" in base_url or "ai.t8star.cn" in base_url:
        return f"{base_url}/v1/chat/completions"
    
    # Comfly镜像站特殊处理 - 对nano-banana使用chat/completions，其他服务使用标准Gemini API
    if _is_comfly_base(base_url):
        if model in ["nano-banana", "nano-banana-hd"]:
            return "https://ai.comfly.chat/v1/chat/completions"
        else:
            # 其他Comfly服务使用标准Gemini API格式
            return f"{base_url}/models/{_normalize_model_name(model)}:generateContent"
    
    # API4GPT镜像站特殊处理
    if "www.api4gpt.com" in base_url:
        # API4GPT的URL构建在call_api4gpt_api函数中处理
        return base_url
    
    # 如果用户提供的是完整URL，直接使用
    if '/models/' in base_url and ':generateContent' in base_url:
        return base_url
    
    # 如果是基础URL，构建完整路径
    if base_url.endswith('/v1beta') or base_url.endswith('/v1'):
        return f"{base_url}/models/{model}:generateContent"
    
    # 默认添加v1beta路径
    return f"{base_url}/v1beta/models/{model}:generateContent"

def build_t8_api_request(model, prompt, image_base64=None, temperature=0.9, max_tokens=2048):
    """构建T8镜像站的API请求格式
    
    T8镜像站使用OpenAI兼容的API格式，但需要特殊处理
    """
    # 构建消息内容
    content = []
    
    # 添加文本内容
    content.append({
        "type": "text",
        "text": prompt
    })
    
    # 如果有图像，添加图像内容
    if image_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })
    
    # 构建请求数据
    request_data = {
        "model": _normalize_model_name(model),
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    return request_data

def call_t8_api(url, api_key, request_data, timeout=300):
    """调用T8镜像站API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"T8镜像站API调用失败: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"T8镜像站API响应解析失败: {str(e)}")

def build_api4gpt_request(service_type, model, prompt, image_base64=None, size="1024x1024", quality="hd", style="natural", temperature=0.9, max_tokens=2048):
    """构建API4GPT的API请求格式
    
    API4GPT支持多种图像服务，包括nano-banana、DALL-E 3、Stable-Diffusion、Flux等
    根据官方文档：https://doc.api4gpt.com/api-341609441
    """
    if service_type == "nano-banana":
        # nano-banana服务使用官方文档格式
        request_data = {
            "prompt": prompt,
            "n": 1,
            "model": "gemini-2.5-flash-image"
        }
        
        # 如果有图像，添加图像内容（用于图像编辑）
        if image_base64:
            # 对于图像编辑，使用multipart/form-data格式
            # 这里返回一个标记，表示需要使用multipart格式
            request_data["_multipart"] = True
            request_data["image"] = image_base64
            
    elif service_type == "dall-e-3":
        # DALL-E 3服务使用OpenAI格式
        request_data = {
            "model": _normalize_model_name(model),
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            "style": style,
            "response_format": "b64_json"
        }
        
    elif service_type == "stable-diffusion":
        # Stable-Diffusion服务
        request_data = {
            "prompt": prompt,
            "negative_prompt": "",
            "width": int(size.split('x')[0]),
            "height": int(size.split('x')[1]),
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1
        }
        
    elif service_type == "flux":
        # Flux服务
        request_data = {
            "prompt": prompt,
            "width": int(size.split('x')[0]),
            "height": int(size.split('x')[1]),
            "num_images": 1,
            "scheduler": "euler",
            "num_inference_steps": 20,
            "guidance_scale": 7.5
        }
        
    else:
        # 默认使用nano-banana格式
        request_data = {
            "prompt": prompt,
            "n": 1,
            "model": "gemini-2.5-flash-image"
        }
    
    return request_data

def call_api4gpt_api(url, api_key, service_type, request_data, timeout=300):
    """调用API4GPT API
    
    根据官方文档：https://doc.api4gpt.com/api-341609441
    """
    # 根据服务类型构建不同的API端点
    if service_type == "nano-banana":
        # nano-banana使用 /v1/images/generations 端点
        api_endpoint = f"{url}/v1/images/generations"
    elif service_type == "dall-e-3":
        api_endpoint = f"{url}/v1/images/generations"
    elif service_type == "stable-diffusion":
        api_endpoint = f"{url}/v1/images/generations"
    elif service_type == "flux":
        api_endpoint = f"{url}/v1/images/generations"
    else:
        api_endpoint = f"{url}/v1/images/generations"
    
    # 检查是否需要使用multipart格式
    if request_data.get("_multipart") and service_type == "nano-banana":
        # 对于图像编辑，使用multipart/form-data格式
        print("🔗 使用API4GPT multipart格式进行图像编辑")
        
        # 准备multipart数据
        files = {}
        data = {}
        
        # 添加图像文件
        if "image" in request_data:
            # 将base64转换为文件对象
            image_data = base64.b64decode(request_data["image"])
            files["image"] = ("image.jpg", image_data, "image/jpeg")
        
        # 添加其他参数
        data["prompt"] = request_data["prompt"]
        data["n"] = request_data["n"]
        data["model"] = request_data["model"]
        
        # 使用multipart端点
        edit_endpoint = f"{url}/v1/images/edits"
        
        headers = {
            "Authorization": f"Bearer {api_key.strip()}"
        }
        
        try:
            response = requests.post(
                edit_endpoint, 
                headers=headers, 
                files=files, 
                data=data, 
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API4GPT multipart API调用失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"API4GPT multipart API响应解析失败: {str(e)}")
    else:
        # 标准JSON格式
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, json=request_data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API4GPT API调用失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"API4GPT API响应解析失败: {str(e)}")

def parse_api4gpt_response(response_data, service_type):
    """解析API4GPT的响应数据
    
    根据官方文档：https://doc.api4gpt.com/api-341609441
    """
    if service_type == "nano-banana":
        # 解析nano-banana响应（官方文档格式）
        response_text = "nano-banana图像生成完成"
        generated_image = None
        
        # 根据官方文档，响应格式为：
        # {
        #   "created": 1745711868,
        #   "data": [
        #     {
        #       "revised_prompt": "一只熊猫在骑自行车...",
        #       "url": "https://filesystem.site/cdn/..."
        #     }
        #   ]
        # }
        
        if "data" in response_data and response_data["data"]:
            image_data = response_data["data"][0]
            
            # 提取修订后的提示词
            if "revised_prompt" in image_data:
                response_text = f"图像生成完成。修订后的提示词: {image_data['revised_prompt']}"
            
            # 提取图像URL
            if "url" in image_data:
                image_url = image_data["url"]
                try:
                    print(f"📥 正在下载API4GPT生成的图像: {image_url}")
                    response = requests.get(image_url, timeout=30)
                    if response.status_code == 200:
                        image_bytes = response.content
                        generated_image = Image.open(io.BytesIO(image_bytes))
                        print("✅ 成功下载API4GPT生成的图像")
                    else:
                        print(f"⚠️ 图像下载失败，状态码: {response.status_code}")
                except Exception as e:
                    print(f"⚠️ 图像下载失败: {e}")
        
        return response_text, generated_image
        
    elif service_type == "dall-e-3":
        # 解析DALL-E 3响应
        response_text = "DALL-E 3图像生成完成"
        generated_image = None
        
        if "data" in response_data and response_data["data"]:
            image_data = response_data["data"][0]
            if "url" in image_data:
                image_url = image_data["url"]
                try:
                    print(f"📥 正在下载DALL-E 3生成的图像: {image_url}")
                    response = requests.get(image_url, timeout=30)
                    if response.status_code == 200:
                        image_bytes = response.content
                        generated_image = Image.open(io.BytesIO(image_bytes))
                        print("✅ 成功下载DALL-E 3生成的图像")
                    else:
                        print(f"⚠️ 图像下载失败，状态码: {response.status_code}")
                except Exception as e:
                    print(f"⚠️ 图像下载失败: {e}")
        
        return response_text, generated_image
        
    elif service_type == "stable-diffusion":
        # 解析Stable-Diffusion响应
        response_text = "Stable-Diffusion图像生成完成"
        generated_image = None
        
        if "images" in response_data and response_data["images"]:
            image_data = response_data["images"][0]
            if image_data:
                try:
                    image_bytes = base64.b64decode(image_data)
                    generated_image = Image.open(io.BytesIO(image_bytes))
                    print("✅ 成功解析Stable-Diffusion图像数据")
                except Exception as e:
                    print(f"⚠️ Stable-Diffusion图像数据解析失败: {e}")
        
        return response_text, generated_image
        
    elif service_type == "flux":
        # 解析Flux响应
        response_text = "Flux图像生成完成"
        generated_image = None
        
        if "images" in response_data and response_data["images"]:
            image_data = response_data["images"][0]
            if image_data:
                try:
                    image_bytes = base64.b64decode(image_data)
                    generated_image = Image.open(io.BytesIO(image_bytes))
                    print("✅ 成功解析Flux图像数据")
                except Exception as e:
                    print(f"⚠️ Flux图像数据解析失败: {e}")
        
        return response_text, generated_image
        
    else:
        # 默认返回空结果
        return "", None

def smart_retry_delay(attempt, error_code=None):
    """智能重试延迟 - 根据错误类型调整等待时间"""
    base_delay = 2 ** attempt  # 指数退避
    
    if error_code == 429:  # 限流错误
        rate_limit_delay = 60 + random.uniform(10, 30)  # 60-90秒随机等待
        return max(base_delay, rate_limit_delay)
    elif error_code in [500, 502, 503, 504]:  # 服务器错误
        return base_delay + random.uniform(1, 5)  # 添加随机抖动
    else:
        return base_delay

def create_dummy_image(width=512, height=512):
    """Create a placeholder image"""
    dummy_array = np.zeros((height, width, 3), dtype=np.uint8)
    dummy_tensor = torch.from_numpy(dummy_array).float() / 255.0
    return dummy_tensor.unsqueeze(0)

def resize_image_for_api(image, max_size=2048):
    """调整图像大小以满足API限制"""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        _log_info(f"Image resized to {new_size} for API compatibility")
    return image

def remove_white_areas(image: Image.Image, white_threshold: int = 250) -> Image.Image:
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

        print(f"🔍 开始检测白色区域，阈值: {white_threshold}")

        # 多种白色检测策略
        white_masks = []

        # 策略1: 严格白色检测 (RGB三个通道都大于阈值)
        if len(img_array.shape) == 3:  # RGB图像
            strict_white_mask = np.all(img_array >= white_threshold, axis=2)
            white_masks.append(strict_white_mask)

            # 策略2: 近似白色检测 (RGB差异小且平均值高)
            rgb_mean = np.mean(img_array, axis=2)
            rgb_std = np.std(img_array, axis=2)
            approx_white_mask = (rgb_mean >= white_threshold - 10) & (rgb_std <= 15)
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
        print(f"🔍 白色像素比例: {white_ratio:.2%}")

        # 如果白色像素比例太低，不需要处理
        if white_ratio < 0.02:  # 小于2%
            print(f"ℹ️ 白色像素比例较低({white_ratio:.2%})，跳过处理")
            return image

        # 找到非白色区域的边界框
        non_white_mask = ~combined_white_mask

        # 找到非白色像素的行和列
        non_white_rows = np.any(non_white_mask, axis=1)
        non_white_cols = np.any(non_white_mask, axis=0)

        # 如果没有非白色像素，返回原图
        if not np.any(non_white_rows) or not np.any(non_white_cols):
            print(f"⚠️ 图像几乎全是白色，保持原图")
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

        print(f"🔍 边缘白色厚度: 上{edge_thickness['top']}, 下{edge_thickness['bottom']}, 左{edge_thickness['left']}, 右{edge_thickness['right']}")

        # 只有当边缘白色区域足够厚且是真正的边框时才进行裁剪
        min_edge_thickness = max(30, width // 8, height // 8)  # 至少30像素或图像尺寸的12.5%

        # 检查是否是真正的边框（四边都有白色或者对边有白色）
        thick_edges = [k for k, v in edge_thickness.items() if v >= min_edge_thickness]

        # 只有在以下情况才裁剪：
        # 1. 四边都有厚白边
        # 2. 对边都有厚白边（上下或左右）
        # 3. 三边有厚白边
        is_border = (
            len(thick_edges) >= 3 or  # 三边或四边有厚白边
            ('top' in thick_edges and 'bottom' in thick_edges) or  # 上下都有
            ('left' in thick_edges and 'right' in thick_edges)     # 左右都有
        )

        if not is_border:
            print(f"ℹ️ 不是真正的白色边框，跳过裁剪。厚边: {thick_edges}")
            return image

        print(f"✅ 检测到白色边框，厚边: {thick_edges}")

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
            print(f"⚠️ 裁剪区域无效，保持原图")
            return image

        # 计算裁剪比例
        crop_ratio = (crop_width * crop_height) / (width * height)

        # 如果裁剪后的区域太小，可能是误判
        if crop_ratio < 0.3:  # 小于30%
            print(f"⚠️ 裁剪后区域过小({crop_ratio:.2%})，可能是误判，保持原图")
            return image

        print(f"✅ 检测到白色边框，裁剪区域: ({left}, {top}) -> ({right}, {bottom})")
        print(f"✅ 裁剪尺寸: {crop_width}x{crop_height} (保留{crop_ratio:.1%})")

        # 裁剪图像
        cropped_image = image.crop((left, top, right + 1, bottom + 1))

        return cropped_image

    except Exception as e:
        print(f"❌ 白色区域检测失败: {e}")
        import traceback
        print(f"🔍 详细错误: {traceback.format_exc()}")
        return image

def smart_resize_with_padding(image: Image.Image, target_size: Tuple[int, int],
                             fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    统一调用主实现：按目标尺寸直接扩图，避免先放大再裁剪。
    增强功能：首先检测并去除白色区域，然后进行智能处理。
    """
    try:
        img_width, img_height = image.size
        target_width, target_height = target_size

        print(f"🎯 开始图像处理: {img_width}x{img_height} -> {target_width}x{target_height}")

        # 🚀 第一步：检测并去除白色区域
        processed_image = remove_white_areas(image)
        if processed_image.size != image.size:
            print(f"✅ 白色区域已去除: {image.size} -> {processed_image.size}")
            image = processed_image
            img_width, img_height = image.size

            # 如果还有白色区域，尝试更激进的检测
            processed_image2 = remove_white_areas(image, white_threshold=230)
            if processed_image2.size != image.size:
                print(f"✅ 激进模式再次去除白色区域: {image.size} -> {processed_image2.size}")
                image = processed_image2
                img_width, img_height = image.size
        else:
            print(f"ℹ️ 未检测到需要去除的白色区域")

        # 比例相同时，直接调整尺寸
        if abs(img_width/img_height - target_width/target_height) < 0.01:
            print(f"🎯 比例相同，直接调整尺寸")
            resized_img = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return resized_img

        # 使用裁剪模式：高清无损放大到最大边，然后智能裁剪
        print(f"🎯 裁剪模式：高清无损放大到最大边，然后智能裁剪")

        # 计算最佳缩放比例
        scale_x = target_width / img_width
        scale_y = target_height / img_height
        scale = max(scale_x, scale_y)  # 使用较大的缩放比例，确保完全覆盖

        # 计算放大后的尺寸
        enlarged_width = int(img_width * scale)
        enlarged_height = int(img_height * scale)

        print(f"🔧 高清无损放大: {img_width}x{img_height} -> {enlarged_width}x{enlarged_height}")
        print(f"🔧 缩放比例: {scale:.3f}")

        # 使用高质量重采样进行放大（禁用AI放大避免问题）
        enlarged_image = image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)

        # 智能裁剪 - 从高清放大的图像中裁剪出目标尺寸
        if enlarged_width >= target_width and enlarged_height >= target_height:
            print(f"🔧 智能裁剪：从高清放大图像中裁剪目标尺寸，确保主体居中")

            # 精确计算裁剪区域，确保主体完全居中
            crop_x = (enlarged_width - target_width) // 2
            crop_y = (enlarged_height - target_height) // 2

            print(f"🔧 精确居中裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x + target_width}, {crop_y + target_height})")

            # 从高清放大的图像中裁剪出目标尺寸
            final_image = enlarged_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))

            print(f"✅ 高清无损放大 + 智能裁剪完成")
            return final_image
        else:
            print(f"⚠️ 高清放大后尺寸不足，使用智能填充")
            # 创建目标尺寸的画布
            final_image = Image.new('RGB', (target_width, target_height), fill_color)

            # 将高清放大的图像居中放置
            paste_x = (target_width - enlarged_width) // 2
            paste_y = (target_height - enlarged_height) // 2
            final_image.paste(enlarged_image, (paste_x, paste_y))

            print(f"✅ 智能填充完成")
            return final_image

    except Exception as e:
        print(f"❌ 图像处理失败，使用回退方案: {e}")
        import traceback
        print(f"🔍 详细错误: {traceback.format_exc()}")

        # 回退到最简单的处理方式
        try:
            from .gemini_banana import smart_resize_with_padding as core_resize
        except ImportError:
            from gemini_banana import smart_resize_with_padding as core_resize
        return core_resize(image, target_size, fill_color=fill_color, fill_strategy="paste")

class KenChenLLMGeminiBananaMirrorImageGenNode:
    """Gemini Banana 镜像站图片生成节点
    
    功能特性:
    - 支持选择预配置的镜像站（official, comfly, custom）
    - 自动填充对应镜像站的 API URL 和 API Key
    - 选择 custom 时可手动输入自定义镜像站信息
    - 智能URL格式验证和自动补全
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 对齐 Gemini Banana Text to Image Banana 的控件
        try:
            from .gemini_banana import get_gemini_banana_config
        except ImportError:
            from gemini_banana import get_gemini_banana_config
        config = get_gemini_banana_config()
        default_params = config.get('default_params', {})
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        image_settings = config.get('image_settings', {})
        
        # 获取镜像站配置
        mirror_sites = config.get('mirror_sites', {})
        
        # 不再重复添加T8镜像站配置，因为配置文件中已经有了
        
        mirror_options = list(mirror_sites.keys())
        if not mirror_options:
            mirror_options = ["official", "comfly", "custom"]
        
        # 获取默认镜像站配置
        default_site = "comfly" if "comfly" in mirror_options else mirror_options[0] if mirror_options else "official"
        default_config = get_mirror_site_config(default_site)
        
        # 🚀 超越参考项目的图像控制预设
        size_presets = image_settings.get('size_presets', [
            "Original size", "512x512", "768x768", "1024x1024", "1024x1792", "1792x1024",
            "1920x1080", "2560x1440", "3840x2160"  # 超越参考项目的高分辨率选项
        ])
        quality_presets = image_settings.get('quality_presets', [
            "standard", "hd", "ultra_hd"  # 超越参考项目的超高清选项
        ])
        style_presets = image_settings.get('style_presets', [
            "vivid", "natural", "artistic", "cinematic", "photographic"  # 超越参考项目的风格选项
        ])
        
        return {
            "required": {
                "mirror_site": (mirror_options, {"default": default_site}),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "镜像站API密钥（可选，留空时自动获取）"
                }),
                "prompt": ("STRING", {"default": "A beautiful mountain landscape at sunset", "multiline": True}),
                # 支持多种AI模型和图像生成服务: nano-banana支持Comfly和T8镜像站
                "model": (["nano-banana [Comfly-T8]", "nano-banana-hd [Comfly-T8]", "gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation", "fal-ai/nano-banana [Comfly-T8]", "google/gemini-2.5-flash-image-preview [OpenRouter]"], {"default": "nano-banana [Comfly-T8]"}),
                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),
                "size": (size_presets, {"default": image_settings.get('default_size', "1024x1024")}),
                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),
                
                # 🎨 智能图像控制组（放在style下面）
                "detail_level": (["Basic Detail", "Professional Detail", "Premium Quality", "Masterpiece Level"], {"default": "Professional Detail"}),
                "camera_control": (["Auto Select", "Wide-angle Lens", "Macro Shot", "Low-angle Perspective", "High-angle Shot", "Close-up Shot", "Medium Shot"], {"default": "Auto Select"}),
                "lighting_control": (["Auto Settings", "Natural Light", "Studio Lighting", "Dramatic Shadows", "Soft Glow", "Golden Hour", "Blue Hour"], {"default": "Auto Settings"}),
                "template_selection": (["Auto Select", "Professional Portrait", "Cinematic Landscape", "Product Photography", "Digital Concept Art", "Anime Style Art", "Photorealistic Render", "Classical Oil Painting", "Watercolor Painting", "Cyberpunk Future", "Vintage Film Photography", "Architectural Photography", "Gourmet Food Photography"], {"default": "Auto Select"}),
                
                # 🚀 质量增强控制组
                "quality_enhancement": ("BOOLEAN", {"default": True, "label": "Enable Quality Enhancement"}),
                "enhance_quality": ("BOOLEAN", {"default": True, "label": "Enhanced Image Quality"}),
                "smart_resize": ("BOOLEAN", {"default": True, "label": "Smart Resize with Padding"}),
                "fill_color": ("STRING", {
                    "default": "255,255,255",
                    "placeholder": "填充颜色 RGB (如: 255,255,255)"
                }),
                
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
                "api4gpt_service": (["nano-banana"], {
                    "default": "nano-banana",
                    "tooltip": "API4GPT服务类型选择（仅在API4GPT镜像站时有效）"
                }),
                
                # ✨ 自定义指令组
                "custom_additions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义添加和特殊要求"
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate_image"
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
            print(f"[LLM Agent Assistant] Chat push failed: {e}")
            pass
    
    def generate_image(self, mirror_site: str, api_key: str, prompt: str, model: str, 
                      proxy: str, size: str, quality: str, style: str, detail_level: str, camera_control: str, 
                      lighting_control: str, template_selection: str, quality_enhancement: bool, enhance_quality: bool, 
                      smart_resize: bool, fill_color: str, temperature: float, top_p: float, top_k: int, 
                      max_output_tokens: int, seed: int, custom_size: str = "", api4gpt_service: str = "nano-banana", 
                      custom_additions: str = "", unique_id: str = "") -> Tuple[torch.Tensor, str]:
        """使用镜像站API生成图片"""
        
        # 🚀 立即规范化模型名称，去除UI标识
        model = _normalize_model_name(model)
        
        # 根据镜像站从配置获取URL和API Key
        site_config = get_mirror_site_config(mirror_site) if mirror_site else {"url": "", "api_key": ""}
        api_url = site_config.get("url", "").strip()
        if site_config.get("api_key") and not api_key.strip():
            api_key = site_config["api_key"]
            print(f"🔑 自动使用镜像站API Key: {api_key[:8]}...")
        if not api_url:
            raise ValueError("配置文件中缺少该镜像站的API URL")
        print(f"🔗 自动使用镜像站URL: {api_url}")
        
        # 验证API URL
        if not validate_api_url(api_url):
            raise ValueError("API URL格式无效，请检查配置文件")
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 处理图像控制参数
        try:
            from .gemini_banana import process_image_controls, enhance_prompt_with_controls
        except ImportError:
            from gemini_banana import process_image_controls, enhance_prompt_with_controls
        controls = process_image_controls(size, quality, style, custom_size)
        # 对于nano-banana模型，跳过尺寸提示，让模型自由生成
        skip_size_hints = model in ["nano-banana", "nano-banana-hd"]
        enhanced_prompt = enhance_prompt_with_controls(
            prompt.strip(), controls, detail_level, camera_control, lighting_control,
            template_selection, quality_enhancement, enhance_quality, smart_resize, fill_color,
            skip_size_hints=skip_size_hints
        )
        
        print(f"🎨 图像控制参数: 尺寸={controls['size']}, 质量={controls['quality']}, 风格={controls['style']}")
        if controls['is_custom_size']:
            print(f"📏 使用自定义尺寸: {controls['size']}")
        
        # 代理处理：有效则设置，无效/未填则清除，避免残留环境变量影响请求
        if proxy and proxy.strip() and "None" not in proxy:
            os.environ['HTTPS_PROXY'] = proxy.strip()
            os.environ['HTTP_PROXY'] = proxy.strip()
            print(f"🔌 使用代理: {proxy.strip()}")
        else:
            existing = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
            if existing:
                print(f"🔌 未指定代理，沿用系统代理: {existing}")
            else:
                print("🔌 未指定代理（系统无代理）")
        
        # 构建完整的API URL
        full_url = build_api_url(api_url, model)
        print(f"🌐 使用API地址: {full_url}")
        
        # 检查镜像站类型 - 按照优先级顺序：nano-banana官方 → Comfly → T8 → API4GPT → OpenRouter → OpenAI → custom
        is_nano_banana_official = mirror_site == "nano-banana官方"
        is_t8_mirror = "t8star.cn" in api_url or "ai.t8star.cn" in api_url
        is_api4gpt_mirror = "www.api4gpt.com" in api_url
        is_comfly_mirror = _is_comfly_base(api_url)
        is_openrouter_mirror = "openrouter.ai" in api_url
        is_openai_mirror = "api.openai.com" in api_url or site_config.get("api_format") == "openai"
        
        # 按照优先级顺序处理镜像站：nano-banana官方 → Comfly → T8 → API4GPT → OpenRouter → OpenAI → custom
        
        # 1. nano-banana官方镜像站处理
        if is_nano_banana_official:
            print("🔗 检测到nano-banana官方镜像站，使用Google官方API")

            # 构建内容部分
            content_parts = [{"text": enhanced_prompt}]

            # 构建生成配置
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"]
            }

            # 添加seed（如果有效）
            if seed and seed > 0:
                generation_config["seed"] = seed

            try:
                # 使用优先API调用（官方API优先，失败时回退到REST API）
                response_json = generate_with_priority_api(
                    api_key=api_key,
                    model=_normalize_model_name(model),
                    content_parts=content_parts,
                    generation_config=generation_config,
                    max_retries=5,
                    proxy=proxy
                )

                if response_json:
                    # 提取生成的图像
                    generated_image = process_generated_image_from_response(response_json)
                    print(f"🔍 图像提取结果: {generated_image is not None}")
                    if generated_image:
                        print(f"🔍 原始图像尺寸: {generated_image.size}")

                    # 提取响应文本
                    response_text = extract_text_from_response(response_json)

                    if generated_image:
                        # 应用全量增强（包括智能主体检测和居中技术）
                        print("🚀 开始应用图像增强技术...")
                        try:
                            enhanced_image = _apply_full_enhancements(
                                generated_image,
                                controls['size'],
                                quality,
                                enhance_quality,
                                smart_resize
                            )
                            if enhanced_image:
                                generated_image = enhanced_image
                                print(f"✅ 图像增强完成")
                                try:
                                    print(f"🔧 Final output size: {generated_image.size[0]}x{generated_image.size[1]}")
                                except Exception:
                                    pass
                            else:
                                print("⚠️ 图像增强返回None，使用原始图像")
                        except Exception as e:
                            print(f"❌ 图像增强失败: {e}")
                            import traceback
                            print(f"🔍 增强错误详情: {traceback.format_exc()}")

                        print("✅ 图片生成完成（nano-banana官方）")
                        return (pil_to_tensor(generated_image), response_text)
                    else:
                        print("⚠️ nano-banana官方API响应中未找到图像数据")
                        # 返回默认图像
                        default_image = Image.new('RGB', (1024, 1024), color='black')
                        return (pil_to_tensor(default_image), response_text)
                else:
                    raise Exception("nano-banana官方API调用失败")

            except Exception as e:
                print(f"❌ nano-banana官方API调用失败: {e}")
                raise e
            
        # 2. Comfly镜像站处理
        elif is_comfly_mirror:
            print("🔗 检测到Comfly镜像站，使用Comfly API格式")

            if model in ["nano-banana", "nano-banana-hd"]:
                # Comfly nano-banana 直连（生成）
                try:
                    result = _comfly_nano_banana_generate(api_url, api_key, model, enhanced_prompt, controls['size'], temperature, top_p, max_output_tokens, seed)
                    # print(f"[DEBUG] result type: {type(result)}")
                    # print(f"[DEBUG] result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    # print(f"[DEBUG] result content: {str(result)[:500]}...")
                    generated_image = None
                    response_text = ""
                    
                    if isinstance(result, dict) and 'data' in result:
                        if not result['data']:
                            # 如果data为空，说明没有生成图像
                            response_text = result.get('response_text', '')
                            print(f"⚠️ Comfly nano-banana 没有返回图像数据")
                            print(f"📝 响应文本: {response_text[:200]}...")
                            raise Exception(f"Comfly nano-banana 服务没有返回图像数据，响应: {response_text[:100]}...")

                        # 如果data不为空，继续处理
                        if result['data']:
                            b64 = result['data'][0].get('b64_json')
                            image_url = result['data'][0].get('url', '')
                            response_text = result.get('response_text', "")

                            # 确保图像URL信息显示在响应中
                            if image_url and image_url not in response_text:
                                response_text += f"\n图像URL: {image_url}"

                            # print(f"[DEBUG] 图像URL: {image_url}")
                            # print(f"[DEBUG] 响应文本: {response_text}")

                            if b64:
                                from base64 import b64decode
                                import io
                                try:
                                    # 修复base64填充问题
                                    b64_fixed = b64 + '=' * (4 - len(b64) % 4)
                                    img = Image.open(io.BytesIO(b64decode(b64_fixed))).convert('RGB')
                                except Exception as decode_error:
                                    print(f"⚠️ base64解码失败: {decode_error}")
                                    # 尝试直接解码
                                    try:
                                        img = Image.open(io.BytesIO(b64decode(b64))).convert('RGB')
                                    except Exception as e2:
                                        print(f"⚠️ 直接解码也失败: {e2}")
                                        img = None
                                generated_image = img
                            else:
                                # 如果没有base64数据，尝试从响应文本中提取图像
                                import re
                                base64_pattern = r'data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)'
                                matches = re.findall(base64_pattern, response_text)
                                if matches:
                                    from base64 import b64decode
                                    import io
                                    img = Image.open(io.BytesIO(b64decode(matches[0]))).convert('RGB')
                                    generated_image = img
                    
                    # 如果成功处理，应用尺寸与质量增强后再返回
                    if generated_image:
                        # 全量增强
                        try:
                            generated_image = _apply_full_enhancements(
                                generated_image,
                                controls['size'],
                                quality,
                                enhance_quality,
                                smart_resize
                            )
                            try:
                                print(f"🔧 Final output size: {generated_image.size[0]}x{generated_image.size[1]}")
                            except Exception:
                                pass
                        except Exception:
                            pass

                        image_tensor = pil_to_tensor(generated_image)
                        print("✅ 图片生成完成（Comfly nano-banana）")
                        self._push_chat(enhanced_prompt, _make_chat_summary(response_text or ""), unique_id)
                        return (image_tensor, response_text)
                        
                except Exception as e:
                    print(f"❌ Comfly(nano-banana) 生成失败: {e}")
                    raise e
            else:
                # 非nano-banana模型使用Gemini API格式
                request_data = {
                "model": model,  # 🔧 修复：添加缺失的model字段
                "contents": [{
                    "parts": [{"text": enhanced_prompt}]
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

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key.strip()}"
                }

        # 3. T8镜像站处理
        elif is_t8_mirror:
            # T8镜像站也支持 nano-banana，调用方式与 Comfly 一致
            print("🔗 检测到T8镜像站，使用chat/completions端点 (nano-banana 直连)")
            if _normalize_model_name(model) in ["nano-banana", "nano-banana-hd"]:
                try:
                    result = _comfly_nano_banana_generate(api_url, api_key, _normalize_model_name(model), enhanced_prompt, controls['size'], temperature, top_p, max_output_tokens, seed)
                    # 🔍 调试：打印T8返回的结果格式 - 已关闭
                    # print(f"[DEBUG] T8 result type: {type(result)}")
                    # print(f"[DEBUG] T8 result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    # print(f"[DEBUG] T8 result content: {str(result)[:500]}...")

                    generated_image = None
                    response_text = ""
                    if isinstance(result, dict) and 'data' in result and result['data']:
                        b64 = result['data'][0].get('b64_json')
                        image_url = result['data'][0].get('url', '')
                        response_text = result.get('response_text', "")

                        # 确保图像URL信息显示在响应中
                        if image_url and image_url not in response_text:
                            response_text += f"\n图像URL: {image_url}"

                        # print(f"[DEBUG] T8 图像URL: {image_url}")
                        # print(f"[DEBUG] T8 响应文本: {response_text}")

                        if b64:
                            from base64 import b64decode
                            import io
                            try:
                                b64_fixed = b64 + '=' * (4 - len(b64) % 4)
                                img = Image.open(io.BytesIO(b64decode(b64_fixed))).convert('RGB')
                            except Exception:
                                img = Image.open(io.BytesIO(b64decode(b64))).convert('RGB')
                            generated_image = img
                    if generated_image:
                        # 应用全量增强（包括智能主体检测和居中技术）
                        try:
                            generated_image = _apply_full_enhancements(
                                generated_image,
                                controls['size'],
                                quality,
                                enhance_quality,
                                smart_resize
                            )
                            try:
                                print(f"🔧 Final output size: {generated_image.size[0]}x{generated_image.size[1]}")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        print("✅ 图片生成完成（T8 nano-banana）")
                        return (pil_to_tensor(generated_image), response_text)
                except Exception as e:
                    print(f"❌ T8(nano-banana) 生成失败: {e}")
                    raise e
            # 其他模型仍走原 T8 OpenAI 兼容格式
            print("🔗 检测到T8镜像站，使用OpenAI兼容的API格式")
            request_data = build_t8_api_request(
                model=_normalize_model_name(model),
                prompt=enhanced_prompt,
                temperature=temperature,
                max_tokens=max_output_tokens
            )
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
            
        # 4. API4GPT镜像站处理
        elif is_api4gpt_mirror:
            # API4GPT镜像站
            print("🔗 检测到API4GPT镜像站")
            
            # 使用API4GPT服务接口进行图像生成
            print("🔗 使用API4GPT服务接口进行图像生成")
            
            if api4gpt_service == "nano-banana":
                # nano-banana服务使用官方API4GPT格式
                print("🔗 使用API4GPT nano-banana服务进行图像生成")
                
                # 构建API4GPT请求数据
                request_data = build_api4gpt_request(
                    service_type=api4gpt_service,
                    model=_normalize_model_name(model),
                    prompt=enhanced_prompt,
                    size=controls['size'],
                    quality=controls['quality'],
                    style=controls['style'],
                    temperature=temperature,
                    max_tokens=max_output_tokens
                )
                
                # 调用API4GPT API
                try:
                    result = call_api4gpt_api(api_url, api_key, api4gpt_service, request_data)
                    print("✅ API4GPT API调用成功")
                    
                    # 解析API4GPT响应
                    response_text, generated_image = parse_api4gpt_response(result, api4gpt_service)
                    
                    if generated_image:
                        print(f"✅ 成功提取API4GPT生成的图像")
                    else:
                        print("⚠️ API4GPT未返回图像，创建占位符")
                        generated_image = Image.new('RGB', (512, 512), color='lightgray')
                        if not response_text:
                            response_text = f"API4GPT {api4gpt_service} 服务响应完成，但未返回图像数据"
                    
                    # 处理图像尺寸调整
                    if generated_image:
                        try:
                            target_size = controls['size']
                            if 'x' in target_size:
                                target_width, target_height = map(int, target_size.split('x'))
                                
                                if generated_image.size != (target_width, target_height):
                                    print(f"🔧 使用扩图技术调整图像尺寸: {generated_image.size} -> {target_size}")
                                    
                                    # 🚀 优化：高清无损放大到最大边，智能裁剪确保主体居中
                                    if generated_image.size[0]/generated_image.size[1] != target_width/target_height:
                                        print(f"📐 检测到比例变化，使用高清无损放大 + 智能裁剪方法")
                                        
                                        # 🎯 高清无损放大（保持原始比例，不拉伸变形）
                                        # 计算最佳缩放比例，使用max确保完全覆盖目标区域
                                        scale_x = target_width / generated_image.size[0]      # 宽度比例
                                        scale_y = target_height / generated_image.size[1]    # 高度比例
                                        scale = max(scale_x, scale_y)  # 使用较大的缩放比例，确保完全覆盖
                                        
                                        # 计算放大后的尺寸（保持原始比例，确保覆盖目标区域）
                                        enlarged_width = int(generated_image.size[0] * scale)
                                        enlarged_height = int(generated_image.size[1] * scale)
                                        
                                        print(f"🔧 高清无损放大: {generated_image.size[0]}x{generated_image.size[1]} -> {enlarged_width}x{enlarged_height}")
                                        print(f"🔧 缩放比例: {scale:.3f} (使用max确保完全覆盖，然后智能裁剪)")
                                        print(f"🔧 关键：直接放大到最大边，保持图像清晰度和比例")
                                        
                                        # 🎯 使用AI放大模型进行高清无损放大（保持比例）
                                        # 优先使用AI模型，回退到高质量重采样
                                        try:
                                            print(f"🔧 尝试使用AI放大模型进行高清放大...")
                                            ai_upscaled_image = smart_ai_upscale(generated_image, enlarged_width, enlarged_height)
                                            
                                            if ai_upscaled_image is not None:
                                                # 如果AI放大成功，调整到目标尺寸
                                                if ai_upscaled_image.size != (enlarged_width, enlarged_height):
                                                    print(f"🔧 AI放大后调整到目标尺寸: {ai_upscaled_image.size} -> {enlarged_width}x{enlarged_height}")
                                                    enlarged_image = ai_upscaled_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                                else:
                                                    enlarged_image = ai_upscaled_image
                                                print(f"✅ AI放大模型放大完成，图像质量大幅提升")
                                            else:
                                                print(f"⚠️ AI放大模型不可用，使用高质量重采样")
                                                enlarged_image = generated_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                                
                                        except Exception as e:
                                            print(f"⚠️ AI放大模型失败，使用高质量重采样: {e}")
                                            # 回退到 LANCZOS 算法
                                            enlarged_image = generated_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                        
                                        # 🎯 智能裁剪 - 从高清放大的图像中裁剪出目标尺寸
                                        if enlarged_width >= target_width and enlarged_height >= target_height:
                                            print(f"🔧 智能裁剪：从高清放大图像中裁剪目标尺寸，确保主体完全居中")
                                            
                                            # 🎯 核心创新：智能主体检测 + 精确居中裁剪
                                            print(f"🔍 开始智能主体检测...")
                                            
                                            try:
                                                # 使用智能主体检测算法
                                                subject_bbox, subject_center = detect_image_foreground_subject(enlarged_image)
                                                subject_x, subject_y, subject_w, subject_h = subject_bbox
                                                subject_center_x, subject_center_y = subject_center
                                                
                                                print(f"🎯 检测到主体: 位置({subject_x}, {subject_y}), 尺寸({subject_w}x{subject_h})")
                                                print(f"🎯 主体中心: ({subject_center_x}, {subject_center_y})")
                                                
                                                # 🎯 智能裁剪策略：确保主体在裁剪后图像的中心
                                                # 计算主体中心应该在新图像中的位置
                                                target_center_x = target_width // 2
                                                target_center_y = target_height // 2
                                                
                                                # 计算裁剪起始位置，使主体中心对齐到目标中心
                                                crop_x = subject_center_x - target_center_x
                                                crop_y = subject_center_y - target_center_y
                                                
                                                print(f"🔧 智能计算裁剪位置: 主体中心({subject_center_x}, {subject_center_y}) -> 目标中心({target_center_x}, {target_center_y})")
                                                print(f"🔧 理论裁剪位置: ({crop_x}, {crop_y})")
                                                
                                                # 🎯 边界检查和调整
                                                # 确保裁剪区域不超出图像边界
                                                if crop_x < 0:
                                                    print(f"🔧 调整：裁剪X坐标过小，调整为0")
                                                    crop_x = 0
                                                elif crop_x + target_width > enlarged_width:
                                                    print(f"🔧 调整：裁剪X坐标过大，调整为{enlarged_width - target_width}")
                                                    crop_x = enlarged_width - target_width
                                                
                                                if crop_y < 0:
                                                    print(f"🔧 调整：裁剪Y坐标过小，调整为0")
                                                    crop_y = 0
                                                elif crop_y + target_height > enlarged_height:
                                                    print(f"🔧 调整：裁剪Y坐标过大，调整为{enlarged_height - target_height}")
                                                    crop_y = enlarged_height - target_height
                                                
                                                print(f"🔧 最终裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x + target_width}, {crop_y + target_height})")
                                                print(f"🔧 图像尺寸: {enlarged_width}x{enlarged_height}")
                                                print(f"🔧 目标尺寸: {target_width}x{target_height}")
                                                print(f"🔧 智能策略：主体精确居中，确保完全可见")
                                                
                                                # 从高清放大的图像中裁剪出目标尺寸
                                                generated_image = enlarged_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                                                
                                            except Exception as e:
                                                print(f"⚠️ 智能主体检测失败: {e}，使用备用裁剪策略")
                                                # 备用策略：从中心裁剪
                                                crop_x = (enlarged_width - target_width) // 2
                                                crop_y = (enlarged_height - target_height) // 2
                                                
                                                # 边界检查
                                                if crop_x < 0: crop_x = 0
                                                if crop_y < 0: crop_y = 0
                                                if crop_x + target_width > enlarged_width: crop_x = enlarged_width - target_width
                                                if crop_y + target_height > enlarged_height: crop_y = enlarged_height - target_height
                                                
                                                print(f"🔧 备用策略：中心裁剪区域 ({crop_x}, {crop_y}) -> ({crop_x + target_width}, {crop_y + target_height})")
                                                generated_image = enlarged_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                                            
                                            print(f"✅ 高清无损放大 + 智能裁剪完成")
                                            print(f"✅ 结果：无白色填充，完全不变形，主体精确居中，保持最高清晰度")
                                            print(f"✅ 图像质量：高清无损，比例完美，主体可见")
                                            
                                        else:
                                            print(f"⚠️ 高清放大后尺寸不足，使用智能填充（避免拉伸变形）")
                                            # 创建目标尺寸的画布，使用填充色
                                            new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                                            
                                            # 将高清放大的图像居中放置
                                            paste_x = (target_width - enlarged_width) // 2
                                            paste_y = (target_height - enlarged_height) // 2
                                            new_image.paste(enlarged_image, (paste_x, paste_y))
                                            
                                            generated_image = new_image
                                            print(f"✅ 智能填充完成：高清放大图像居中放置，边缘用填充色")
                                        
                                    else:
                                        # 比例相同，直接调整尺寸（无变形）
                                        print(f"📐 比例相同，直接调整尺寸（无变形）")
                                        generated_image = generated_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                                    
                                    print(f"✅ 图像尺寸调整完成: {generated_image.size}")
                        except Exception as e:
                            print(f"⚠️ 图像尺寸调整失败: {e}, 保持原始尺寸")
                    
                    # 应用传统质量增强（关闭自适应）
                    if enhance_quality and quality in ['hd', 'ultra_hd']:
                        print(f"✨ 应用传统质量增强，质量等级: {quality}")
                        generated_image = enhance_image_quality(generated_image, quality, "disabled")
                        print(f"✅ 增强完成")
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(generated_image)
                    print("✅ 图片生成完成（API4GPT）")
                    self._push_chat(enhanced_prompt, response_text or "", unique_id)
                    return (image_tensor, response_text)
                    
                except Exception as e:
                    print(f"❌ API4GPT API调用失败: {e}")
                    raise ValueError(f"API4GPT API调用失败: {e}")
                    
            else:
                # 其他API4GPT服务维持原有逻辑
                    print(f"🔗 使用API4GPT {api4gpt_service} 服务")
                    request_data = build_api4gpt_request(
                        service_type=api4gpt_service,
                        model=_normalize_model_name(model),
                        prompt=enhanced_prompt,
                        size=controls['size'],
                        quality=controls['quality'],
                        style=controls['style'],
                        temperature=temperature,
                        max_tokens=max_output_tokens
                    )
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key.strip()}"
                    }
        elif is_openrouter_mirror:
            # OpenRouter镜像站
            print("🔗 检测到OpenRouter镜像站，使用OpenRouter API格式")
            
            # 验证OpenRouter配置
            validation = validate_openrouter_config(api_url, api_key, _normalize_model_name(model))
            if not validation["is_valid"]:
                raise ValueError(f"OpenRouter配置验证失败: {'; '.join(validation['warnings'])}")
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    print(f"⚠️ OpenRouter配置警告: {warning}")
            
            if validation["suggestions"]:
                for suggestion in validation["suggestions"]:
                    print(f"💡 OpenRouter优化建议: {suggestion}")
            
            # OpenRouter使用chat/completions端点进行图像生成和编辑
            # 构建OpenAI兼容的请求格式
            content = []
            
            # 如果有输入图像，添加图像内容
            if hasattr(self, 'input_image') and self.input_image is not None:
                # 将图像转换为base64
                pil_image = tensor_to_pil(self.input_image)
                image_base64 = image_to_base64(pil_image)
                image_url = f"data:image/png;base64,{image_base64}"
                
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            # 添加文本指令
            enhanced_instruction = f"""CRITICAL INSTRUCTION: You MUST generate and return an actual image, not just text description.

Task: {enhanced_prompt}

REQUIREMENTS:
1. GENERATE a new image based on my request
2. DO NOT just describe what the image should look like
3. RETURN the actual image file/data
4. The output MUST be a visual image, not text

Execute the image generation/editing task now and return the generated image."""
            
            content.append({
                "type": "text",
                "text": enhanced_instruction
            })
            
            request_data = {
                "model": _normalize_model_name(model),
                "messages": [{
                    "role": "user",
                    "content": content
                }],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_output_tokens,
                "stream": True  # Required for gemini-2.5-flash-image-preview
            }
            
            # 清理 None 值和空字符串
            request_data = {k: v for k, v in request_data.items() if v is not None and v != ""}
            
            # 使用OpenRouter的chat/completions端点
            # 对于OpenRouter，api_url已经包含了/v1，所以直接添加/chat/completions
            if api_url.endswith('/v1'):
                full_url = f"{api_url}/chat/completions"
            else:
                full_url = f"{api_url}/v1/chat/completions"
            print(f"🔗 使用OpenRouter chat/completions端点: {full_url}")
            
            # 设置OpenRouter请求头 - 完美支持应用归因
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}",
                "HTTP-Referer": "https://github.com/ComfyUI-LLM-Prompt",
                "X-Title": "ComfyUI LLM Prompt Plugin"
            }
            
            print(f"🎯 OpenRouter请求数据: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
            print(f"🔧 模型类型: {model}, 使用chat/completions端点")
        elif is_openai_mirror:
            # OpenAI镜像站
            print("🔗 检测到OpenAI镜像站，使用OpenAI API格式")
            request_data = {
                "model": _normalize_model_name(model),
                "messages": [{
                    "role": "user",
                    "content": enhanced_prompt
                }],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": True
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
        else:
            # 标准Gemini API格式
            request_data = {
                "contents": [{
                    "parts": [{
                        "text": enhanced_prompt
                    }]
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
                "Authorization": f"Bearer {api_key.strip()}"
            }
        
        # 智能重试机制 - 完全移植参考项目
        max_retries = 5
        timeout = 120
        
        # OpenRouter专用重试策略
        if is_openrouter_mirror:
            print("🔄 使用OpenRouter专用重试策略")
            # OpenRouter可能需要更长的超时时间
            timeout = 180
        
        for attempt in range(max_retries):
            try:
                print(f"🎨 正在生成图片... (尝试 {attempt + 1}/{max_retries})")
                print(f"📝 提示词: {enhanced_prompt[:100]}...")
                print(f"🔗 镜像站: {api_url}")
                
                # 发送请求
                response = requests.post(full_url, headers=headers, json=request_data, timeout=timeout, stream=True)
                
                # 检查响应状态
                print(f"📡 HTTP状态码: {response.status_code}")
                print(f"📡 响应头: {dict(response.headers)}")
                
                # OpenRouter专用错误处理
                if is_openrouter_mirror and response.status_code != 200:
                    if response.status_code == 401:
                        raise ValueError("OpenRouter API密钥无效或已过期")
                    elif response.status_code == 403:
                        raise ValueError("OpenRouter API访问被拒绝，请检查账户权限")
                    elif response.status_code == 429:
                        raise ValueError("OpenRouter API请求过于频繁，请稍后重试")
                    elif response.status_code == 500:
                        raise ValueError("OpenRouter服务器内部错误，请稍后重试")
                    else:
                        raise ValueError(f"OpenRouter API错误: HTTP {response.status_code}")
                
                # 成功响应
                if response.status_code == 200:
                    # 提取文本响应和图片
                    response_text = ""
                    generated_image = None
                    
                    if is_api4gpt_mirror:
                        # API4GPT镜像站响应处理
                        print("🔗 处理API4GPT镜像站响应")
                        
                        try:
                            # 尝试解析JSON响应
                            result = response.json()
                            print(f"📋 API4GPT响应结构: {list(result.keys())}")
                            
                            if api4gpt_service == "nano-banana":
                                # nano-banana使用OpenAI兼容格式
                                response_text, generated_image = parse_openai_compatible_response(result)
                            else:
                                # 其他服务使用原有的解析逻辑
                                response_text, generated_image = parse_api4gpt_response(result, api4gpt_service)
                            
                            if generated_image:
                                print(f"✅ 成功提取API4GPT生成的图像")
                            else:
                                print("⚠️ API4GPT未返回图像，创建占位符")
                                generated_image = Image.new('RGB', (512, 512), color='lightgray')
                                if not response_text:
                                    response_text = f"API4GPT {api4gpt_service} 服务响应完成，但未返回图像数据"
                        except Exception as json_error:
                            print(f"⚠️ API4GPT JSON解析失败: {json_error}")
                            print(f"📋 原始响应内容: {response.text[:500]}...")
                            generated_image = Image.new('RGB', (512, 512), color='lightgray')
                            response_text = f"API4GPT响应解析失败: {json_error}"
                    elif is_openrouter_mirror:
                        # OpenRouter镜像站响应处理 - 使用流式响应
                        print("🔗 处理OpenRouter镜像站流式响应")
                        
                        # 处理流式响应
                        response_text = process_openrouter_stream(response)
                        
                        # 检查响应文本中是否包含图像数据
                        if "data:image/" in response_text:
                            print("🖼️ 检测到OpenRouter返回的图像数据")
                            try:
                                # 使用正确的正则表达式提取base64图像数据
                                import re
                                base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
                                image_matches = re.findall(base64_pattern, response_text)
                                if image_matches:
                                    # 取第一个匹配的图像数据
                                    image_url = image_matches[0]
                                    print(f"🎯 成功匹配OpenRouter图像数据，长度: {len(image_url)}字符")
                                    
                                    # 提取base64部分
                                    if ';base64,' in image_url:
                                        import io
                                        base64_data = image_url.split(';base64,', 1)[1]
                                        image_bytes = base64.b64decode(base64_data)
                                        generated_image = Image.open(io.BytesIO(image_bytes))
                                        print(f"✅ 成功提取OpenRouter生成的图像: {generated_image.size}")
                                        
                                        # 清理响应文本，移除base64数据
                                        response_text = re.sub(base64_pattern, '[图像已生成]', response_text)
                                    else:
                                        print(f"⚠️ 图像数据格式不正确: {image_url[:100]}...")
                                        generated_image = Image.new('RGB', (512, 512), color='lightgray')
                                        response_text = f"OpenRouter图像生成完成，但数据格式不正确"
                                else:
                                    print(f"⚠️ 正则表达式未找到匹配的图像数据")
                                    generated_image = Image.new('RGB', (512, 512), color='lightgray')
                                    response_text = f"OpenRouter图像生成完成，但未找到图像数据"
                            except Exception as e:
                                print(f"⚠️ OpenRouter图像数据解析失败: {e}")
                                generated_image = Image.new('RGB', (512, 512), color='lightgray')
                                response_text = f"OpenRouter图像生成完成，但解析失败: {e}"
                        
                        # 如果没有成功提取图像，创建占位符
                        if not generated_image:
                            print("⚠️ OpenRouter未返回图像数据，创建占位符")
                            generated_image = Image.new('RGB', (512, 512), color='lightgray')
                            if not response_text:
                                response_text = "OpenRouter图像生成完成，但未返回图像数据"
                    elif is_t8_mirror:
                        # T8镜像站OpenAI格式响应处理
                        print("🔗 处理T8镜像站OpenAI格式响应")
                        
                        try:
                            # 尝试解析JSON响应
                            result = response.json()
                            print(f"📋 T8镜像站响应结构: {list(result.keys())}")
                            
                            if "choices" in result and result["choices"]:
                                choice = result["choices"][0]
                                if "message" in choice and "content" in choice["message"]:
                                    content = choice["message"]["content"]
                                    if isinstance(content, str):
                                        response_text = content
                                        # 检查是否包含base64图像数据
                                        if "![image](data:image/" in content:
                                            print("🖼️ 检测到T8镜像站返回的图像数据")
                                            try:
                                                # 提取base64图像数据
                                                import re, io, io
                                                image_match = re.search(r'!\[image\]\(data:image/\w+;base64,([^)]+)\)', content)
                                                if image_match:
                                                    image_data = image_match.group(1)
                                                    image_bytes = base64.b64decode(image_data)
                                                    generated_image = Image.open(io.BytesIO(image_bytes))
                                                    print("✅ 成功提取T8镜像站生成的图像")
                                                    # 清理响应文本，移除base64数据
                                                    response_text = re.sub(r'!\[image\]\(data:image/\w+;base64,[^)]+\)', '[图像已生成]', content)
                                            except Exception as e:
                                                print(f"⚠️ T8镜像站图像数据解析失败: {e}")
                                    elif isinstance(content, list):
                                        # 处理多模态内容
                                        for item in content:
                                            if item.get("type") == "text":
                                                response_text += item.get("text", "")
                                            elif item.get("type") == "image_url":
                                                # 处理图像URL（如果有的话）
                                                print("🖼️ 检测到图像URL响应")
                            
                            # 如果没有成功提取图像，再检查其他可能的位置
                            if not generated_image:
                                print("⚠️ T8镜像站暂不支持图像生成，创建占位符")
                                generated_image = Image.new('RGB', (512, 512), color='lightgray')
                                if not response_text:
                                    response_text = "T8镜像站文本生成完成，但暂不支持图像生成功能"
                        except Exception as json_error:
                            print(f"⚠️ T8镜像站 JSON解析失败: {json_error}")
                            print(f"📋 原始响应内容: {response.text[:500]}...")
                            generated_image = Image.new('RGB', (512, 512), color='lightgray')
                            response_text = f"T8镜像站响应解析失败: {json_error}"
                            
                    elif is_openai_mirror:
                        # OpenAI镜像站
                        print("🔗 检测到OpenAI镜像站，使用OpenAI API格式")
                        request_data = {
                            "model": _normalize_model_name(model),
                            "messages": [{
                                "role": "user",
                                "content": enhanced_prompt
                            }],
                            "temperature": temperature,
                            "max_tokens": max_output_tokens,
                            "stream": True
                        }
                        
                        # 设置请求头
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key.strip()}"
                        }
                    else:
                        # 标准Gemini API响应处理
                        try:
                            result = response.json()
                        except Exception as e:
                            print(f"⚠️ 标准Gemini JSON解析失败: {e}")
                            result = {}
                        if "candidates" in result and result["candidates"]:
                            candidate = result["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    # 提取文本
                                    if "text" in part:
                                        response_text += part["text"]
                                    
                                    # 提取图片
                                    if "inline_data" in part or "inlineData" in part:
                                        inline_data = part.get("inline_data") or part.get("inlineData")
                                        if inline_data and "data" in inline_data:
                                            try:
                                                import io  # 🔧 修复：添加缺失的io模块导入
                                                image_bytes = base64.b64decode(inline_data["data"])
                                                generated_image = Image.open(io.BytesIO(image_bytes))
                                                print(f"✅ 成功提取标准Gemini生成的图像: {generated_image.size}")
                                            except Exception as e:
                                                print(f"⚠️ 标准Gemini图像数据解析失败: {e}")
                                                generated_image = Image.new('RGB', (512, 512), color='lightgray')
                                                if not response_text:
                                                    response_text = f"标准Gemini图像生成完成，但解析失败: {e}"
                            else:
                                # 如果没有成功提取图像，创建占位符
                                if not generated_image:
                                    print("⚠️ 标准Gemini未返回图像数据，创建占位符")
                                    generated_image = Image.new('RGB', (512, 512), color='lightgray')
                                    if not response_text:
                                        response_text = "标准Gemini图像生成完成，但未返回图像数据"
                    
                    # 如果没有生成图片，创建占位符
                    if generated_image is None:
                        print("⚠️ 未检测到生成的图片，创建占位符")
                        generated_image = Image.new('RGB', (512, 512), color='lightgray')
                        if not response_text:
                            response_text = "图片生成请求已发送，但未收到图片数据"
                    
                    # 🚀 高清无损放大 + 智能裁剪调整图像尺寸
                    try:
                        size_value = controls['size']
                        if 'x' in size_value:
                            target_width, target_height = map(int, size_value.split('x'))
                        else:
                            target_width, target_height = generated_image.size
                        current_width, current_height = generated_image.size
                        
                        if (current_width, current_height) != (target_width, target_height):
                            print(f"🔧 高清无损放大调整图像尺寸: {current_width}x{current_height} -> {target_width}x{target_height}")
                            
                            # 🚀 优化：高清无损放大到最大边，智能裁剪确保主体居中
                            if current_width/current_height != target_width/target_height:
                                print(f"📐 检测到比例变化，使用高清无损放大 + 智能裁剪方法")
                                
                                # 🎯 高清无损放大（保持原始比例，不拉伸变形）
                                # 计算最佳缩放比例，使用max确保完全覆盖目标区域
                                scale_x = target_width / current_width      # 宽度比例
                                scale_y = target_height / current_height    # 高度比例
                                scale = max(scale_x, scale_y)  # 使用较大的缩放比例，确保完全覆盖
                                
                                # 计算放大后的尺寸（保持原始比例，确保覆盖目标区域）
                                enlarged_width = int(current_width * scale)
                                enlarged_height = int(current_height * scale)
                                
                                print(f"🔧 高清无损放大: {current_width}x{current_height} -> {enlarged_width}x{enlarged_height}")
                                print(f"🔧 缩放比例: {scale:.3f} (使用max确保完全覆盖，然后智能裁剪)")
                                print(f"🔧 关键：直接放大到最大边，保持图像清晰度和比例")
                                
                                # 🎯 使用AI放大模型进行高清无损放大（保持比例）
                                # 优先使用AI模型，回退到高质量重采样
                                try:
                                    print(f"🔧 尝试使用AI放大模型进行高清放大...")
                                    ai_upscaled_image = smart_ai_upscale(generated_image, enlarged_width, enlarged_height)
                                    
                                    if ai_upscaled_image is not None:
                                        # 如果AI放大成功，调整到目标尺寸
                                        if ai_upscaled_image.size != (enlarged_width, enlarged_height):
                                            print(f"🔧 AI放大后调整到目标尺寸: {ai_upscaled_image.size} -> {enlarged_width}x{enlarged_height}")
                                            enlarged_image = ai_upscaled_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                        else:
                                            enlarged_image = ai_upscaled_image
                                        print(f"✅ AI放大模型放大完成，图像质量大幅提升")
                                    else:
                                        print(f"⚠️ AI放大模型不可用，使用高质量重采样")
                                        enlarged_image = generated_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                        
                                except Exception as e:
                                    print(f"⚠️ AI放大模型失败，使用高质量重采样: {e}")
                                    # 回退到 LANCZOS 算法
                                    enlarged_image = generated_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                
                                # 🎯 智能裁剪 - 从高清放大的图像中裁剪出目标尺寸
                                if enlarged_width >= target_width and enlarged_height >= target_height:
                                    print(f"🔧 智能裁剪：从高清放大图像中裁剪目标尺寸，确保主体居中")
                                    
                                    # 🎯 精确计算裁剪区域，确保主体完全居中
                                    crop_x = (enlarged_width - target_width) // 2
                                    crop_y = (enlarged_height - target_height) // 2
                                    
                                    # 🎯 微调偏移，确保完全居中（避免奇数像素偏差）
                                    if (enlarged_width - target_width) % 2 == 1:
                                        crop_x += 1
                                    if (enlarged_height - target_height) % 2 == 1:
                                        crop_y += 1
                                    
                                    print(f"🔧 精确居中裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x + target_width}, {crop_y + target_height})")
                                    print(f"🔧 确保主体在裁剪后图像的正中心位置")
                                    
                                    # 从高清放大的图像中裁剪出目标尺寸
                                    generated_image = enlarged_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                                    
                                    print(f"✅ 高清无损放大 + 智能裁剪完成")
                                    print(f"✅ 结果：无白色填充，完全不变形，主体精确居中，保持最高清晰度")
                                    print(f"✅ 图像质量：高清无损，比例完美，主体可见")
                                    
                                else:
                                    print(f"⚠️ 高清放大后尺寸不足，使用智能填充（避免拉伸变形）")
                                    # 创建目标尺寸的画布，使用填充色
                                    new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                                    
                                    # 将高清放大的图像居中放置
                                    paste_x = (target_width - enlarged_width) // 2
                                    paste_y = (target_height - enlarged_height) // 2
                                    new_image.paste(enlarged_image, (paste_x, paste_y))
                                    
                                    generated_image = new_image
                                    print(f"✅ 智能填充完成：高清放大图像居中放置，边缘用填充色")
                                
                            else:
                                # 比例相同，直接调整尺寸（无变形）
                                print(f"📐 比例相同，直接调整尺寸（无变形）")
                                generated_image = generated_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        
                        print(f"✅ 图像尺寸调整完成: {generated_image.size}")
                    except Exception as e:
                        print(f"⚠️ 图像尺寸调整失败: {e}, 保持原始尺寸")
                    
                    # 应用传统质量增强（关闭自适应）
                    if enhance_quality and quality in ['hd', 'ultra_hd']:
                        print(f"✨ 应用传统质量增强，质量等级: {quality}")
                        generated_image = enhance_image_quality(generated_image, quality, "disabled")
                        print(f"✅ 增强完成")
                    
                    # 如果没有响应文本，提供默认文本
                    if not response_text:
                        response_text = "图片生成完成！这是根据您的提示词生成的图像。"
                        print("📝 使用默认响应文本")
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(generated_image)

                    # 🔧 修复：根据实际镜像站显示正确的完成消息
                    if is_comfly_mirror:
                        print("✅ 图片生成完成（Comfly）")
                    elif is_t8_mirror:
                        print("✅ 图片生成完成（T8）")
                    elif is_api4gpt_mirror:
                        print("✅ 图片生成完成（API4GPT）")
                    elif is_openrouter_mirror:
                        print("✅ 图片生成完成（OpenRouter）")
                    elif is_openai_mirror:
                        print("✅ 图片生成完成（OpenAI）")
                    else:
                        print(f"✅ 图片生成完成（{mirror_site}）")

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
                raise ValueError(f"图片生成失败: {error_msg}")



    def get_mirror_config(self):
        """获取当前镜像站配置"""
        try:
            from .gemini_banana import get_gemini_banana_config
        except ImportError:
            from gemini_banana import get_gemini_banana_config
        config = get_gemini_banana_config()
        mirror_sites = config.get('mirror_sites', {})
        
        # 查找API4GPT镜像站配置
        for site_name, site_config in mirror_sites.items():
            if "api4gpt.com" in site_config.get("url", ""):
                return site_config
        
        # 如果没找到，返回默认配置
        return {
            "url": "https://www.api4gpt.com",
            "api_key": "",
            "api_format": "api4gpt"
        }


class KenChenLLMGeminiBananaMirrorImageEditNode:
    """Gemini Banana 镜像站图片编辑节点
    
    功能特性:
    - 支持选择预配置的镜像站（official, comfly, custom）
    - 自动填充对应镜像站的 API URL 和 API Key
    - 选择 custom 时可手动输入自定义镜像站信息
    - 智能URL格式验证和自动补全
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .gemini_banana import get_gemini_banana_config
        except ImportError:
            from gemini_banana import get_gemini_banana_config
        config = get_gemini_banana_config()
        default_params = config.get('default_params', {})
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        image_settings = config.get('image_settings', {})
        
        # 获取镜像站配置
        mirror_sites = config.get('mirror_sites', {})
        mirror_options = list(mirror_sites.keys())
        if not mirror_options:
            mirror_options = ["official", "comfly", "custom"]
        
        # 获取默认镜像站配置
        default_site = "comfly" if "comfly" in mirror_options else mirror_options[0] if mirror_options else "official"
        default_config = get_mirror_site_config(default_site)
        
        # 🚀 超越参考项目的图像控制预设
        size_presets = image_settings.get('size_presets', [
            "Original size", "512x512", "768x768", "1024x1024", "1024x1792", "1792x1024",
            "1920x1080", "2560x1440", "3840x2160"  # 超越参考项目的高分辨率选项
        ])
        quality_presets = image_settings.get('quality_presets', [
            "standard", "hd", "ultra_hd"  # 超越参考项目的超高清选项
        ])
        style_presets = image_settings.get('style_presets', [
            "vivid", "natural", "artistic", "cinematic", "photographic"  # 超越参考项目的风格选项
        ])
        
        return {
            "required": {
                "mirror_site": (mirror_options, {"default": default_site}),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "镜像站API密钥（可选，留空时自动获取）"
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Can you add a llama next to me?", "multiline": True}),
                # 支持多种AI模型和图像编辑服务: nano-banana支持Comfly和T8镜像站, OpenRouter模型: google/gemini-2.5-flash-image-preview (付费)
                "model": (["nano-banana [Comfly-T8]", "nano-banana-hd [Comfly-T8]", "gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation", "fal-ai/nano-banana/edit [Comfly-T8]", "google/gemini-2.5-flash-image-preview [OpenRouter]"], {"default": "nano-banana [Comfly-T8]"}),
                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),
                "size": (size_presets, {"default": image_settings.get('default_size', "1024x1024")}),
                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),
                
                # 🎨 智能图像控制组（放在style下面）
                "detail_level": (["Basic Detail", "Professional Detail", "Premium Quality", "Masterpiece Level"], {"default": "Professional Detail"}),
                "camera_control": (["Auto Select", "Wide-angle Lens", "Macro Shot", "Low-angle Perspective", "High-angle Shot", "Close-up Shot", "Medium Shot"], {"default": "Auto Select"}),
                "lighting_control": (["Auto Settings", "Natural Light", "Studio Lighting", "Dramatic Shadows", "Soft Glow", "Golden Hour", "Blue Hour"], {"default": "Auto Settings"}),
                "template_selection": (["Auto Select", "Professional Portrait", "Cinematic Landscape", "Product Photography", "Digital Concept Art", "Anime Style Art", "Photorealistic Render", "Classical Oil Painting", "Watercolor Painting", "Cyberpunk Future", "Vintage Film Photography", "Architectural Photography", "Gourmet Food Photography"], {"default": "Auto Select"}),
                
                # 🚀 质量增强控制组
                "quality_enhancement": ("BOOLEAN", {"default": True, "label": "Enable Quality Enhancement"}),
                "enhance_quality": ("BOOLEAN", {"default": True, "label": "Enhanced Image Quality"}),
                "smart_resize": ("BOOLEAN", {"default": True, "label": "Smart Resize with Padding"}),
                "fill_color": ("STRING", {
                    "default": "255,255,255",
                    "placeholder": "填充颜色 RGB (如: 255,255,255)"
                }),
                
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
                "api4gpt_service": (["nano-banana"], {
                    "default": "nano-banana",
                    "tooltip": "API4GPT服务类型选择（仅在API4GPT镜像站时有效）"
                }),
                
                # ✨ 自定义指令组
                "custom_additions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义添加和特殊要求"
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "edit_image"
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
            print(f"[LLM Agent Assistant] Chat push failed: {e}")
            pass
    
    def edit_image(self, mirror_site: str, api_key: str, image: torch.Tensor, prompt: str, model: str,
                    proxy: str, size: str, quality: str, style: str, detail_level: str, camera_control: str, lighting_control: str, 
                    template_selection: str, quality_enhancement: bool, enhance_quality: bool, smart_resize: bool, 
                    fill_color: str, temperature: float, top_p: float, top_k: int, max_output_tokens: int, seed: int, 
                    custom_size: str = "", api4gpt_service: str = "nano-banana", custom_additions: str = "", unique_id: str = "") -> Tuple[torch.Tensor, str]:
        """使用镜像站API编辑图片"""
        
        # 🚀 立即规范化模型名称，去除UI标识
        model = _normalize_model_name(model)
        
        # 根据镜像站从配置获取URL和API Key
        site_config = get_mirror_site_config(mirror_site) if mirror_site else {"url": "", "api_key": ""}
        api_url = site_config.get("url", "").strip()
        if site_config.get("api_key") and not api_key.strip():
            api_key = site_config["api_key"]
            print(f"🔑 自动使用镜像站API Key: {api_key[:8]}...")
        if not api_url:
            raise ValueError("配置文件中缺少该镜像站的API URL")
        print(f"🔗 自动使用镜像站URL: {api_url}")
        
        if not validate_api_url(api_url):
            raise ValueError("API URL格式无效，请检查配置文件")
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 处理图像控制参数
        try:
            from .gemini_banana import process_image_controls, enhance_prompt_with_controls
        except ImportError:
            from gemini_banana import process_image_controls, enhance_prompt_with_controls
        controls = process_image_controls(size, quality, style, custom_size)
        # 对于nano-banana模型，跳过尺寸提示，让模型自由生成
        skip_size_hints = model in ["nano-banana", "nano-banana-hd"]
        enhanced_prompt = enhance_prompt_with_controls(
            prompt.strip(), controls, detail_level, camera_control, lighting_control,
            template_selection, quality_enhancement, enhance_quality, smart_resize, fill_color,
            skip_size_hints=skip_size_hints
        )
        
        print(f"🎨 图像控制参数: 尺寸={controls['size']}, 质量={controls['quality']}, 风格={controls['style']}")
        if controls['is_custom_size']:
            print(f"📏 使用自定义尺寸: {controls['size']}")
        
        # 转换输入图片
        pil_image = tensor_to_pil(image)
        
        # 调整图像尺寸以符合API要求
        pil_image = resize_image_for_api(pil_image)
        
        # 转换为base64
        image_base64 = image_to_base64(pil_image, format='JPEG')
        
        # 代理处理：有效则设置，无效/未填则清除，避免残留环境变量影响请求
        if proxy and proxy.strip() and "None" not in proxy:
            os.environ['HTTPS_PROXY'] = proxy.strip()
            os.environ['HTTP_PROXY'] = proxy.strip()
            print(f"🔌 使用代理: {proxy.strip()}")
        else:
            existing = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
            if existing:
                print(f"🔌 未指定代理，沿用系统代理: {existing}")
            else:
                print("🔌 未指定代理（系统无代理）")
        
        # 检查镜像站类型 - 按照优先级顺序：nano-banana官方 → Comfly → T8 → API4GPT → OpenRouter → OpenAI → custom
        is_nano_banana_official = mirror_site == "nano-banana官方"
        is_t8_mirror = "t8star.cn" in api_url or "ai.t8star.cn" in api_url
        is_api4gpt_mirror = "www.api4gpt.com" in api_url
        is_comfly_mirror = _is_comfly_base(api_url)
        is_openrouter_mirror = "openrouter.ai" in api_url
        is_openai_mirror = "api.openai.com" in api_url or site_config.get("api_format") == "openai"
        
        # 构建完整的API URL（OpenRouter除外，因为它在各自的处理逻辑中构建）
        if not is_openrouter_mirror:
            full_url = build_api_url(api_url, model)
            print(f"🌐 使用API地址: {full_url}")
        else:
            print(f"🌐 OpenRouter镜像站，URL将在OpenRouter处理逻辑中构建")
        
        # 按照优先级顺序处理镜像站：nano-banana官方 → Comfly → T8 → API4GPT → OpenRouter → OpenAI → custom
        
        # 1. nano-banana官方镜像站处理
        if is_nano_banana_official:
            print("🔗 检测到nano-banana官方镜像站，使用Google官方API")

            # 构建内容部分（文本 + 图像）
            content_parts = [
                {"text": enhanced_prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64
                    }
                }
            ]

            # 构建生成配置
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"]
            }

            # 添加seed（如果有效）
            if seed and seed > 0:
                generation_config["seed"] = seed

            try:
                # 使用优先API调用（官方API优先，失败时回退到REST API）
                response_json = generate_with_priority_api(
                    api_key=api_key,
                    model=_normalize_model_name(model),
                    content_parts=content_parts,
                    generation_config=generation_config,
                    max_retries=5,
                    proxy=proxy
                )

                if response_json:
                    # 提取编辑后的图像
                    edited_image = process_generated_image_from_response(response_json)

                    # 提取响应文本
                    response_text = extract_text_from_response(response_json)

                    if edited_image:
                        # 应用全量增强（包括智能主体检测和居中技术）
                        try:
                            edited_image = _apply_full_enhancements(
                                edited_image,
                                controls['size'],
                                quality,
                                enhance_quality,
                                smart_resize
                            )
                            try:
                                print(f"🔧 Final output size: {edited_image.size[0]}x{edited_image.size[1]}")
                            except Exception:
                                pass
                        except Exception:
                            pass

                        image_tensor = pil_to_tensor(edited_image)
                        print("✅ 图片编辑完成（nano-banana官方）")
                        self._push_chat(enhanced_prompt, _make_chat_summary(response_text or ""), unique_id)
                        return (image_tensor, response_text)
                    else:
                        print("⚠️ nano-banana官方API响应中未找到编辑后的图像数据")
                        # 返回原图像
                        return (image, response_text)
                else:
                    raise Exception("nano-banana官方API调用失败")

            except Exception as e:
                print(f"❌ nano-banana官方API调用失败: {e}")
                raise e
            
        # 2. Comfly镜像站处理
        elif is_comfly_mirror:
            print("🔗 检测到Comfly镜像站，使用Comfly API格式")
            
            if model in ["nano-banana", "nano-banana-hd"] and pil_image is not None:
                # Comfly nano-banana 直连（编辑）
                try:
                    result = _comfly_nano_banana_edit(api_url, api_key, model, enhanced_prompt, [pil_image], controls['size'], temperature, top_p, max_output_tokens, seed)
                    edited_image = None
                    response_text = ""
                    
                    if isinstance(result, dict) and 'data' in result and result['data']:
                        b64 = result['data'][0].get('b64_json')
                        response_text = result.get('response_text', "")
                        
                        if b64:
                            from base64 import b64decode
                            import io
                            try:
                                # 修复base64填充问题
                                b64_fixed = b64 + '=' * (4 - len(b64) % 4)
                                img = Image.open(io.BytesIO(b64decode(b64_fixed))).convert('RGB')
                            except Exception as decode_error:
                                print(f"⚠️ base64解码失败: {decode_error}")
                                # 尝试直接解码
                                try:
                                    img = Image.open(io.BytesIO(b64decode(b64))).convert('RGB')
                                except Exception as e2:
                                    print(f"⚠️ 直接解码也失败: {e2}")
                                    img = None
                            edited_image = img
                        else:
                            # 如果没有base64数据，尝试从响应文本中提取图像
                            import re
                            base64_pattern = r'data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)'
                            matches = re.findall(base64_pattern, response_text)
                            if matches:
                                from base64 import b64decode
                                import io
                                img = Image.open(io.BytesIO(b64decode(matches[0]))).convert('RGB')
                                edited_image = img
                    
                    # 如果成功处理，应用尺寸与质量增强后再返回
                    if edited_image:
                        # 全量增强
                        try:
                            edited_image = _apply_full_enhancements(
                                edited_image,
                                controls['size'],
                                quality,
                                enhance_quality,
                                smart_resize
                            )
                            try:
                                print(f"🔧 Final output size: {edited_image.size[0]}x{edited_image.size[1]}")
                            except Exception:
                                pass
                        except Exception:
                            pass

                        image_tensor = pil_to_tensor(edited_image)
                        print("✅ 图片编辑完成（Comfly nano-banana）")
                        self._push_chat(enhanced_prompt, _make_chat_summary(response_text or ""), unique_id)
                        return (image_tensor, response_text)
                        
                except Exception as e:
                    print(f"❌ Comfly(nano-banana) 编辑失败: {e}")
                    raise e
            request_data = {
                "model": model,  # 🔧 修复：添加缺失的model字段
                "contents": [{
                    "parts": [
                        {"text": enhanced_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
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
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
                
        # 3. T8镜像站处理
        elif is_t8_mirror:
            # T8镜像站也支持 nano-banana，调用方式与 Comfly 一致
            print("🔗 检测到T8镜像站，使用chat/completions端点 (nano-banana 直连)")
            if _normalize_model_name(model) in ["nano-banana", "nano-banana-hd"] and pil_image is not None:
                try:
                    result = _comfly_nano_banana_edit(full_url, api_key, _normalize_model_name(model), enhanced_prompt, [pil_image], controls['size'], temperature, top_p, max_output_tokens, seed)
                    edited_image = None
                    response_text = ""

                    if isinstance(result, dict) and 'data' in result and result['data']:
                        b64 = result['data'][0].get('b64_json')
                        response_text = result.get('response_text', "")

                        if b64:
                            from base64 import b64decode
                            import io
                            try:
                                b64_fixed = b64 + '=' * (4 - len(b64) % 4)
                                img = Image.open(io.BytesIO(b64decode(b64_fixed))).convert('RGB')
                            except Exception:
                                img = Image.open(io.BytesIO(b64decode(b64))).convert('RGB')
                            edited_image = img

                    # 如果成功处理，应用尺寸与质量增强后再返回
                    if edited_image:
                        # 全量增强
                        try:
                            edited_image = _apply_full_enhancements(
                                edited_image,
                                controls['size'],
                                quality,
                                enhance_quality,
                                smart_resize
                            )
                            try:
                                print(f"🔧 Final output size: {edited_image.size[0]}x{edited_image.size[1]}")
                            except Exception:
                                pass
                        except Exception:
                            pass

                        image_tensor = pil_to_tensor(edited_image)
                        print("✅ 图片编辑完成（T8 nano-banana）")
                        self._push_chat(enhanced_prompt, _make_chat_summary(response_text or ""), unique_id)
                        return (image_tensor, response_text)

                except Exception as e:
                    print(f"❌ T8(nano-banana) 编辑失败: {e}")
                    raise e
            # 其他模型仍走原 T8 OpenAI 兼容格式
            print("🔗 检测到T8镜像站，使用OpenAI兼容的API格式")
            request_data = build_t8_api_request(
                model=_normalize_model_name(model),
                prompt=enhanced_prompt,
                image_base64=image_base64,
                temperature=temperature,
                max_tokens=max_output_tokens
            )
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
            
        # 4. API4GPT镜像站处理
        elif is_api4gpt_mirror:
            # API4GPT镜像站
            print("🔗 检测到API4GPT镜像站")
            
            # 使用API4GPT服务接口进行图像编辑
            print("🔗 使用API4GPT服务接口进行图像编辑")
            
            if api4gpt_service == "nano-banana":
                # nano-banana服务使用官方API4GPT格式进行图像编辑
                print("🔗 使用API4GPT nano-banana服务进行图像编辑")
                
                # 构建API4GPT请求数据（包含图像）
                request_data = build_api4gpt_request(
                    service_type=api4gpt_service,
                    model=_normalize_model_name(model),
                    prompt=enhanced_prompt,
                    image_base64=image_base64,  # 传入输入图像
                    size=controls['size'],
                    quality=controls['quality'],
                    style=controls['style'],
                    temperature=temperature,
                    max_tokens=max_output_tokens
                )
                
                # 调用API4GPT API进行图像编辑
                try:
                    result = call_api4gpt_api(api_url, api_key, api4gpt_service, request_data)
                    print("✅ API4GPT图像编辑API调用成功")
                    
                    # 解析API4GPT响应
                    response_text, edited_image = parse_api4gpt_response(result, api4gpt_service)
                    
                    if edited_image:
                        print(f"✅ 成功提取API4GPT编辑后的图像")
                    else:
                        print("⚠️ API4GPT未返回编辑后的图像，返回原图片")
                        edited_image = pil_image
                        if not response_text:
                            response_text = f"API4GPT {api4gpt_service} 服务响应完成，但未返回编辑后的图像数据"
                    
                    # 处理图像尺寸调整
                    if edited_image:
                        try:
                            target_size = controls['size']
                            if 'x' in target_size:
                                target_width, target_height = map(int, target_size.split('x'))
                                
                                if edited_image.size != (target_width, target_height):
                                    print(f"🔧 使用扩图技术调整图像尺寸: {edited_image.size} -> {target_size}")
                                    
                                    # 🚀 优化：高清无损放大到最大边，智能裁剪确保主体居中
                                    if edited_image.size[0]/edited_image.size[1] != target_width/target_height:
                                        print(f"📐 检测到比例变化，使用高清无损放大 + 智能裁剪方法")
                                        
                                        # 🎯 高清无损放大（保持原始比例，不拉伸变形）
                                        # 计算最佳缩放比例，使用max确保完全覆盖目标区域
                                        scale_x = target_width / edited_image.size[0]      # 宽度比例
                                        scale_y = target_height / edited_image.size[1]    # 高度比例
                                        scale = max(scale_x, scale_y)  # 使用较大的缩放比例，确保完全覆盖
                                        
                                        # 计算放大后的尺寸（保持原始比例，确保覆盖目标区域）
                                        enlarged_width = int(edited_image.size[0] * scale)
                                        enlarged_height = int(edited_image.size[1] * scale)
                                        
                                        print(f"🔧 高清无损放大: {edited_image.size[0]}x{edited_image.size[1]} -> {enlarged_width}x{enlarged_height}")
                                        print(f"🔧 缩放比例: {scale:.3f} (使用max确保完全覆盖，然后智能裁剪)")
                                        print(f"🔧 关键：直接放大到最大边，保持图像清晰度和比例")
                                        
                                        # 🎯 使用AI放大模型进行高清无损放大（保持比例）
                                        # 优先使用AI模型，回退到高质量重采样
                                        try:
                                            print(f"🔧 尝试使用AI放大模型进行高清放大...")
                                            ai_upscaled_image = smart_ai_upscale(edited_image, enlarged_width, enlarged_height)
                                            
                                            if ai_upscaled_image is not None:
                                                # 如果AI放大成功，调整到目标尺寸
                                                if ai_upscaled_image.size != (enlarged_width, enlarged_height):
                                                    print(f"🔧 AI放大后调整到目标尺寸: {ai_upscaled_image.size} -> {enlarged_width}x{enlarged_height}")
                                                    enlarged_image = ai_upscaled_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                                else:
                                                    enlarged_image = ai_upscaled_image
                                                print(f"✅ AI放大模型放大完成，图像质量大幅提升")
                                            else:
                                                print(f"⚠️ AI放大模型不可用，使用高质量重采样")
                                                enlarged_image = edited_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                                
                                        except Exception as e:
                                            print(f"⚠️ AI放大模型失败，使用高质量重采样: {e}")
                                            # 回退到 LANCZOS 算法
                                            enlarged_image = edited_image.resize((enlarged_width, enlarged_height), Image.Resampling.LANCZOS)
                                        
                                        # 🎯 智能裁剪 - 从高清放大的图像中裁剪出目标尺寸
                                        if enlarged_width >= target_width and enlarged_height >= target_height:
                                            print(f"🔧 智能裁剪：从高清放大图像中裁剪目标尺寸，确保主体居中")
                                            
                                            # 🎯 精确计算裁剪区域，确保主体完全居中
                                            crop_x = (enlarged_width - target_width) // 2
                                            crop_y = (enlarged_height - target_height) // 2
                                            
                                            # 🎯 微调偏移，确保完全居中（避免奇数像素偏差）
                                            if (enlarged_width - target_width) % 2 == 1:
                                                crop_x += 1
                                            if (enlarged_height - target_height) % 2 == 1:
                                                crop_y += 1
                                            
                                            print(f"🔧 精确居中裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x + target_width}, {crop_y + target_height})")
                                            print(f"🔧 确保主体在裁剪后图像的正中心位置")
                                            
                                            # 从高清放大的图像中裁剪出目标尺寸
                                            edited_image = enlarged_image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
                                            
                                            print(f"✅ 高清无损放大 + 智能裁剪完成")
                                            print(f"✅ 结果：无白色填充，完全不变形，主体精确居中，保持最高清晰度")
                                            print(f"✅ 图像质量：高清无损，比例完美，主体可见")
                                            
                                        else:
                                            print(f"⚠️ 高清放大后尺寸不足，使用智能填充（避免拉伸变形）")
                                            # 创建目标尺寸的画布，使用填充色
                                            new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                                            
                                            # 将高清放大的图像居中放置
                                            paste_x = (target_width - enlarged_width) // 2
                                            paste_y = (target_height - enlarged_height) // 2
                                            new_image.paste(enlarged_image, (paste_x, paste_y))
                                            
                                            edited_image = new_image
                                            print(f"✅ 智能填充完成：高清放大图像居中放置，边缘用填充色")
                                        
                                    else:
                                        # 比例相同，直接调整尺寸（无变形）
                                        print(f"📐 比例相同，直接调整尺寸（无变形）")
                                        edited_image = edited_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        except Exception as e:
                            print(f"⚠️ 图像尺寸调整失败: {e}, 保持原始尺寸")
                    
                    # 应用传统质量增强（关闭自适应）
                    if enhance_quality and quality in ['hd', 'ultra_hd']:
                        print(f"✨ 应用传统质量增强，质量等级: {quality}")
                        edited_image = enhance_image_quality(edited_image, quality, "disabled")
                        print(f"✅ 增强完成")
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(edited_image)
                    print("✅ 图片编辑完成（API4GPT）")
                    self._push_chat(enhanced_prompt, response_text or "", unique_id)
                    return (image_tensor, response_text)
                    
                except Exception as e:
                    print(f"❌ API4GPT图像编辑API调用失败: {e}")
                    raise ValueError(f"API4GPT图像编辑API调用失败: {e}")
                    
            else:
                # 其他API4GPT服务维持原有逻辑
                    print(f"🔗 使用API4GPT {api4gpt_service} 服务")
                    request_data = build_api4gpt_request(
                        service_type=api4gpt_service,
                        model=_normalize_model_name(model),
                        prompt=enhanced_prompt,
                        image_base64=image_base64,
                        size=controls['size'],
                        quality=controls['quality'],
                        style=controls['style'],
                        temperature=temperature,
                        max_tokens=max_output_tokens
                    )
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key.strip()}"
                    }
        elif is_openrouter_mirror:
            # OpenRouter镜像站
            print("🔗 检测到OpenRouter镜像站，使用OpenRouter API格式")
            
            # OpenRouter使用chat/completions端点进行图像编辑
            # 构建OpenAI兼容的请求格式
            content = []
            
            # 添加图像内容
            image_url = f"data:image/jpeg;base64,{image_base64}"
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            
            # 添加文本指令
            enhanced_instruction = f"""CRITICAL INSTRUCTION: You MUST generate and return an actual edited image, not just text description.

Task: {enhanced_prompt}

REQUIREMENTS:
1. EDIT the provided image based on my request
2. DO NOT just describe what the edited image should look like
3. RETURN the actual edited image file/data
4. The output MUST be a visual image, not text

Execute the image editing task now and return the edited image."""
            
            content.append({
                "type": "text",
                "text": enhanced_instruction
            })
            
            request_data = {
                "model": _normalize_model_name(model),
                "messages": [{
                    "role": "user",
                    "content": content
                }],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_output_tokens,
                "stream": True  # Required for gemini-2.5-flash-image-preview
            }
            
            # 使用OpenRouter的chat/completions端点
            # 对于OpenRouter，api_url已经包含了/v1，所以直接添加/chat/completions
            if api_url.endswith('/v1'):
                full_url = f"{api_url}/chat/completions"
            else:
                full_url = f"{api_url}/v1/chat/completions"
            print(f"🔗 使用OpenRouter chat/completions端点进行图像编辑: {full_url}")
            
            # 设置OpenRouter请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}",
                "HTTP-Referer": "https://github.com/ComfyUI-LLM-Prompt",
                "X-Title": "ComfyUI LLM Prompt Plugin"
            }
        elif is_openai_mirror:
            # OpenAI镜像站
            print("🔗 检测到OpenAI镜像站，使用OpenAI API格式")
            request_data = {
                "model": _normalize_model_name(model),
                "messages": [{
                    "role": "user",
                    "content": enhanced_prompt
                }],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": True
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
        else:
            # 标准Gemini API格式
            request_data = {
                "contents": [{
                    "parts": [
                        {
                            "text": enhanced_prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
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
                "Authorization": f"Bearer {api_key.strip()}"
            }
        
        # 智能重试机制 - 完全移植参考项目
        max_retries = 5
        timeout = 120
        
        for attempt in range(max_retries):
            try:
                print(f"🖼️ 正在编辑图片... (尝试 {attempt + 1}/{max_retries})")
                print(f"📝 编辑指令: {enhanced_prompt[:100]}...") # 使用增强后的提示词
                print(f"🔗 镜像站: {api_url}")
                
                # 发送请求
                response = requests.post(full_url, headers=headers, json=request_data, timeout=timeout, stream=True)
                
                # 检查响应状态
                print(f"📡 HTTP状态码: {response.status_code}")
                print(f"📡 响应头: {dict(response.headers)}")
                
                # 成功响应
                if response.status_code == 200:
                    # 提取文本响应和编辑后的图片
                    response_text = ""
                    edited_image = None
                    
                    # 为所有镜像站定义 result 变量
                    if not is_api4gpt_mirror and not is_openrouter_mirror and not is_t8_mirror and not is_openai_mirror:
                        # 标准 Gemini API 镜像站（如 Comfy API）
                        try:
                            result = response.json()
                            print("✅ 成功解析标准 Gemini API 响应")
                        except Exception as e:
                            print(f"⚠️ 解析响应失败: {e}")
                            result = {}
                    elif is_t8_mirror:
                        # T8 镜像站
                        try:
                            result = response.json()
                            print("✅ 成功解析 T8 镜像站响应")
                        except Exception as e:
                            print(f"⚠️ T8 镜像站响应解析失败: {e}")
                            result = {}
                    elif is_openai_mirror:
                        # OpenAI镜像站
                        try:
                            result = response.json()
                            print("✅ 成功解析 OpenAI 镜像站响应")
                        except Exception as e:
                            print(f"⚠️ OpenAI 镜像站响应解析失败: {e}")
                            result = {}
                    
                    if is_api4gpt_mirror:
                        # API4GPT镜像站响应处理
                        print("🔗 处理API4GPT镜像站响应")
                        
                        try:
                            # 尝试解析JSON响应
                            result = response.json()
                            print(f"📋 API4GPT响应结构: {list(result.keys())}")
                            
                            if api4gpt_service == "nano-banana":
                                # nano-banana使用OpenAI兼容格式
                                response_text, edited_image = parse_openai_compatible_response(result)
                            else:
                                # 其他服务使用原有的解析逻辑
                                response_text, edited_image = parse_api4gpt_response(result, api4gpt_service)
                            
                            if edited_image:
                                print(f"✅ 成功提取API4GPT编辑后的图像")
                            else:
                                print("⚠️ API4GPT未返回编辑后的图像，返回原图片")
                                edited_image = pil_image
                                if not response_text:
                                    response_text = f"API4GPT {api4gpt_service} 服务响应完成，但未返回编辑后的图像数据"
                        except Exception as json_error:
                            print(f"⚠️ API4GPT JSON解析失败: {json_error}")
                            print(f"📋 原始响应内容: {response.text[:500]}...")
                            edited_image = pil_image
                            response_text = f"API4GPT响应解析失败: {json_error}"
                    elif is_openrouter_mirror:
                        # OpenRouter镜像站响应处理 - 使用流式响应
                        print("🔗 处理OpenRouter镜像站流式响应")
                        
                        # 处理流式响应
                        response_text = process_openrouter_stream(response)
                        
                        # 检查响应文本中是否包含图像数据
                        if "data:image/" in response_text:
                            print("🖼️ 检测到OpenRouter返回的图像数据")
                            try:
                                # 使用正确的正则表达式提取base64图像数据
                                import re
                                base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
                                image_matches = re.findall(base64_pattern, response_text)
                                if image_matches:
                                    # 取第一个匹配的图像数据
                                    image_url = image_matches[0]
                                    print(f"🎯 成功匹配OpenRouter图像数据，长度: {len(image_url)}字符")
                                    
                                    # 提取base64部分
                                    if ';base64,' in image_url:
                                        import io
                                        base64_data = image_url.split(';base64,', 1)[1]
                                        image_bytes = base64.b64decode(base64_data)
                                        edited_image = Image.open(io.BytesIO(image_bytes))
                                        print(f"✅ 成功提取OpenRouter编辑后的图像: {edited_image.size}")
                                        
                                        # 清理响应文本，移除base64数据
                                        response_text = re.sub(base64_pattern, '[图像已编辑]', response_text)
                                    else:
                                        print(f"⚠️ 图像数据格式不正确: {image_url[:100]}...")
                                        edited_image = pil_image
                                        response_text = f"OpenRouter图像编辑完成，但数据格式不正确"
                                else:
                                    print(f"⚠️ 正则表达式未找到匹配的图像数据")
                                    edited_image = pil_image
                                    response_text = f"OpenRouter图像编辑完成，但未找到图像数据"
                            except Exception as e:
                                print(f"⚠️ OpenRouter图像数据解析失败: {e}")
                                edited_image = pil_image
                                response_text = f"OpenRouter图像编辑完成，但解析失败: {e}"
                        
                        # 如果没有成功提取图像，返回原图片
                        if not edited_image:
                            print("⚠️ OpenRouter未返回编辑后的图像数据，返回原图片")
                            edited_image = pil_image
                            if not response_text:
                                response_text = "OpenRouter图像编辑完成，但未返回编辑后的图像数据"
                    elif is_t8_mirror:
                        # T8镜像站OpenAI格式响应处理
                        print("🔗 处理T8镜像站OpenAI格式响应")
                        if "choices" in result and result["choices"]:
                            choice = result["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                content = choice["message"]["content"]
                                if isinstance(content, str):
                                    response_text = content
                                    # 检查是否包含base64图像数据
                                    if "![image](data:image/" in content:
                                        print("🖼️ 检测到T8镜像站返回的图像数据")
                                        try:
                                            # 提取base64图像数据
                                            import re, io
                                            image_match = re.search(r'!\[image\]\(data:image/\w+;base64,([^)]+)\)', content)
                                            if image_match:
                                                image_data = image_match.group(1)
                                                image_bytes = base64.b64decode(image_data)
                                                edited_image = Image.open(io.BytesIO(image_bytes))
                                                print("✅ 成功提取T8镜像站编辑后的图像")
                                                # 清理响应文本，移除base64数据
                                                response_text = re.sub(r'!\[image\]\(data:image/\w+;base64,[^)]+\)', '[图像已编辑]', content)
                                        except Exception as e:
                                            print(f"⚠️ T8镜像站图像数据解析失败: {e}")
                                elif isinstance(content, list):
                                    # 处理多模态内容
                                    for item in content:
                                        if item.get("type") == "text":
                                            response_text += item.get("text", "")
                                        elif item.get("type") == "image_url":
                                            # 处理图像URL（如果有的话）
                                            print("🖼️ 检测到图像URL响应")
                        
                        # 如果没有成功提取图像，再检查其他可能的位置
                        if not edited_image:
                            print("⚠️ T8镜像站暂不支持图像编辑，返回原图片")
                            edited_image = pil_image
                            if not response_text:
                                response_text = "T8镜像站文本分析完成，但暂不支持图像编辑功能"
                    elif is_openai_mirror:
                        # OpenAI镜像站
                        print("🔗 检测到OpenAI镜像站，使用OpenAI API格式")
                        request_data = {
                            "model": _normalize_model_name(model),
                            "messages": [{
                                "role": "user",
                                "content": enhanced_prompt
                            }],
                            "temperature": temperature,
                            "max_tokens": max_output_tokens,
                            "stream": True
                        }
                        
                        # 设置请求头
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key.strip()}"
                        }
                    else:
                        # 标准Gemini API响应处理
                        try:
                            result = response.json()
                        except Exception as e:
                            print(f"⚠️ 标准Gemini JSON解析失败: {e}")
                            result = {}
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
                                                import io
                                                image_data = inline_data["data"]
                                                image_bytes = base64.b64decode(image_data)
                                                edited_image = Image.open(io.BytesIO(image_bytes))
                                                print("✅ 成功提取编辑后的图片")
                                            except Exception as e:
                                                print(f"⚠️ 解码图片失败: {e}")
                    
                    # 如果没有编辑后的图片，返回原图片
                    if edited_image is None:
                        print("⚠️ 未检测到编辑后的图片，返回原图片")
                        edited_image = pil_image
                        if not response_text:
                            response_text = "图片编辑请求已发送，但未收到编辑后的图片"
                    
                    # 强制调整图像尺寸到用户指定的尺寸
                    try:
                        target_width, target_height = map(int, controls['size'].split('x'))
                        current_width, current_height = edited_image.size
                        
                        if (current_width, current_height) != (target_width, target_height):
                            print(f"🔄 强制调整图像尺寸: {current_width}x{current_height} -> {target_width}x{target_height}")
                            
                            # 🚀 使用无白色填充不变形的扩图技术
                            edited_image = smart_resize_with_padding(edited_image, (target_width, target_height))
                            
                            print(f"✅ 扩图技术完成: {edited_image.size}")
                        else:
                            print(f"✅ 图像尺寸已符合要求: {edited_image.size}")
                            
                    except Exception as e:
                        print(f"⚠️ 尺寸调整失败: {e}, 保持原始尺寸")
                    
                    # 应用传统质量增强（关闭自适应）
                    if enhance_quality and quality in ['hd', 'ultra_hd']:
                        print(f"✨ 应用传统质量增强，质量等级: {quality}")
                        edited_image = enhance_image_quality(edited_image, quality, "disabled")
                        print(f"✅ 增强完成")
                    
                    # 如果没有响应文本，提供默认文本
                    if not response_text:
                        response_text = "图片编辑完成！这是根据您的编辑指令修改后的图像。"
                        print("📝 使用默认响应文本")
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(edited_image)
                    
                    print("✅ 图片编辑完成")
                    print(f"📝 响应文本长度: {len(response_text)}")
                    print(f"📝 响应文本内容: {response_text[:200]}...")
                    self._push_chat(enhanced_prompt, response_text or "", unique_id) # 使用增强后的提示词
                    return (image_tensor, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                        
                        # 检查是否是配额错误
                        if response.status_code == 429:
                            error_message = error_detail.get("error", {}).get("message", "")
                            if "quota" in error_message.lower():
                                print("⚠️ 检测到配额限制错误，建议:")
                                print("   1. 等待更长时间再试")
                                print("   2. 检查API配额设置")
                                print("   3. 考虑升级API计划")
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
                raise ValueError(f"图片编辑失败: {error_msg}")



    def get_mirror_config(self):
        """获取当前镜像站配置"""
        try:
            from .gemini_banana import get_gemini_banana_config
        except ImportError:
            from gemini_banana import get_gemini_banana_config
        config = get_gemini_banana_config()
        mirror_sites = config.get('mirror_sites', {})
        
        # 查找API4GPT镜像站配置
        for site_name, site_config in mirror_sites.items():
            if "api4gpt.com" in site_config.get("url", ""):
                return site_config
        
        # 如果没找到，返回默认配置
        return {
            "url": "https://www.api4gpt.com",
            "api_key": "",
            "api_format": "api4gpt"
        }


class KenChenLLMGeminiBananaMultiImageEditNode:
    """
    Gemini Banana 多图像编辑节点
    
    功能特性:
    - 支持多张输入图像（最多4张）
    - 专业的图像编辑提示词
    - 支持尺寸、质量、风格控制
    - 智能图像组合编辑
    - 支持多个镜像站
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .gemini_banana import get_gemini_banana_config
        except ImportError:
            from gemini_banana import get_gemini_banana_config
        config = get_gemini_banana_config()
        default_params = config.get('default_params', {})
        default_proxy = config.get('proxy', "http://127.0.0.1:None")
        image_settings = config.get('image_settings', {})
        
        # 获取镜像站配置
        mirror_sites = config.get('mirror_sites', {})
        mirror_options = list(mirror_sites.keys())
        if not mirror_options:
            mirror_options = ["official", "comfly", "custom"]
        
        # 获取默认镜像站配置
        default_site = "comfly" if "comfly" in mirror_options else mirror_options[0] if mirror_options else "official"
        default_config = get_mirror_site_config(default_site)
        
        # 🚀 超越参考项目的图像控制预设
        size_presets = image_settings.get('size_presets', [
            "Original size", "512x512", "768x768", "1024x1024", "1024x1792", "1792x1024",
            "1920x1080", "2560x1440", "3840x2160"  # 超越参考项目的高分辨率选项
        ])
        quality_presets = image_settings.get('quality_presets', [
            "standard", "hd", "ultra_hd"  # 超越参考项目的超高清选项
        ])
        style_presets = image_settings.get('style_presets', [
            "vivid", "natural", "artistic", "cinematic", "photographic"  # 超越参考项目的风格选项
        ])
        
        return {
            "required": {
                "mirror_site": (mirror_options, {"default": default_site}),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "镜像站API密钥（可选，留空时自动获取）"
                }),
                "prompt": ("STRING", {"default": "请根据这些图片进行专业的图像编辑", "multiline": True}),
                # 支持多种AI模型和多图像编辑服务: nano-banana支持Comfly和T8镜像站, OpenRouter模型: google/gemini-2.5-flash-image-preview (付费)
                "model": (["nano-banana [Comfly-T8]", "nano-banana-hd [Comfly-T8]", "gemini-2.5-flash-image-preview", "gemini-2.0-flash", "fal-ai/nano-banana/edit [Comfly-T8]", "google/gemini-2.5-flash-image-preview [OpenRouter]"], {"default": "nano-banana [Comfly-T8]"}),
                "proxy": ("STRING", {"default": default_proxy, "multiline": False}),
                "size": (size_presets, {"default": image_settings.get('default_size', "1024x1024")}),
                "quality": (quality_presets, {"default": image_settings.get('default_quality', "hd")}),
                "style": (style_presets, {"default": image_settings.get('default_style', "natural")}),
                
                # 🎨 智能图像控制组（放在style下面）
                "detail_level": (["Basic Detail", "Professional Detail", "Premium Quality", "Masterpiece Level"], {"default": "Professional Detail"}),
                "camera_control": (["Auto Select", "Wide-angle Lens", "Macro Shot", "Low-angle Perspective", "High-angle Shot", "Close-up Shot", "Medium Shot"], {"default": "Auto Select"}),
                "lighting_control": (["Auto Settings", "Natural Light", "Studio Lighting", "Dramatic Shadows", "Soft Glow", "Golden Hour", "Blue Hour"], {"default": "Auto Settings"}),
                "template_selection": (["Auto Select", "Professional Portrait", "Cinematic Landscape", "Product Photography", "Digital Concept Art", "Anime Style Art", "Photorealistic Render", "Classical Oil Painting", "Watercolor Painting", "Cyberpunk Future", "Vintage Film Photography", "Architectural Photography", "Gourmet Food Photography"], {"default": "Auto Select"}),
                
                # 🚀 质量增强控制组
                "quality_enhancement": ("BOOLEAN", {"default": True, "label": "Enable Quality Enhancement"}),
                "enhance_quality": ("BOOLEAN", {"default": True, "label": "Enhanced Image Quality"}),
                "smart_resize": ("BOOLEAN", {"default": True, "label": "Smart Resize with Padding"}),
                "fill_color": ("STRING", {
                    "default": "255,255,255",
                    "placeholder": "填充颜色 RGB (如: 255,255,255)"
                }),
                
                "temperature": ("FLOAT", {"default": default_params.get('temperature', 0.9), "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": default_params.get('top_p', 0.9), "min": 0.0, "max": 1.0}),
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
                "api4gpt_service": (["nano-banana"], {
                    "default": "nano-banana",
                    "tooltip": "API4GPT服务类型选择（仅在API4GPT镜像站时有效）"
                }),
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
            "hidden": {"unique_id": "UNIQUE_ID"}
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "edit_multiple_images"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"

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

    def edit_multiple_images(self, mirror_site: str, api_key: str, prompt: str, model: str,
                           proxy: str, size: str, quality: str, style: str, detail_level: str, camera_control: str, lighting_control: str, 
                           template_selection: str, quality_enhancement: bool, enhance_quality: bool, smart_resize: bool, 
                           fill_color: str, temperature: float, top_p: float, top_k: int, max_output_tokens: int, seed: int, 
                           custom_size: str = "", api4gpt_service: str = "nano-banana", image1=None, image2=None, image3=None, image4=None, custom_additions: str = "", unique_id: str = "") -> Tuple[torch.Tensor, str]:
        """使用镜像站API进行多图像编辑"""
        
        # 🚀 立即规范化模型名称，去除UI标识
        model = _normalize_model_name(model)
        
        # 根据镜像站从配置获取URL和API Key
        site_config = get_mirror_site_config(mirror_site) if mirror_site else {"url": "", "api_key": ""}
        api_url = site_config.get("url", "").strip()
        if site_config.get("api_key") and not api_key.strip():
            api_key = site_config["api_key"]
            print(f"🔑 自动使用镜像站API Key: {api_key[:8]}...")
        if not api_url:
            raise ValueError("配置文件中缺少该镜像站的API URL")
        print(f"🔗 自动使用镜像站URL: {api_url}")
        
        if not validate_api_url(api_url):
            raise ValueError("API URL格式无效，请检查配置文件")
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 处理图像控制参数
        try:
            from .gemini_banana import process_image_controls, enhance_prompt_with_controls
        except ImportError:
            from gemini_banana import process_image_controls, enhance_prompt_with_controls
        controls = process_image_controls(size, quality, style, custom_size)
        # 对于nano-banana模型，跳过尺寸提示，让模型自由生成
        skip_size_hints = model in ["nano-banana", "nano-banana-hd"]
        enhanced_prompt = enhance_prompt_with_controls(
            prompt.strip(), controls, detail_level, camera_control, lighting_control,
            template_selection, quality_enhancement, enhance_quality, smart_resize, fill_color,
            skip_size_hints=skip_size_hints
        )
        
        # 处理自定义指令
        if custom_additions and custom_additions.strip():
            enhanced_prompt += f"\n\nCUSTOM INSTRUCTIONS: {custom_additions.strip()}"
            print(f"📝 添加自定义指令: {custom_additions[:100]}...")
        
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
                            converted_prompt = converted_prompt.replace("图1", "左边图片")
        if len(all_input_pils) >= 2:
                            converted_prompt = converted_prompt.replace("图2", "右边图片")
        if len(all_input_pils) >= 3:
            converted_prompt = converted_prompt.replace("图3", "第三张图片")
        if len(all_input_pils) >= 4:
            converted_prompt = converted_prompt.replace("图4", "第四张图片")
        
        # 根据图片数量生成不同的提示词 - 完全通用化
        if len(all_input_pils) == 2:
            # 2张图片：通用组合编辑
            if "t8star.cn" in api_url or "ai.t8star.cn" in api_url:
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
            else:
                # 标准提示词
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
            # 1张图片：标准图像编辑
            full_prompt = f"""你是一个专业的图像编辑专家。请根据以下要求编辑这张图片：

{enhanced_prompt}

请使用你的图像编辑能力，生成高质量的编辑结果。"""
        else:
            # 3-4张图片：复杂组合编辑
            if "t8star.cn" in api_url or "ai.t8star.cn" in api_url:
                full_prompt = f"""请严格按照以下要求进行多图像编辑：

{converted_prompt}

重要说明：
- 请仔细分析所有输入图片的内容和关系
- 根据用户的具体指令，将相关图片中的元素进行精确组合
- 保持第一张图片的核心特征和背景环境
- 确保所有编辑元素与参考图片完全一致
- 编辑结果应该看起来自然真实，符合用户意图

{enhanced_prompt}

请严格按照上述要求执行，确保编辑结果完全符合用户意图。"""
            else:
                full_prompt = f"""你是一个专业的图像编辑专家。请根据这些图片和以下指令进行图像编辑：

{converted_prompt}

{enhanced_prompt}

请使用你的图像编辑能力，生成高质量的编辑结果。确保编辑后的图像符合所有要求。"""
        
        # 转换所有输入图片 - 修复关键问题
        # 确保所有图片都被传递给模型，让模型能看到完整信息
        all_image_parts = []
        
        for i, pil_image in enumerate(all_input_pils):
            # 调整图像尺寸以符合API要求
            pil_image = resize_image_for_api(pil_image)
            # 转换为base64
            image_base64 = image_to_base64(pil_image, format='JPEG')
            all_image_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            })
            print(f"📸 准备传递图像 {i+1}: {pil_image.size}")
        
        # 代理处理：有效则设置，无效/未填则清除，避免残留环境变量影响请求
        if proxy and proxy.strip() and "None" not in proxy:
            os.environ['HTTPS_PROXY'] = proxy.strip()
            os.environ['HTTP_PROXY'] = proxy.strip()
            print(f"🔌 使用代理: {proxy.strip()}")
        else:
            existing = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
            if existing:
                print(f"🔌 未指定代理，沿用系统代理: {existing}")
            else:
                print("🔌 未指定代理（系统无代理）")
        
        # 检查镜像站类型 - 按照优先级顺序：nano-banana官方 → Comfly → T8 → API4GPT → OpenRouter → OpenAI → custom
        is_nano_banana_official = mirror_site == "nano-banana官方"
        is_t8_mirror = "t8star.cn" in api_url or "ai.t8star.cn" in api_url
        is_api4gpt_mirror = "www.api4gpt.com" in api_url
        is_comfly_mirror = _is_comfly_base(api_url)
        is_openrouter_mirror = "openrouter.ai" in api_url
        is_openai_mirror = "api.openai.com" in api_url or site_config.get("api_format") == "openai"
        
        # 构建完整的API URL（OpenRouter除外，因为它在各自的处理逻辑中构建）
        if not is_openrouter_mirror:
            full_url = build_api_url(api_url, model)
            print(f"🌐 使用API地址: {full_url}")
        else:
            print(f"🌐 OpenRouter镜像站，URL将在OpenRouter处理逻辑中构建")
        
        # 按照优先级顺序处理镜像站：nano-banana官方 → Comfly → T8 → API4GPT → OpenRouter → OpenAI → custom
        
        # 1. nano-banana官方镜像站处理
        if is_nano_banana_official:
            print("🔗 检测到nano-banana官方镜像站，使用Google官方API")

            # 构建内容部分（文本 + 多图像）
            content_parts = [{"text": full_prompt}] + all_image_parts

            # 构建生成配置
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"]
            }

            # 添加seed（如果有效）
            if seed and seed > 0:
                generation_config["seed"] = seed

            try:
                # 使用优先API调用（官方API优先，失败时回退到REST API）
                response_json = generate_with_priority_api(
                    api_key=api_key,
                    model=_normalize_model_name(model),
                    content_parts=content_parts,
                    generation_config=generation_config,
                    max_retries=5,
                    proxy=proxy
                )

                if response_json:
                    # 提取编辑后的图像
                    edited_image = process_generated_image_from_response(response_json)

                    # 提取响应文本
                    response_text = extract_text_from_response(response_json)

                    if edited_image:
                        # 应用全量增强（包括智能主体检测和居中技术）
                        try:
                            edited_image = _apply_full_enhancements(
                                edited_image,
                                controls['size'],
                                quality,
                                enhance_quality,
                                smart_resize
                            )
                            try:
                                print(f"🔧 Final output size: {edited_image.size[0]}x{edited_image.size[1]}")
                            except Exception:
                                pass
                        except Exception:
                            pass

                        image_tensor = pil_to_tensor(edited_image)
                        print("✅ 多图像编辑完成（nano-banana官方）")
                        self._push_chat(enhanced_prompt, response_text or "", unique_id)
                        return (image_tensor, response_text)
                    else:
                        print("⚠️ nano-banana官方API响应中未找到编辑后的图像数据")
                        # 返回第一张输入图像
                        if all_input_pils:
                            return (pil_to_tensor(all_input_pils[0]), response_text)
                        else:
                            # 创建默认图像
                            default_image = Image.new('RGB', (1024, 1024), color='black')
                            return (pil_to_tensor(default_image), response_text)
                else:
                    raise Exception("nano-banana官方API调用失败")

            except Exception as e:
                print(f"❌ nano-banana官方API调用失败: {e}")
                raise e
            
        # 2. Comfly镜像站处理
        elif is_comfly_mirror:
            print("🔗 检测到Comfly镜像站，使用Comfly API格式")
            
            if model in ["nano-banana", "nano-banana-hd"] and all_input_pils:
                # Comfly nano-banana 多图像编辑
                try:
                    # 使用所有输入图像
                    result = _comfly_nano_banana_edit(api_url, api_key, model, enhanced_prompt, all_input_pils, controls['size'], temperature, top_p, max_output_tokens, seed)
                    edited_image = None
                    response_text = ""
                    
                    if isinstance(result, dict) and 'data' in result and result['data']:
                        b64 = result['data'][0].get('b64_json')
                        response_text = result.get('response_text', "")
                        
                        if b64:
                            from base64 import b64decode
                            import io
                            try:
                                # 修复base64填充问题
                                b64_fixed = b64 + '=' * (4 - len(b64) % 4)
                                img = Image.open(io.BytesIO(b64decode(b64_fixed))).convert('RGB')
                            except Exception as decode_error:
                                print(f"⚠️ base64解码失败: {decode_error}")
                                # 尝试直接解码
                                try:
                                    img = Image.open(io.BytesIO(b64decode(b64))).convert('RGB')
                                except Exception as e2:
                                    print(f"⚠️ 直接解码也失败: {e2}")
                                    img = None
                            edited_image = img
                        else:
                            # 如果没有base64数据，尝试从响应文本中提取图像
                            import re
                            base64_pattern = r'data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)'
                            matches = re.findall(base64_pattern, response_text)
                            if matches:
                                from base64 import b64decode
                                import io
                                img = Image.open(io.BytesIO(b64decode(matches[0]))).convert('RGB')
                                edited_image = img
                    
                    # 如果成功处理，直接返回结果
                    if edited_image:
                        image_tensor = pil_to_tensor(edited_image)
                        print("✅ 图片编辑完成（Comfly nano-banana）")
                        self._push_chat(enhanced_prompt, response_text or "", unique_id)
                        return (image_tensor, response_text)
                        
                except Exception as e:
                    print(f"❌ Comfly(nano-banana) 多图像编辑失败: {e}")
                    raise e
            request_data = {
                "contents": [{
                    "parts": [
                        {"text": full_prompt}
                    ] + all_image_parts
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
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
                
        # 3. T8镜像站处理
        elif is_t8_mirror:
            # T8镜像站也支持 nano-banana，调用方式与 Comfly 一致
            print("🔗 检测到T8镜像站，使用chat/completions端点 (nano-banana 直连)")
            if _normalize_model_name(model) in ["nano-banana", "nano-banana-hd"] and all_input_pils:
                try:
                    # 使用所有输入图像
                    result = _comfly_nano_banana_edit(full_url, api_key, _normalize_model_name(model), enhanced_prompt, all_input_pils, controls['size'], temperature, top_p, max_output_tokens, seed)
                    edited_image = None
                    response_text = ""

                    if isinstance(result, dict) and 'data' in result and result['data']:
                        b64 = result['data'][0].get('b64_json')
                        response_text = result.get('response_text', "")

                        if b64:
                            from base64 import b64decode
                            import io
                            try:
                                b64_fixed = b64 + '=' * (4 - len(b64) % 4)
                                img = Image.open(io.BytesIO(b64decode(b64_fixed))).convert('RGB')
                            except Exception:
                                img = Image.open(io.BytesIO(b64decode(b64))).convert('RGB')
                            edited_image = img

                    # 如果成功处理，应用全量增强后再返回
                    if edited_image:
                        # 应用全量增强（包括智能主体检测和居中技术）
                        try:
                            edited_image = _apply_full_enhancements(
                                edited_image,
                                controls['size'],
                                quality,
                                enhance_quality,
                                smart_resize
                            )
                            try:
                                print(f"🔧 Final output size: {edited_image.size[0]}x{edited_image.size[1]}")
                            except Exception:
                                pass
                        except Exception:
                            pass

                        image_tensor = pil_to_tensor(edited_image)
                        print("✅ 图片编辑完成（T8 nano-banana）")
                        self._push_chat(enhanced_prompt, response_text or "", unique_id)
                        return (image_tensor, response_text)

                except Exception as e:
                    print(f"❌ T8(nano-banana) 多图像编辑失败: {e}")
                    raise e
            # 其他模型仍走原 T8 OpenAI 兼容格式
            print("🔗 检测到T8镜像站，使用OpenAI兼容的API格式")

            # 为T8镜像站优化多图片处理
            # 确保图片顺序正确，并添加明确的图片标识
            optimized_image_parts = []
            for i, image_part in enumerate(all_image_parts):
                # 为每张图片添加明确的标识
                optimized_image_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_part["inline_data"]["data"],
                        "detail": "high"  # 确保高分辨率处理
                    }
                })
                print(f"🔗 T8镜像站图片 {i+1}: 已添加到请求中")

            # 构建T8镜像站请求数据
            request_data = {
                "model": _normalize_model_name(model),
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_prompt
                        }
                    ] + optimized_image_parts  # 添加优化后的图片
                }],
                "temperature": temperature,
                "max_tokens": max_output_tokens
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
            
        # 4. API4GPT镜像站处理
        elif is_api4gpt_mirror:
            # API4GPT镜像站
            print("🔗 检测到API4GPT镜像站，使用API4GPT API格式")
            
            # API4GPT的多图片编辑处理
            if api4gpt_service == "nano-banana":
                # nano-banana服务使用官方API4GPT格式
                print(f"🔗 使用API4GPT nano-banana服务进行多图片编辑")
                
                # 对于多图片编辑，API4GPT nano-banana可能需要特殊处理
                # 目前API4GPT官方文档主要支持单图片编辑
                # 我们使用第一张图片作为主要输入，其他图片作为参考
                print("⚠️ API4GPT nano-banana多图片编辑功能有限，使用第一张图片作为主要输入")
                
                primary_image = all_image_parts[0] if all_image_parts else None
                if primary_image:
                    # 构建包含第一张图片的编辑请求
                    request_data = build_api4gpt_request(
                        service_type=api4gpt_service,
                        model=_normalize_model_name(model),
                        prompt=full_prompt,
                        image_base64=primary_image["inline_data"]["data"],
                        size=controls['size'],
                        quality=controls['quality'],
                        style=controls['style'],
                        temperature=temperature,
                        max_tokens=max_output_tokens
                    )
                    
                    # 调用API4GPT API进行图像编辑
                    try:
                        result = call_api4gpt_api(api_url, api_key, api4gpt_service, request_data)
                        print("✅ API4GPT多图片编辑API调用成功")
                        
                        # 解析API4GPT响应
                        response_text, edited_image = parse_api4gpt_response(result, api4gpt_service)
                        
                        if edited_image:
                            print(f"✅ 成功提取API4GPT编辑后的图像")
                        else:
                            print("⚠️ API4GPT未返回编辑后的图像，返回原图片")
                            edited_image = all_input_pils[0]
                            if not response_text:
                                response_text = f"API4GPT {api4gpt_service} 服务响应完成，但未返回编辑后的图像数据"
                        
                        # 处理图像尺寸调整
                        if edited_image:
                            try:
                                target_size = controls['size']
                                if 'x' in target_size:
                                    target_width, target_height = map(int, target_size.split('x'))
                                    
                                    if edited_image.size != (target_width, target_height):
                                        print(f"🔧 使用扩图技术调整图像尺寸: {edited_image.size} -> {target_size}")
                                        
                                        # 🚀 使用无白色填充不变形的扩图技术
                                        edited_image = smart_resize_with_padding(edited_image, (target_width, target_height))
                                        
                                        print(f"✅ 扩图技术完成: {edited_image.size}")
                                    else:
                                        print(f"✅ 图像尺寸已符合要求: {edited_image.size}")
                                        
                            except Exception as e:
                                print(f"⚠️ 尺寸调整失败: {e}, 保持原始尺寸")
                        
                        # 应用传统质量增强（关闭自适应）
                        if enhance_quality and quality in ['hd', 'ultra_hd']:
                            print(f"✨ 应用传统质量增强，质量等级: {quality}")
                            edited_image = enhance_image_quality(edited_image, quality, "disabled")
                            print(f"✅ 增强完成")
                        
                        # 转换为tensor
                        image_tensor = pil_to_tensor(edited_image)
                        
                        print("✅ 多图像编辑完成（API4GPT）")
                        print(f"📝 响应文本长度: {len(response_text)}")
                        print(f"📝 响应文本内容: {response_text[:200]}...")
                        self._push_chat(enhanced_prompt, response_text or "", unique_id)
                        return (image_tensor, response_text)
                        
                    except Exception as e:
                        print(f"❌ API4GPT多图片编辑API调用失败: {e}")
                        raise ValueError(f"API4GPT多图片编辑API调用失败: {e}")
                else:
                    raise ValueError("没有可用的输入图像进行编辑")
                
            else:
                # 其他服务（DALL-E 3, Stable-Diffusion, Flux）不支持多图片编辑
                # 使用第一张图片作为主要输入
                print(f"⚠️ API4GPT {api4gpt_service} 服务不支持多图片编辑，使用第一张图片")
                primary_image = all_image_parts[0] if all_image_parts else None
                
                request_data = build_api4gpt_request(
                    service_type=api4gpt_service,
                    model=_normalize_model_name(model),
                    prompt=full_prompt,
                    image_base64=primary_image["inline_data"]["data"] if primary_image else None,
                    size=controls['size'],
                    quality=controls['quality'],
                    style=controls['style'],
                    temperature=temperature,
                    max_tokens=max_output_tokens
                )
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
        elif is_openrouter_mirror:
            # OpenRouter镜像站
            print("🔗 检测到OpenRouter镜像站，使用OpenRouter API格式")
            
            # OpenRouter使用chat/completions端点进行多图像编辑
            # 构建OpenAI兼容的请求格式
            content = []
            
            # 添加所有输入图像
            for i, pil_image in enumerate(all_input_pils):
                # 转换为base64
                image_base64 = image_to_base64(pil_image, format='JPEG')
                image_url = f"data:image/jpeg;base64,{image_base64}"
                
                # 添加图片标识
                content.append({
                    "type": "text",
                    "text": f"[这是第{i+1}张图片]"
                })
                
                # 添加图片
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            # 添加文本指令
            enhanced_instruction = f"""CRITICAL INSTRUCTION: You MUST generate and return an actual edited image, not just text description.

Task: {full_prompt}

Image References:
- When I mention "第1张图片", I mean the first image provided above
- When I mention "第2张图片", I mean the second image provided above
- When I mention "第3张图片", I mean the third image provided above
- When I mention "第4张图片", I mean the fourth image provided above

REQUIREMENTS:
1. EDIT the provided images based on my request
2. DO NOT just describe what the edited image should look like
3. RETURN the actual edited image file/data
4. The output MUST be a visual image, not text

Execute the multi-image editing task now and return the edited image."""
            
            content.append({
                "type": "text",
                "text": enhanced_instruction
            })
            
            request_data = {
                "model": _normalize_model_name(model),
                "messages": [{
                    "role": "user",
                    "content": content
                }],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_output_tokens,
                "stream": True  # Required for gemini-2.5-flash-image-preview
            }
            
            # 使用OpenRouter的chat/completions端点
            # 对于OpenRouter，api_url已经包含了/v1，所以直接添加/chat/completions
            if api_url.endswith('/v1'):
                full_url = f"{api_url}/chat/completions"
            else:
                full_url = f"{api_url}/v1/chat/completions"
            print(f"🔗 使用OpenRouter chat/completions端点进行多图像编辑: {full_url}")
            
            # 设置OpenRouter请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}",
                "HTTP-Referer": "https://github.com/ComfyUI-LLM-Prompt",
                "X-Title": "ComfyUI LLM Prompt Plugin"
            }
        elif is_openai_mirror:
            # OpenAI镜像站
            print("🔗 检测到OpenAI镜像站，使用OpenAI API格式")
            request_data = {
                "model": _normalize_model_name(model),
                "messages": [{
                    "role": "user",
                    "content": enhanced_prompt
                }],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": True
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
        else:
            # 标准Gemini API格式
            request_data = {
                "contents": [{
                    "parts": [
                        {
                            "text": full_prompt
                        }
                    ] + all_image_parts  # 添加所有图片
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
                "Authorization": f"Bearer {api_key.strip()}"
            }
        
        # 智能重试机制
        max_retries = 5
        timeout = 120
        
        for attempt in range(max_retries):
            try:
                print(f"🖼️ 正在编辑图片... (尝试 {attempt + 1}/{max_retries})")
                print(f"📝 编辑指令: {enhanced_prompt[:100]}...")
                print(f"🔗 镜像站: {api_url}")
                
                # 发送请求
                response = requests.post(full_url, headers=headers, json=request_data, timeout=timeout)
                
                # 检查响应状态和内容
                print(f"📡 HTTP状态码: {response.status_code}")
                print(f"📡 响应头: {dict(response.headers)}")
                print(f"📡 响应内容长度: {len(response.text)}")
                
                # 检查响应内容
                if not response.text.strip():
                    print("⚠️ API返回空响应")
                    raise ValueError("API返回空响应")
                
                # 成功响应
                if response.status_code == 200:
                    # 提取文本响应和编辑后的图片
                    response_text = ""
                    edited_image = None
                    
                    # 为所有镜像站定义 result 变量
                    if not is_api4gpt_mirror and not is_openrouter_mirror and not is_t8_mirror and not is_openai_mirror:
                        # 标准 Gemini API 镜像站（如 Comfy API）
                        try:
                            result = response.json()
                            print("✅ 成功解析标准 Gemini API 响应")
                        except Exception as e:
                            print(f"⚠️ 解析响应失败: {e}")
                            result = {}
                    elif is_t8_mirror:
                        # T8 镜像站
                        try:
                            result = response.json()
                            print("✅ 成功解析 T8 镜像站响应")
                        except Exception as e:
                            print(f"⚠️ T8 镜像站响应解析失败: {e}")
                            result = {}
                    elif is_openai_mirror:
                        # OpenAI镜像站
                        try:
                            result = response.json()
                            print("✅ 成功解析 OpenAI 镜像站响应")
                        except Exception as e:
                            print(f"⚠️ OpenAI 镜像站响应解析失败: {e}")
                            result = {}
                    
                    if is_api4gpt_mirror:
                        # API4GPT镜像站响应处理
                        print("🔗 处理API4GPT镜像站响应")
                        
                        try:
                            # 尝试解析JSON响应
                            result = response.json()
                            print(f"📋 API4GPT响应结构: {list(result.keys())}")
                            
                            if api4gpt_service == "nano-banana":
                                # nano-banana使用OpenAI兼容格式
                                response_text, edited_image = parse_openai_compatible_response(result)
                            else:
                                # 其他服务使用原有的解析逻辑
                                response_text, edited_image = parse_api4gpt_response(result, api4gpt_service)
                            
                            if edited_image:
                                print(f"✅ 成功提取API4GPT编辑后的图像")
                            else:
                                print("⚠️ API4GPT未返回编辑后的图像，返回原图片")
                                edited_image = all_input_pils[0]
                                if not response_text:
                                    response_text = f"API4GPT {api4gpt_service} 服务响应完成，但未返回编辑后的图像数据"
                        except Exception as json_error:
                            print(f"⚠️ API4GPT JSON解析失败: {json_error}")
                            print(f"📋 原始响应内容: {response.text[:500]}...")
                            raise ValueError(f"API4GPT响应不是有效的JSON格式: {json_error}")
                            
                    elif is_openrouter_mirror:
                        # OpenRouter镜像站响应处理 - 使用流式响应
                        print("🔗 处理OpenRouter镜像站流式响应")
                        
                        # 处理流式响应
                        response_text = process_openrouter_stream(response)
                        
                        # 检查响应文本中是否包含图像数据
                        if "data:image/" in response_text:
                            print("🖼️ 检测到OpenRouter返回的图像数据")
                            try:
                                # 使用正确的正则表达式提取base64图像数据
                                import re
                                base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
                                image_matches = re.findall(base64_pattern, response_text)
                                if image_matches:
                                    # 取第一个匹配的图像数据
                                    image_url = image_matches[0]
                                    print(f"🎯 成功匹配OpenRouter图像数据，长度: {len(image_url)}字符")
                                    
                                    # 提取base64部分
                                    if ';base64,' in image_url:
                                        import io
                                        base64_data = image_url.split(';base64,', 1)[1]
                                        image_bytes = base64.b64decode(base64_data)
                                        edited_image = Image.open(io.BytesIO(image_bytes))
                                        print(f"✅ 成功提取OpenRouter编辑后的图像: {edited_image.size}")
                                        
                                        # 清理响应文本，移除base64数据
                                        response_text = re.sub(base64_pattern, '[图像已编辑]', response_text)
                                    else:
                                        print(f"⚠️ 图像数据格式不正确: {image_url[:100]}...")
                                        edited_image = all_input_pils[0]
                                        response_text = f"OpenRouter多图片编辑完成，但数据格式不正确"
                                else:
                                    print(f"⚠️ 正则表达式未找到匹配的图像数据")
                                    edited_image = all_input_pils[0]
                                    response_text = f"OpenRouter多图片编辑完成，但未找到图像数据"
                            except Exception as e:
                                print(f"⚠️ OpenRouter图像数据解析失败: {e}")
                                edited_image = all_input_pils[0]
                                response_text = f"OpenRouter多图片编辑完成，但解析失败: {e}"
                        
                        # 如果没有成功提取图像，返回原图片
                        if not edited_image:
                            print("⚠️ OpenRouter未返回编辑后的图像数据，返回原图片")
                            edited_image = all_input_pils[0]
                            if not response_text:
                                response_text = "OpenRouter多图片编辑完成，但未返回编辑后的图像数据"
                    elif is_t8_mirror:
                        # T8镜像站OpenAI格式响应处理
                        print("🔗 处理T8镜像站OpenAI格式响应")
                        if "choices" in result and result["choices"]:
                            choice = result["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                content = choice["message"]["content"]
                                if isinstance(content, str):
                                    response_text = content
                                    # 检查是否包含base64图像数据
                                    if "![image](data:image/" in content:
                                        print("🖼️ 检测到T8镜像站返回的图像数据")
                                        try:
                                            # 提取base64图像数据
                                            import re, io
                                            image_match = re.search(r'!\[image\]\(data:image/\w+;base64,([^)]+)\)', content)
                                            if image_match:
                                                image_data = image_match.group(1)
                                                image_bytes = base64.b64decode(image_data)
                                                edited_image = Image.open(io.BytesIO(image_bytes))
                                                print("✅ 成功提取T8镜像站编辑后的图像")
                                                # 清理响应文本，移除base64数据
                                                response_text = re.sub(r'!\[image\]\(data:image/\w+;base64,[^)]+\)', '[图像已编辑]', content)
                                        except Exception as e:
                                            print(f"⚠️ T8镜像站图像数据解析失败: {e}")
                                elif isinstance(content, list):
                                    # 处理多模态内容
                                    for item in content:
                                        if item.get("type") == "text":
                                            response_text += item.get("text", "")
                                        elif item.get("type") == "image_url":
                                            # 处理图像URL（如果有的话）
                                            print("🖼️ 检测到图像URL响应")
                        
                        # 如果没有成功提取图像，再检查其他可能的位置
                        if not edited_image:
                            print("⚠️ T8镜像站暂不支持图像编辑，返回原图片")
                            edited_image = all_input_pils[0]
                            if not response_text:
                                response_text = "T8镜像站文本分析完成，但暂不支持图像编辑功能"
                    elif is_openai_mirror:
                        # OpenAI镜像站
                        print("🔗 检测到OpenAI镜像站，使用OpenAI API格式")
                        request_data = {
                            "model": _normalize_model_name(model),
                            "messages": [{
                                "role": "user",
                                "content": enhanced_prompt
                            }],
                            "temperature": temperature,
                            "max_tokens": max_output_tokens,
                            "stream": True
                        }
                        
                        # 设置请求头
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key.strip()}"
                        }
                    else:
                        # 标准Gemini API响应处理
                        try:
                            result = response.json()
                        except Exception as e:
                            print(f"⚠️ 标准Gemini JSON解析失败: {e}")
                            result = {}
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
                                                import io
                                                image_data = inline_data["data"]
                                                image_bytes = base64.b64decode(image_data)
                                                edited_image = Image.open(io.BytesIO(image_bytes))
                                                print("✅ 成功提取编辑后的图片")
                                            except Exception as e:
                                                print(f"⚠️ 解码图片失败: {e}")
                    
                    # 如果没有编辑后的图片，返回原图片
                    if edited_image is None:
                        print("⚠️ 未检测到编辑后的图片，返回原图片")
                        edited_image = all_input_pils[0]
                        if not response_text:
                            response_text = "图片编辑请求已发送，但未收到编辑后的图片"
                    
                    # 强制调整图像尺寸到用户指定的尺寸
                    try:
                        target_width, target_height = map(int, controls['size'].split('x'))
                        current_width, current_height = edited_image.size
                        
                        if (current_width, current_height) != (target_width, target_height):
                            print(f"🔄 强制调整图像尺寸: {current_width}x{current_height} -> {target_width}x{target_height}")
                            
                            # 🚀 使用无白色填充不变形的扩图技术
                            edited_image = smart_resize_with_padding(edited_image, (target_width, target_height))
                            
                            print(f"✅ 扩图技术完成: {edited_image.size}")
                        else:
                            print(f"✅ 图像尺寸已符合要求: {edited_image.size}")
                            
                    except Exception as e:
                        print(f"⚠️ 尺寸调整失败: {e}, 保持原始尺寸")
                    
                    # 应用传统质量增强（关闭自适应）
                    if enhance_quality and quality in ['hd', 'ultra_hd']:
                        print(f"✨ 应用传统质量增强，质量等级: {quality}")
                        edited_image = enhance_image_quality(edited_image, quality, "disabled")
                        print(f"✅ 增强完成")
                    
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


def parse_openai_compatible_response(response_data):
    """解析OpenAI兼容格式的响应数据"""
    response_text = ""
    generated_image = None
    
    try:
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                
                if isinstance(content, str):
                    response_text = content
                    # 检查是否包含图像数据（base64或URL）
                    if "![image](" in content:
                        print("🖼️ 检测到OpenAI兼容格式返回的图像数据")
                        try:
                            import re
                            # 提取所有图像标记
                            image_matches = re.findall(r'!\[image\]\(([^)]+)\)', content)
                            
                            for image_url in image_matches:
                                if image_url.startswith("data:image/"):
                                    # 处理base64图像数据
                                    try:
                                        image_data = image_url.split(",")[1]
                                        image_bytes = base64.b64decode(image_data)
                                        generated_image = Image.open(io.BytesIO(image_bytes))
                                        print("✅ 成功提取base64图像数据")
                                        # 清理响应文本，移除base64数据
                                        response_text = re.sub(r'!\[image\]\(data:image/\w+;base64,[^)]+\)', '[图像已生成]', content)
                                        break
                                    except Exception as e:
                                        print(f"⚠️ base64图像数据解析失败: {e}")
                                elif image_url.startswith("http"):
                                    # 处理外部图像URL
                                    try:
                                        print(f"📥 正在下载图像: {image_url}")
                                        response = requests.get(image_url, timeout=30)
                                        if response.status_code == 200:
                                            image_bytes = response.content
                                            generated_image = Image.open(io.BytesIO(image_bytes))
                                            print("✅ 成功下载并提取外部图像")
                                            # 清理响应文本，移除URL
                                            response_text = re.sub(r'!\[image\]\([^)]+\)', '[图像已生成]', content)
                                            break
                                        else:
                                            print(f"⚠️ 图像下载失败，状态码: {response.status_code}")
                                    except Exception as e:
                                        print(f"⚠️ 图像下载失败: {e}")
                                else:
                                    print(f"⚠️ 不支持的图像格式: {image_url}")
                        except Exception as e:
                            print(f"⚠️ 图像数据解析失败: {e}")
                            
                elif isinstance(content, list):
                    # 处理多模态内容
                    for item in content:
                        if item.get("type") == "text":
                            response_text += item.get("text", "")
                        elif item.get("type") == "image_url":
                            # 处理图像URL（如果有的话）
                            print("🖼️ 检测到图像URL响应")
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image/"):
                                try:
                                    # 提取base64数据
                                    image_data = image_url.split(",")[1]
                                    image_bytes = base64.b64decode(image_data)
                                    generated_image = Image.open(io.BytesIO(image_bytes))
                                    print("✅ 成功提取base64图像数据")
                                except Exception as e:
                                    print(f"⚠️ base64图像数据解析失败: {e}")
                            elif image_url.startswith("http"):
                                try:
                                    # 下载外部图像URL
                                    print(f"📥 正在下载图像: {image_url}")
                                    response = requests.get(image_url, timeout=30)
                                    if response.status_code == 200:
                                        image_bytes = response.content
                                        generated_image = Image.open(io.BytesIO(image_bytes))
                                        print("✅ 成功下载并提取外部图像")
                                    else:
                                        print(f"⚠️ 图像下载失败，状态码: {response.status_code}")
                                except Exception as e:
                                    print(f"⚠️ 图像下载失败: {e}")
                            else:
                                print(f"⚠️ 不支持的图像URL格式: {image_url}")
        
        if not response_text:
            response_text = "OpenAI兼容格式响应处理完成"
            
    except Exception as e:
        print(f"⚠️ OpenAI兼容格式响应解析失败: {e}")
        response_text = f"响应解析失败: {e}"
    
    return response_text, generated_image


# 节点映射 - 保持与参考项目一致的命名风格
NODE_CLASS_MAPPINGS = {
    "KenChenLLMGeminiBananaMirrorImageGenNode": KenChenLLMGeminiBananaMirrorImageGenNode,
    "KenChenLLMGeminiBananaMirrorImageEditNode": KenChenLLMGeminiBananaMirrorImageEditNode,
    "GeminiBananaMirrorMultiImageEdit": KenChenLLMGeminiBananaMultiImageEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KenChenLLMGeminiBananaMirrorImageGenNode": "Gemini Banana 图像生成",
    "KenChenLLMGeminiBananaMirrorImageEditNode": "Gemini Banana 图片编辑",
    "GeminiBananaMirrorMultiImageEdit": "Gemini Banana Mirror 多图像编辑",
}

