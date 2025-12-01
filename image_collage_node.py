"""
图像拼接节点 - 支持最多30张图片的智能拼接
为 Gemini Banana 2 多图编辑优化，添加序号标识便于模型识别
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

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

def calculate_grid_layout(num_images):
    """
    计算最优的网格布局
    
    优先使用接近正方形的布局，便于模型识别
    """
    if num_images <= 0:
        return 1, 1
    elif num_images == 1:
        return 1, 1
    elif num_images == 2:
        return 2, 1  # 横向排列
    elif num_images <= 4:
        return 2, 2  # 2x2
    elif num_images <= 6:
        return 3, 2  # 3x2
    elif num_images <= 9:
        return 3, 3  # 3x3
    elif num_images <= 12:
        return 4, 3  # 4x3
    elif num_images <= 16:
        return 4, 4  # 4x4
    elif num_images <= 20:
        return 5, 4  # 5x4
    elif num_images <= 25:
        return 5, 5  # 5x5
    else:  # 26-30
        return 6, 5  # 6x5 (最多30张)

class KenChenLLMGeminiBananaImageCollageNode:
    """
    图像拼接节点 - 智能拼接最多30张图片
    
    功能特性:
    - 支持1-30张图片输入
    - 自动计算最优网格布局
    - 添加序号标识（1-30）
    - 智能调整图片尺寸
    - 生成图片位置说明文本
    - 优化用于 Gemini Banana 2 多图编辑
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                # 拼接设置
                "max_cell_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "单元格最大尺寸（图片会等比缩放到此尺寸内，保持长宽比）"
                }),
                "resize_mode": (["keep_aspect_ratio", "fit_to_cell", "original_size"], {
                    "default": "keep_aspect_ratio",
                    "tooltip": "缩放模式：keep_aspect_ratio=保持长宽比，fit_to_cell=填满单元格（可能变形），original_size=保持原始尺寸（小图不放大）"
                }),
                "add_numbers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否在每张图片上添加序号标识"
                }),
                "number_size": ("INT", {
                    "default": 48,
                    "min": 24,
                    "max": 128,
                    "step": 8,
                    "tooltip": "序号文字大小"
                }),
                "number_position": (["top-left", "top-right", "bottom-left", "bottom-right", "center"], {
                    "default": "top-left",
                    "tooltip": "序号位置"
                }),
                "background_color": (["white", "black", "gray"], {
                    "default": "white",
                    "tooltip": "背景颜色（用于填充空白区域）"
                }),
            },
            "optional": {}
        }
        
        # 动态添加 image1 到 image30
        for i in range(1, 31):
            inputs["optional"][f"image{i}"] = ("IMAGE",)
            
        return inputs
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("collage_image", "position_guide")
    FUNCTION = "create_collage"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"
    
    # 设置节点颜色
    color = "#9370DB"  # 中紫色
    bgcolor = "#8B008B"  # 深紫色
    groupcolor = "#DDA0DD"  # 梅红色
    
    def __init__(self):
        self.color = "#9370DB"
        self.bgcolor = "#8B008B"
        self.groupcolor = "#DDA0DD"
    
    def create_collage(self, max_cell_size, resize_mode, add_numbers, number_size, number_position, background_color, **kwargs):
        """创建图片拼接"""
        
        # 收集所有输入的图片
        input_images = []
        for i in range(1, 31):
            img_name = f"image{i}"
            if img_name in kwargs:
                input_images.append(kwargs[img_name])
            else:
                input_images.append(None)
        
        # 过滤掉 None 的图片并转换为 PIL
        valid_pil_images = []
        image_indices = []
        for i, img in enumerate(input_images):
            if img is not None:
                pil_img = tensor_to_pil(img)
                valid_pil_images.append(pil_img)
                image_indices.append(i + 1)  # 1-based index
        
        if not valid_pil_images:
            raise ValueError("至少需要输入一张图片")
        
        num_images = len(valid_pil_images)
        print(f"🖼️ 收集到 {num_images} 张图片，序号: {image_indices}")
        
        # 计算网格布局
        cols, rows = calculate_grid_layout(num_images)
        print(f"📐 使用 {cols}x{rows} 网格布局")
        
        # 🚀 智能处理每张图片的尺寸
        processed_images = []
        actual_cell_width = 0
        actual_cell_height = 0
        
        for idx, pil_img in enumerate(valid_pil_images):
            orig_w, orig_h = pil_img.size
            print(f"📸 图片 {image_indices[idx]} 原始尺寸: {orig_w}x{orig_h}")
            
            if resize_mode == "original_size":
                # 保持原始尺寸（小图不放大）
                if orig_w <= max_cell_size and orig_h <= max_cell_size:
                    resized_img = pil_img
                    print(f"  ✅ 保持原始尺寸: {orig_w}x{orig_h}")
                else:
                    # 等比缩放到 max_cell_size 内
                    scale = min(max_cell_size / orig_w, max_cell_size / orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    print(f"  ✅ 等比缩放: {orig_w}x{orig_h} → {new_w}x{new_h}")
            
            elif resize_mode == "keep_aspect_ratio":
                # 等比缩放到 max_cell_size 内（推荐）
                scale = min(max_cell_size / orig_w, max_cell_size / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                print(f"  ✅ 等比缩放: {orig_w}x{orig_h} → {new_w}x{new_h}")
            
            else:  # fit_to_cell
                # 强制缩放到正方形（可能变形）
                resized_img = pil_img.resize((max_cell_size, max_cell_size), Image.Resampling.LANCZOS)
                print(f"  ⚠️ 强制缩放: {orig_w}x{orig_h} → {max_cell_size}x{max_cell_size}")
            
            processed_images.append(resized_img)
            
            # 更新实际单元格尺寸（取最大值）
            actual_cell_width = max(actual_cell_width, resized_img.width)
            actual_cell_height = max(actual_cell_height, resized_img.height)
        
        # 使用实际的最大尺寸作为单元格尺寸
        cell_width = actual_cell_width
        cell_height = actual_cell_height
        print(f"📏 实际单元格尺寸: {cell_width}x{cell_height}")
        
        # 计算拼接图尺寸
        collage_width = cols * cell_width
        collage_height = rows * cell_height
        print(f"🎨 拼接图尺寸: {collage_width}x{collage_height}")
        
        # 创建背景画布
        bg_colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128)
        }
        bg_color = bg_colors.get(background_color, (255, 255, 255))
        collage = Image.new('RGB', (collage_width, collage_height), bg_color)
        draw = ImageDraw.Draw(collage)
        
        # 尝试加载字体
        try:
            # 尝试使用系统字体
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",  # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, number_size)
                    break
            if font is None:
                font = ImageFont.load_default()
                print("⚠️ 使用默认字体")
        except Exception as e:
            print(f"⚠️ 加载字体失败: {e}，使用默认字体")
            font = ImageFont.load_default()
        
        # 生成位置说明文本
        position_guide_lines = [
            f"图片拼接布局: {cols}列 x {rows}行",
            f"总共 {num_images} 张图片",
            "",
            "图片位置说明:"
        ]
        
        # 拼接图片
        for idx, (pil_img, img_num) in enumerate(zip(processed_images, image_indices)):
            # 计算网格位置
            row = idx // cols
            col = idx % cols
            cell_x = col * cell_width
            cell_y = row * cell_height
            
            # 计算图片在单元格中的居中位置
            img_w, img_h = pil_img.size
            paste_x = cell_x + (cell_width - img_w) // 2
            paste_y = cell_y + (cell_height - img_h) // 2
            
            # 粘贴图片（居中）
            collage.paste(pil_img, (paste_x, paste_y))
            
            print(f"  📍 图片 {img_num}: 粘贴到 ({paste_x}, {paste_y}), 尺寸 {img_w}x{img_h}")
            
            # 添加序号标识
            if add_numbers:
                # 计算序号位置（基于实际图片位置）
                if number_position == "top-left":
                    text_x, text_y = paste_x + 10, paste_y + 10
                elif number_position == "top-right":
                    text_x, text_y = paste_x + img_w - number_size - 10, paste_y + 10
                elif number_position == "bottom-left":
                    text_x, text_y = paste_x + 10, paste_y + img_h - number_size - 10
                elif number_position == "bottom-right":
                    text_x, text_y = paste_x + img_w - number_size - 10, paste_y + img_h - number_size - 10
                else:  # center
                    text_x, text_y = paste_x + img_w // 2 - number_size // 2, paste_y + img_h // 2 - number_size // 2
                
                # 绘制序号背景（半透明）
                padding = 8
                bbox = draw.textbbox((text_x, text_y), str(img_num), font=font)
                bg_rect = [
                    bbox[0] - padding,
                    bbox[1] - padding,
                    bbox[2] + padding,
                    bbox[3] + padding
                ]
                
                # 根据背景色选择序号颜色
                if background_color == "black":
                    number_bg_color = (255, 255, 255, 200)  # 白色背景
                    number_text_color = (0, 0, 0)  # 黑色文字
                else:
                    number_bg_color = (0, 0, 0, 200)  # 黑色背景
                    number_text_color = (255, 255, 255)  # 白色文字
                
                # 绘制半透明背景
                overlay = Image.new('RGBA', collage.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(bg_rect, fill=number_bg_color)
                collage_rgba = collage.convert('RGBA')
                collage_rgba = Image.alpha_composite(collage_rgba, overlay)
                collage = collage_rgba.convert('RGB')
                draw = ImageDraw.Draw(collage)
                
                # 绘制序号文字
                draw.text((text_x, text_y), str(img_num), fill=number_text_color, font=font)
            
            # 添加到位置说明
            position_guide_lines.append(f"  图片 {img_num}: 第 {row + 1} 行，第 {col + 1} 列")
        
        # 生成完整的位置说明文本
        position_guide_lines.extend([
            "",
            "使用说明:",
            "- 在提示词中使用 '图1'、'图2' 等来引用对应的图片",
            "- 例如: '将图1的人物和图2的背景结合'",
            "- 模型会根据序号识别每张图片",
            f"- 布局: {cols}列 x {rows}行，从左到右、从上到下编号"
        ])
        
        position_guide = "\n".join(position_guide_lines)
        
        print(f"✅ 拼接完成: {num_images} 张图片 -> {collage_width}x{collage_height}")
        print(f"📋 位置说明:\n{position_guide}")
        
        # 转换为 tensor
        collage_tensor = pil_to_tensor(collage)
        
        return (collage_tensor, position_guide)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "KenChenLLMGeminiBananaImageCollageNode": KenChenLLMGeminiBananaImageCollageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KenChenLLMGeminiBananaImageCollageNode": "🍌 Gemini Banana 图片拼接 (最多30张)",
}
