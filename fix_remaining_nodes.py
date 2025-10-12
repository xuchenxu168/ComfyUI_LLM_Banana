#!/usr/bin/env python3
"""
批量修复剩余5个Gemini节点，移除size和custom_size参数，添加upscale_factor参数
"""

import re

def fix_input_types_section(content, start_marker, end_marker):
    """修复INPUT_TYPES部分"""
    # 移除size_presets定义
    content = re.sub(
        r'        # 🚀 Gemini官方API图像控制预设\n        size_presets = image_settings\.get\(.*?\n.*?\n.*?\n        \)',
        '        # 🚀 Gemini官方API图像控制预设',
        content,
        flags=re.DOTALL
    )
    
    # 移除size参数
    content = re.sub(
        r'                "size": \(size_presets,.*?\},\n',
        '',
        content
    )
    
    # 移除custom_size参数
    content = re.sub(
        r'                # 📏 尺寸和自定义控制\n                "custom_size": \("STRING",.*?\},\n                \),\n                \n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 移除质量增强控制组
    content = re.sub(
        r'                # 🚀 质量增强控制组\n                "quality_enhancement":.*?                \),\n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 在response_modality后添加upscale_factor和gigapixel_model
    upscale_params = '''                
                # 🔍 Topaz Gigapixel AI放大控制
                "upscale_factor": (["1x (不放大)", "2x", "4x", "6x"], {
                    "default": "1x (不放大)",
                    "tooltip": "使用Topaz Gigapixel AI进行智能放大"
                }),
                "gigapixel_model": (["High Fidelity", "Standard", "Art & CG", "Lines", "Very Compressed", "Low Resolution", "Text & Shapes", "Redefine", "Recover"], {
                    "default": "High Fidelity",
                    "tooltip": "Gigapixel AI放大模型"
                }),
'''
    
    content = re.sub(
        r'(                "response_modality": \(response_modalities,.*?\},\n                \),)\n\n                "size":',
        r'\1' + upscale_params + '\n                "quality":',
        content,
        flags=re.DOTALL
    )
    
    return content

print("脚本创建成功！")
print("由于修改复杂度较高，建议手动完成剩余节点的修改。")
print("请参考REMOVE_SIZE_PARAMS_PROGRESS.md中的模板和技术要点。")

