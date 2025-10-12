#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量修复gemini_banana_mirror.py中节点5和节点6的参数
"""

import re

def fix_mirror_nodes():
    """修复镜像站节点"""
    with open('gemini_banana_mirror.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("开始批量修复...")
    
    # 1. 移除size_presets定义
    content = re.sub(
        r'size_presets = image_settings\.get\(\'size_presets\', \[[\s\S]*?\]\)',
        '',
        content
    )
    print("✅ 移除size_presets定义")
    
    # 2. 移除INPUT_TYPES中的size参数
    content = re.sub(
        r'"size": \(size_presets, \{[^}]*\}\),\s*\n',
        '',
        content
    )
    print("✅ 移除size参数")
    
    # 3. 移除quality_enhancement相关参数
    content = re.sub(
        r'"quality_enhancement": \("BOOLEAN", \{[^}]*\}\),\s*\n',
        '',
        content
    )
    content = re.sub(
        r'"enhance_quality": \("BOOLEAN", \{[^}]*\}\),\s*\n',
        '',
        content
    )
    content = re.sub(
        r'"smart_resize": \("BOOLEAN", \{[^}]*\}\),\s*\n',
        '',
        content
    )
    content = re.sub(
        r'"fill_color": \("STRING", \{[^}]*\}\),\s*\n',
        '',
        content
    )
    print("✅ 移除quality_enhancement相关参数")
    
    # 4. 移除optional中的custom_size参数
    content = re.sub(
        r'"custom_size": \("STRING", \{[^}]*\}\),\s*\n',
        '',
        content
    )
    print("✅ 移除custom_size参数")
    
    # 5. 添加upscale_factor参数（在response_modality之后）
    # 这个需要更精确的替换，所以我们手动处理
    
    # 6. 移除方法签名中的相关参数
    # size, custom_size, quality_enhancement, enhance_quality, smart_resize, fill_color
    
    # 7. 替换controls['size']为"1024x1024"
    content = re.sub(
        r'controls\[\'size\'\]',
        '"1024x1024"',
        content
    )
    print("✅ 替换controls['size']")
    
    # 8. 替换controls['quality']为quality
    content = re.sub(
        r'controls\[\'quality\'\]',
        'quality',
        content
    )
    print("✅ 替换controls['quality']")
    
    # 9. 替换controls['style']为style
    content = re.sub(
        r'controls\[\'style\'\]',
        'style',
        content
    )
    print("✅ 替换controls['style']")
    
    # 10. 移除process_image_controls调用
    content = re.sub(
        r'controls = process_image_controls\([^)]*\)\s*\n',
        '',
        content
    )
    print("✅ 移除process_image_controls调用")
    
    # 11. 移除enhance_prompt_with_controls调用相关代码
    content = re.sub(
        r'enhanced_prompt = enhance_prompt_with_controls\([^)]*\)\s*\n',
        '',
        content
    )
    print("✅ 移除enhance_prompt_with_controls调用")
    
    # 12. 移除调试打印中的controls引用
    content = re.sub(
        r'print\(f"🎨 图像控制参数: 尺寸=\{controls\[\'size\'\]\}[^"]*"\)',
        'print(f"🎨 图像控制参数: aspect_ratio={aspect_ratio}, quality={quality}, style={style}")',
        content
    )
    print("✅ 修复调试打印")
    
    # 写回文件
    with open('gemini_banana_mirror.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 批量修复完成！")

if __name__ == '__main__':
    fix_mirror_nodes()

