#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理gemini_banana_mirror.py文件中的重复代码
"""

def cleanup_mirror_file():
    """清理文件"""
    with open('gemini_banana_mirror.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"原始文件总行数: {len(lines)}")
    
    # 找到需要删除的范围：从4410行到5185行（Python索引从0开始，所以是4409到5184）
    # 保留4409行（注释行）和5185行之后的内容
    
    # 删除4410行到5185行的内容
    new_lines = lines[:4409] + ['\n'] + lines[5185:]
    
    print(f"清理后文件总行数: {len(new_lines)}")
    print(f"删除了 {len(lines) - len(new_lines)} 行")
    
    # 写回文件
    with open('gemini_banana_mirror.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✅ 文件清理完成！")

if __name__ == '__main__':
    cleanup_mirror_file()

