#!/usr/bin/env python3
"""
检查ComfyUI中的节点注册情况
"""

import sys
import os

# 添加ComfyUI路径
comfyui_path = r"d:\Ken_ComfyUI_312\ComfyUI"
sys.path.insert(0, comfyui_path)

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def check_comfyui_nodes():
    """检查ComfyUI中的节点注册"""
    try:
        print("🔍 正在检查ComfyUI节点注册...")
        
        # 模拟ComfyUI的节点加载过程
        from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        print(f"✅ 从__init__.py加载到 {len(NODE_CLASS_MAPPINGS)} 个节点:")
        
        # 查找多图像编辑节点
        multi_image_nodes = []
        for key, value in NODE_CLASS_MAPPINGS.items():
            if "MultiImage" in key or "多图" in str(value):
                multi_image_nodes.append((key, value))
                
        if multi_image_nodes:
            print(f"✅ 找到 {len(multi_image_nodes)} 个多图像相关节点:")
            for key, value in multi_image_nodes:
                display_name = NODE_DISPLAY_NAME_MAPPINGS.get(key, "未知")
                print(f"   - {key}: {display_name}")
                print(f"     类: {value}")
        else:
            print("❌ 未找到多图像相关节点")
            
        # 特别检查GeminiBananaMultiImageEdit
        if "GeminiBananaMultiImageEdit" in NODE_CLASS_MAPPINGS:
            print(f"\n✅ 确认找到 GeminiBananaMultiImageEdit 节点!")
            node_class = NODE_CLASS_MAPPINGS["GeminiBananaMultiImageEdit"]
            display_name = NODE_DISPLAY_NAME_MAPPINGS.get("GeminiBananaMultiImageEdit", "未知")
            print(f"   - 显示名称: {display_name}")
            print(f"   - 类名: {node_class.__name__}")
            print(f"   - 模块: {node_class.__module__}")
            
            # 测试节点功能
            try:
                input_types = node_class.INPUT_TYPES()
                print(f"   - INPUT_TYPES 正常: ✅")
                print(f"   - 必需参数: {len(input_types.get('required', {}))}")
                print(f"   - 可选参数: {len(input_types.get('optional', {}))}")
            except Exception as e:
                print(f"   - INPUT_TYPES 错误: ❌ {e}")
                
        else:
            print(f"\n❌ 未找到 GeminiBananaMultiImageEdit 节点")
            print("可用的节点列表:")
            for key in sorted(NODE_CLASS_MAPPINGS.keys()):
                print(f"   - {key}")
                
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始检查ComfyUI节点注册...")
    success = check_comfyui_nodes()
    
    if success:
        print("\n🎉 检查完成!")
    else:
        print("\n💥 检查失败!")
        sys.exit(1)
