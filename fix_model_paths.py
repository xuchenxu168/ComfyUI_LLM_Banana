#!/usr/bin/env python3
"""
🔧 模型路径检测和修复脚本
解决Real-ESRGAN模型找不到的问题
"""

import os
import sys
import shutil
from pathlib import Path

def find_model_files():
    """查找所有模型文件"""
    print("🔍 开始查找Real-ESRGAN模型文件...")
    
    # 可能的模型文件名
    model_files = [
        'RealESRGAN_x2plus.pth',
        'RealESRGAN_x4plus.pth', 
        'RealESRGAN_x8plus.pth'
    ]
    
    # 可能的目录
    search_dirs = [
        '.',  # 当前目录
        'upscale_models',  # upscale_models文件夹
        'models',  # models文件夹
        'downloads',  # downloads文件夹
        'Downloads',  # Downloads文件夹
        'download',  # download文件夹
        'Download'   # Download文件夹
    ]
    
    found_models = {}
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"📁 检查目录: {search_dir}")
            for model_file in model_files:
                model_path = os.path.join(search_dir, model_file)
                if os.path.exists(model_path):
                    found_models[model_file] = model_path
                    print(f"✅ 找到模型: {model_file} -> {model_path}")
    
    return found_models

def create_models_directory():
    """创建models目录"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"📁 创建目录: {models_dir}")
    else:
        print(f"📁 目录已存在: {models_dir}")
    return models_dir

def organize_model_files(found_models):
    """整理模型文件到models目录"""
    if not found_models:
        print("❌ 未找到任何模型文件")
        return False
    
    models_dir = create_models_directory()
    
    print(f"\n🔧 开始整理模型文件到 {models_dir} 目录...")
    
    for model_name, model_path in found_models.items():
        target_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(target_path):
            print(f"⚠️ 目标文件已存在: {target_path}")
            # 询问是否覆盖
            response = input(f"是否覆盖 {model_name}? (y/N): ").strip().lower()
            if response != 'y':
                print(f"⏭️ 跳过 {model_name}")
                continue
        
        try:
            shutil.copy2(model_path, target_path)
            print(f"✅ 复制成功: {model_name} -> {target_path}")
        except Exception as e:
            print(f"❌ 复制失败 {model_name}: {e}")
    
    return True

def download_missing_models():
    """下载缺失的模型文件"""
    print("\n📥 检查缺失的模型文件...")
    
    required_models = [
        'RealESRGAN_x4plus.pth',  # 最常用的4x模型
        'RealESRGAN_x2plus.pth',  # 2x模型
        'RealESRGAN_x8plus.pth'   # 8x模型
    ]
    
    models_dir = 'models'
    missing_models = []
    
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
            print(f"❌ 缺失模型: {model}")
        else:
            print(f"✅ 模型已存在: {model}")
    
    if missing_models:
        print(f"\n📥 需要下载 {len(missing_models)} 个模型文件")
        print("💡 下载链接:")
        for model in missing_models:
            if 'x4plus' in model:
                print(f"   {model}: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model}")
            elif 'x2plus' in model:
                print(f"   {model}: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model}")
            elif 'x8plus' in model:
                print(f"   {model}: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model}")
        
        print("\n🔧 手动下载步骤:")
        print("1. 点击上述链接下载模型文件")
        print("2. 将下载的文件移动到 models/ 目录")
        print("3. 重新运行此脚本验证")
    else:
        print("🎉 所有必需模型都已存在！")

def test_model_loading():
    """测试模型加载"""
    print("\n🧪 测试模型加载...")
    
    try:
        # 测试导入
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        print("✅ 成功导入Real-ESRGAN相关模块")
        
        # 测试模型文件
        models_dir = 'models'
        test_model = 'RealESRGAN_x4plus.pth'
        model_path = os.path.join(models_dir, test_model)
        
        if os.path.exists(model_path):
            print(f"✅ 模型文件存在: {model_path}")
            
            # 尝试加载模型
            try:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
                upsampler = RealESRGANer(scale=4, model_path=model_path, model=model)
                print("✅ 模型加载成功！")
                return True
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                return False
        else:
            print(f"❌ 模型文件不存在: {model_path}")
            return False
            
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("💡 请先安装Real-ESRGAN: pip install realesrgan basicsr facexlib")
        return False

def main():
    """主函数"""
    print("🔧 Real-ESRGAN模型路径修复脚本")
    print("=" * 50)
    
    # 查找现有模型文件
    found_models = find_model_files()
    
    if found_models:
        print(f"\n📊 找到 {len(found_models)} 个模型文件")
        
        # 整理模型文件
        if organize_model_files(found_models):
            print("✅ 模型文件整理完成")
        else:
            print("❌ 模型文件整理失败")
    else:
        print("❌ 未找到任何模型文件")
    
    # 检查缺失的模型
    download_missing_models()
    
    # 测试模型加载
    if test_model_loading():
        print("\n🎉 所有问题已解决！")
        print("💡 现在可以在Gemini Banana节点中使用AI放大了")
    else:
        print("\n⚠️ 仍有问题需要解决")
        print("💡 请检查模型文件和依赖安装")

if __name__ == "__main__":
    main() 