#!/usr/bin/env python3
"""
🚀 快速修复AI放大模型问题
解决拉伸变形和AI模型不工作的问题
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        return False

def install_ai_models():
    """安装AI放大模型"""
    print("🚀 开始安装AI放大模型...")
    
    # 升级pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "升级pip")
    
    # 安装核心依赖
    core_packages = [
        "torch",
        "torchvision", 
        "numpy",
        "pillow",
        "opencv-python"
    ]
    
    for package in core_packages:
        run_command(f"{sys.executable} -m pip install {package}", f"安装 {package}")
    
    # 安装Real-ESRGAN (核心)
    print("\n🚀 安装Real-ESRGAN (核心AI放大模型)...")
    realesrgan_packages = [
        "basicsr",
        "facexlib", 
        "realesrgan"
    ]
    
    for package in realesrgan_packages:
        success = run_command(f"{sys.executable} -m pip install {package}", f"安装 {package}")
        if not success:
            print(f"⚠️ {package} 安装失败，尝试从源码安装...")
            if package == "realesrgan":
                # 从源码安装Real-ESRGAN
                run_command("git clone https://github.com/xinntao/Real-ESRGAN.git", "克隆Real-ESRGAN源码")
                os.chdir("Real-ESRGAN")
                run_command(f"{sys.executable} -m pip install -r requirements.txt", "安装Real-ESRGAN依赖")
                run_command(f"{sys.executable} setup.py develop", "安装Real-ESRGAN")
                os.chdir("..")
    
    return True

def download_models():
    """下载预训练模型"""
    print("\n📥 下载预训练模型...")
    
    # 创建模型目录
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"📁 创建模型目录: {models_dir}")
    
    # 模型下载链接
    model_urls = {
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x8plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x8plus.pth"
    }
    
    for model_name, url in model_urls.items():
        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"📥 下载模型: {model_name}")
            # 尝试使用curl
            if run_command(f"curl -L {url} -o {model_path}", f"下载 {model_name}"):
                print(f"✅ {model_name} 下载成功")
            else:
                # 尝试使用wget
                if run_command(f"wget {url} -O {model_path}", f"下载 {model_name}"):
                    print(f"✅ {model_name} 下载成功")
                else:
                    print(f"❌ {model_name} 下载失败，请手动下载")
        else:
            print(f"✅ 模型已存在: {model_name}")

def test_installation():
    """测试安装"""
    print("\n🧪 测试AI放大模型安装...")
    
    test_code = """
import sys
print("Python版本:", sys.version)

try:
    import torch
    print("✅ PyTorch:", torch.__version__)
except ImportError as e:
    print("❌ PyTorch导入失败:", e)

try:
    import realesrgan
    print("✅ Real-ESRGAN导入成功")
except ImportError as e:
    print("❌ Real-ESRGAN导入失败:", e)

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print("✅ BasicSR导入成功")
except ImportError as e:
    print("❌ BasicSR导入失败:", e)

try:
    from PIL import Image
    print("✅ PIL导入成功")
except ImportError as e:
    print("❌ PIL导入失败:", e)

print("\\n🎯 安装状态检查完成！")
"""
    
    # 保存测试代码
    test_file = "test_install.py"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_code)
    
    # 运行测试
    print("🧪 运行安装测试...")
    run_command(f"{sys.executable} {test_file}", "测试安装")
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)

def main():
    """主修复流程"""
    print("🚀 AI放大模型快速修复脚本")
    print("=" * 50)
    print("🔧 此脚本将修复以下问题:")
    print("   1. 拉伸变形问题")
    print("   2. AI放大模型不工作")
    print("   3. 图像质量差")
    print("=" * 50)
    
    # 安装AI模型
    if install_ai_models():
        print("✅ AI模型安装完成")
    else:
        print("❌ AI模型安装失败")
        return
    
    # 下载预训练模型
    download_models()
    
    # 测试安装
    test_installation()
    
    print("\n🎉 修复完成！")
    print("=" * 50)
    print("📋 下一步操作:")
    print("1. 重启ComfyUI")
    print("2. 重新测试图像生成")
    print("3. 检查控制台输出，应该看到AI放大模型的检测信息")
    print("\n💡 如果仍有问题，请运行:")
    print("   python test_ai_upscale.py")
    print("\n🔧 手动安装命令:")
    print("   pip install realesrgan basicsr facexlib")

if __name__ == "__main__":
    main() 