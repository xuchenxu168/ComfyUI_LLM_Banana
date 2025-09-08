#!/usr/bin/env python3
"""
OpenRouter API 诊断工具
用于检查OpenRouter API密钥和账户状态
"""

import requests
import json
import sys
import os

def test_openrouter_api(api_key: str):
    """测试OpenRouter API连接和账户状态"""
    
    print("🔍 OpenRouter API 诊断工具")
    print("=" * 50)
    
    # 1. 检查API密钥格式
    print(f"🔑 API密钥格式检查:")
    if not api_key:
        print("❌ API密钥为空")
        return False
    
    if not api_key.startswith("sk-or-v1-"):
        print(f"⚠️ API密钥格式可能不正确: {api_key[:20]}...")
        print("   正确格式应该是: sk-or-v1-xxxxxxxx")
    else:
        print(f"✅ API密钥格式正确: {api_key[:20]}...")
    
    # 2. 测试账户信息
    print(f"\n💰 账户信息检查:")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 获取账户余额
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers=headers,
            timeout=10
        )
        
        print(f"📡 状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API密钥有效")
            print(f"📊 账户信息: {json.dumps(data, indent=2, ensure_ascii=False)}")
        elif response.status_code == 401:
            error_data = response.json()
            print("❌ 认证失败 (401)")
            print(f"🔍 错误详情: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            
            if "User not found" in str(error_data):
                print("\n💡 可能的解决方案:")
                print("   1. 检查API密钥是否正确复制")
                print("   2. 确认OpenRouter账户状态正常")
                print("   3. 检查账户是否有足够余额")
                print("   4. 尝试重新生成API密钥")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            try:
                error_data = response.json()
                print(f"🔍 错误详情: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"🔍 响应内容: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求失败: {e}")
        return False
    
    # 3. 测试模型列表
    print(f"\n🤖 模型列表检查:")
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 成功获取 {len(models.get('data', []))} 个模型")
            
            # 显示一些热门模型
            popular_models = [
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-2.0-flash-exp",
                "meta-llama/llama-3.2-90b-vision-instruct"
            ]
            
            available_models = [model['id'] for model in models.get('data', [])]
            print("\n🔥 热门模型可用性:")
            for model in popular_models:
                if model in available_models:
                    print(f"   ✅ {model}")
                else:
                    print(f"   ❌ {model}")
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 获取模型列表失败: {e}")
    
    # 4. 测试简单的chat请求
    print(f"\n💬 Chat API 测试:")
    try:
        test_data = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=30
        )
        
        print(f"📡 状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Chat API 测试成功")
        else:
            print(f"❌ Chat API 测试失败: {response.status_code}")
            try:
                error_data = response.json()
                print(f"🔍 错误详情: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"🔍 响应内容: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Chat API 测试失败: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 诊断完成")
    
    return True

def main():
    """主函数"""
    
    # 从环境变量或命令行参数获取API密钥
    api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if not api_key:
        print("请提供OpenRouter API密钥:")
        print("方法1: python openrouter_diagnostic.py sk-or-v1-your-key-here")
        print("方法2: 设置环境变量 OPENROUTER_API_KEY")
        return
    
    test_openrouter_api(api_key)

if __name__ == "__main__":
    main()
