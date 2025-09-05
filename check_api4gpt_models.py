#!/usr/bin/env python3
"""
检查API4GPT可用模型的脚本
"""

import requests
import json

def check_api4gpt_models():
    """检查API4GPT的可用模型"""
    
    base_url = "https://www.api4gpt.com"
    api_key = "sk-7vQ50rx6H3nYp5g0lBK81oJNZCcqAnJeNgOI6AEUEZncsOGB"
    
    print("🔍 检查API4GPT的可用模型...")
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            if "data" in result and result["data"]:
                print(f"✅ 找到 {len(result['data'])} 个模型")
                
                # 按类型分组模型
                model_types = {}
                for model in result["data"]:
                    model_id = model.get("id", "Unknown")
                    model_type = model_id.split("-")[0] if "-" in model_id else "other"
                    
                    if model_type not in model_types:
                        model_types[model_type] = []
                    model_types[model_type].append(model_id)
                
                # 显示分组结果
                for model_type, models in model_types.items():
                    print(f"\n📋 {model_type.upper()} 类型模型 ({len(models)} 个):")
                    for model in models[:10]:  # 只显示前10个
                        print(f"  - {model}")
                    if len(models) > 10:
                        print(f"  ... 还有 {len(models) - 10} 个模型")
                
                # 查找可能的图像生成模型
                print(f"\n🔍 查找可能的图像生成模型...")
                image_models = []
                for model in result["data"]:
                    model_id = model.get("id", "").lower()
                    if any(keyword in model_id for keyword in ["dall", "stable", "diffusion", "flux", "image", "generation"]):
                        image_models.append(model["id"])
                
                if image_models:
                    print(f"✅ 找到 {len(image_models)} 个可能的图像生成模型:")
                    for model in image_models:
                        print(f"  - {model}")
                else:
                    print("⚠️ 未找到明显的图像生成模型")
                
                # 查找可能的聊天模型
                print(f"\n🔍 查找可能的聊天模型...")
                chat_models = []
                for model in result["data"]:
                    model_id = model.get("id", "").lower()
                    if any(keyword in model_id for keyword in ["gpt", "claude", "gemini", "llama", "chat", "completion"]):
                        chat_models.append(model["id"])
                
                if chat_models:
                    print(f"✅ 找到 {len(chat_models)} 个可能的聊天模型:")
                    for model in chat_models[:10]:  # 只显示前10个
                        print(f"  - {model}")
                    if len(chat_models) > 10:
                        print(f"  ... 还有 {len(chat_models) - 10} 个模型")
                else:
                    print("⚠️ 未找到明显的聊天模型")
                
            else:
                print("❌ 响应中没有模型数据")
                print(f"📋 响应结构: {list(result.keys())}")
                
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"📋 错误响应: {response.text}")
            
    except Exception as e:
        print(f"❌ 检查模型失败: {e}")

if __name__ == "__main__":
    print("🚀 API4GPT 模型检查工具")
    print("=" * 50)
    
    check_api4gpt_models()
    
    print("\n" + "=" * 50)
    print("🏁 检查完成！") 