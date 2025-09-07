#!/usr/bin/env python3
"""
🌐 Gemini Banana AI文本翻译模块
独立的翻译节点实现，支持多种免费和付费翻译服务

作者: Ken Chen
版本: 1.0.0
创建时间: 2024年
"""

import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

# 导入配置和工具函数
try:
    from .gemini_banana import (
        get_gemini_banana_config,
        generate_with_priority_api,
        extract_text_from_response,
        _log_info, _log_warning, _log_error
    )
except ImportError:
    try:
        from gemini_banana import (
            get_gemini_banana_config,
            generate_with_priority_api,
            extract_text_from_response,
            _log_info, _log_warning, _log_error
        )
    except ImportError:
        # 翻译模块使用独立模式运行
        
        def get_gemini_banana_config():
            return {}
        
        def generate_with_priority_api(*args, **kwargs):
            raise Exception("Gemini API不可用")
        
        def extract_text_from_response(response):
            return ""
        
        def _log_info(msg): print(f"[INFO] {msg}")
        def _log_warning(msg): print(f"[WARNING] {msg}")
        def _log_error(msg): print(f"[ERROR] {msg}")

# 翻译配置管理
def get_translation_config():
    """获取翻译配置"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "Translation_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            _log_warning(f"翻译配置文件不存在: {config_path}")
            return _get_default_translation_config()
    except Exception as e:
        _log_error(f"读取翻译配置文件失败: {e}")
        return _get_default_translation_config()

def _get_default_translation_config():
    """获取默认翻译配置"""
    return {
        "translation_engines": {
            "gemini-ai": {
                "enabled": True,
                "api_key": "free_service",
                "model": "gemini-2.0-flash-lite",
                "temperature": 0.3,
                "max_output_tokens": 4096
            }
        },
        "default_settings": {
            "quality_mode": "high",
            "preserve_formatting": True,
            "context_aware": True,
            "timeout": 30
        }
    }

def get_engine_config(engine_name: str) -> Optional[Dict]:
    """获取指定翻译引擎的配置"""
    config = get_translation_config()
    engines = config.get("translation_engines", {})
    return engines.get(engine_name)

def get_system_proxy():
    """自动获取系统代理设置"""
    try:
        import urllib.request
        import os

        # 方法1: 从环境变量获取
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

        if http_proxy or https_proxy:
            proxy = http_proxy or https_proxy
            _log_info(f"🔗 从环境变量获取代理: {proxy}")
            return proxy

        # 方法2: 尝试检测常见代理端口
        common_proxies = [
            "http://127.0.0.1:7897",  # Clash
            "http://127.0.0.1:7890",  # Clash
            "http://127.0.0.1:1080",  # Shadowsocks
            "http://127.0.0.1:8080",  # 通用代理
            "http://127.0.0.1:10809", # V2Ray
        ]

        for proxy in common_proxies:
            try:
                import requests
                # 快速测试代理是否可用
                response = requests.get(
                    "http://httpbin.org/ip",
                    proxies={"http": proxy, "https": proxy},
                    timeout=3
                )
                if response.status_code == 200:
                    _log_info(f"🔗 自动检测到可用代理: {proxy}")
                    return proxy
            except:
                continue

        _log_info("🔗 未检测到代理，使用直连")
        return None

    except Exception as e:
        _log_warning(f"代理检测失败: {e}")
        return None


class KenChenLLMGeminiBananaTextTranslationNode:
    """
    🌐 先进的AI文本翻译节点
    
    功能特性:
    - 支持多种先进翻译引擎（免费+付费）
    - 智能语言检测
    - 高质量神经网络翻译
    - 支持100+种语言
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        config = get_gemini_banana_config()
        
        # 支持的翻译引擎
        translation_engines = [
            "google-translate",    # Google翻译 (免费) ✅ 推荐
            "youdao-translate",    # 有道翻译 (免费) ✅ 可用
            "libre-translate",     # LibreTranslate (免费开源) ⚠️ 部分实例不稳定
            "mymemory-translate",  # MyMemory翻译 (免费)
            # "baidu-translate",     # 百度翻译 (免费) ❌ 已移除，总是失败
            "baidu-api",           # 百度翻译API (需要API) ✅ 官方API
            # "bing-translate",      # 必应翻译 (免费) ❌ API已失效，暂时移除
            "deepl-free",          # DeepL免费版
            "gemini-ai",           # Google Gemini AI翻译 (需要API)
            "openai-gpt",          # OpenAI GPT翻译 (需要API)
            "auto-best"            # 自动选择最佳引擎 ✅ 推荐
        ]
        
        # 常用语言列表
        languages = [
            "auto",                # 自动检测
            "zh-CN",              # 简体中文
            "zh-TW",              # 繁体中文
            "en",                 # 英语
            "ja",                 # 日语
            "ko",                 # 韩语
            "fr",                 # 法语
            "de",                 # 德语
            "es",                 # 西班牙语
            "it",                 # 意大利语
            "pt",                 # 葡萄牙语
            "ru",                 # 俄语
            "ar",                 # 阿拉伯语
            "hi",                 # 印地语
            "th",                 # 泰语
            "vi",                 # 越南语
            "id",                 # 印尼语
            "ms",                 # 马来语
            "tr",                 # 土耳其语
            "pl",                 # 波兰语
            "nl",                 # 荷兰语
            "sv",                 # 瑞典语
            "da",                 # 丹麦语
            "no",                 # 挪威语
            "fi",                 # 芬兰语
        ]
        
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "API密钥（仅Gemini AI/OpenAI需要，免费服务可留空）"
                }),
                "text": ("STRING", {
                    "default": "Hello, how are you today?",
                    "multiline": True,
                    "placeholder": "请输入要翻译的文本..."
                }),
                "translation_engine": (translation_engines, {"default": "google-translate"}),
                "source_language": (languages, {"default": "auto"}),
                "target_language": (languages, {"default": "zh-CN"}),
                
                # 翻译质量控制
                "quality_mode": (["standard", "high", "creative", "formal", "casual"], {"default": "high"}),
                "preserve_formatting": ("BOOLEAN", {"default": True, "label": "保持格式"}),
                "context_aware": ("BOOLEAN", {"default": True, "label": "上下文感知"}),
                
                # AI参数
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "max_output_tokens": ("INT", {"default": 4096, "min": 100, "max": 8192}),
            },
            "optional": {
                "context_info": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "上下文信息（可选，帮助提高翻译质量）"
                }),
                "custom_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义翻译指令（如：保持专业术语、口语化等）"
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("translated_text", "detected_language", "translation_info")
    FUNCTION = "translate_text"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"

    # 移除聊天记录推送功能，翻译节点只返回纯文本结果

    def translate_text(self, api_key: str, text: str, translation_engine: str, source_language: str, 
                      target_language: str, quality_mode: str, preserve_formatting: bool, 
                      context_aware: bool, temperature: float, max_output_tokens: int,
                      context_info: str = "", custom_instructions: str = "", unique_id: str = "") -> Tuple[str, str, str]:
        """使用先进AI技术进行文本翻译"""
        
        try:
            # 验证输入
            if not text.strip():
                raise ValueError("翻译文本不能为空")
            
            # 检查是否需要API密钥
            api_required_engines = ["gemini-ai", "openai-gpt"]
            
            if translation_engine in api_required_engines:
                # 需要API密钥的引擎
                if not api_key or not api_key.strip():
                    # 首先尝试从翻译引擎配置中获取API密钥
                    engine_config = get_engine_config(translation_engine)
                    if engine_config and engine_config.get("enabled", False):
                        auto_api_key = engine_config.get('api_key', '')
                        if auto_api_key and auto_api_key.strip():
                            api_key = auto_api_key.strip()
                            _log_info(f"🔑 自动使用{translation_engine}配置文件中的API密钥: {api_key[:8]}...")
                        else:
                            raise ValueError(f"引擎 {translation_engine} 需要API密钥，请在Translation_config.json中设置api_key")
                    else:
                        # 回退到旧的配置方式（兼容性）
                        config = get_gemini_banana_config()
                        auto_api_key = config.get('api_key', '')
                        if auto_api_key and auto_api_key.strip():
                            api_key = auto_api_key.strip()
                            _log_info(f"🔑 自动使用Gemini配置文件中的API密钥: {api_key[:8]}...")
                        else:
                            raise ValueError(f"引擎 {translation_engine} 需要API密钥，请在Translation_config.json中配置或手动输入")
            else:
                # 免费引擎不需要API密钥
                _log_info(f"🆓 使用免费翻译引擎: {translation_engine}")
                if not api_key or not api_key.strip():
                    api_key = "free_service"  # 占位符
            
            _log_info(f"🌐 开始翻译: {translation_engine} | {source_language} → {target_language}")
            _log_info(f"📝 原文长度: {len(text)} 字符")
            
            # 根据翻译引擎选择翻译方法
            if translation_engine == "gemini-ai":
                translated_text, detected_lang, info = self._translate_with_gemini(
                    api_key, text, source_language, target_language, quality_mode,
                    preserve_formatting, context_aware, temperature, max_output_tokens,
                    context_info, custom_instructions
                )
            elif translation_engine == "google-translate":
                translated_text, detected_lang, info = self._translate_with_google_translate(
                    text, source_language, target_language
                )
            elif translation_engine == "baidu-translate":
                translated_text, detected_lang, info = self._translate_with_baidu_translate(
                    text, source_language, target_language
                )
            elif translation_engine == "baidu-api":
                translated_text, detected_lang, info = self._translate_with_baidu_api(
                    text, source_language, target_language
                )
            elif translation_engine == "youdao-translate":
                translated_text, detected_lang, info = self._translate_with_youdao_translate(
                    text, source_language, target_language
                )
            elif translation_engine == "libre-translate":
                translated_text, detected_lang, info = self._translate_with_libre_translate(
                    text, source_language, target_language
                )
            elif translation_engine == "mymemory-translate":
                translated_text, detected_lang, info = self._translate_with_mymemory_translate(
                    text, source_language, target_language
                )
            elif translation_engine == "bing-translate":
                translated_text, detected_lang, info = self._translate_with_bing_translate(
                    text, source_language, target_language
                )
            elif translation_engine == "deepl-free":
                translated_text, detected_lang, info = self._translate_with_deepl_free(
                    text, source_language, target_language
                )
            elif translation_engine == "openai-gpt":
                translated_text, detected_lang, info = self._translate_with_openai(
                    api_key, text, source_language, target_language, quality_mode,
                    preserve_formatting, context_aware, temperature, max_output_tokens,
                    context_info, custom_instructions
                )
            elif translation_engine == "auto-best":
                # 自动选择最佳引擎（智能回退策略）
                engines_to_try = [
                    ("Google免费翻译", lambda: self._translate_with_google_translate(text, source_language, target_language)),
                    ("有道翻译", lambda: self._translate_with_youdao_translate(text, source_language, target_language)),
                    ("LibreTranslate", lambda: self._translate_with_libre_translate(text, source_language, target_language)),
                    ("MyMemory翻译", lambda: self._translate_with_mymemory_translate(text, source_language, target_language)),
                    # ("百度翻译", lambda: self._translate_with_baidu_translate(text, source_language, target_language)),  # 已移除，总是失败
                    # ("必应翻译", lambda: self._translate_with_bing_translate(text, source_language, target_language)),  # 暂时移除，API已失效
                ]

                # 如果有API密钥，添加Gemini AI作为最后的回退
                if api_key and api_key != "free_service":
                    engines_to_try.append((
                        "Gemini AI",
                        lambda: self._translate_with_gemini(
                            api_key, text, source_language, target_language, quality_mode,
                            preserve_formatting, context_aware, temperature, max_output_tokens,
                            context_info, custom_instructions
                        )
                    ))

                last_error = None
                for engine_name, engine_func in engines_to_try:
                    try:
                        translated_text, detected_lang, info = engine_func()
                        info += f" (自动选择: {engine_name})"
                        _log_info(f"✅ 自动选择成功使用: {engine_name}")
                        break
                    except Exception as e:
                        last_error = e
                        _log_warning(f"{engine_name}失败，尝试下一个引擎: {e}")
                        continue
                else:
                    # 所有引擎都失败
                    raise Exception(f"所有翻译引擎都失败，最后错误: {last_error}")
            else:
                # 默认使用Google免费翻译
                _log_warning(f"引擎 {translation_engine} 暂未实现，使用Google免费翻译")
                translated_text, detected_lang, info = self._translate_with_google_translate(
                    text, source_language, target_language
                )
                info += f" (使用Google免费翻译代替{translation_engine})"
            
            _log_info(f"✅ 翻译完成: {len(translated_text)} 字符")
            _log_info(f"🔍 检测语言: {detected_lang}")
            
            # 不推送聊天记录，直接返回翻译结果
            
            return (translated_text, detected_lang, info)
            
        except Exception as e:
            error_msg = str(e)
            _log_error(f"翻译失败: {error_msg}")
            
            # 返回错误信息
            error_translation = f"翻译失败: {error_msg}"
            return (error_translation, "unknown", f"错误: {error_msg}")

    def _translate_with_gemini(self, api_key: str, text: str, source_lang: str, target_lang: str,
                              quality_mode: str, preserve_formatting: bool, context_aware: bool,
                              temperature: float, max_output_tokens: int, context_info: str,
                              custom_instructions: str) -> Tuple[str, str, str]:
        """使用Gemini AI进行翻译"""

        # 获取Gemini AI配置
        engine_config = get_engine_config("gemini-ai")
        if not engine_config or not engine_config.get("enabled", False):
            raise Exception("Gemini AI翻译引擎未启用或配置不存在，请检查Translation_config.json")

        # 如果API密钥是免费服务标识，使用配置文件中的密钥
        if api_key == "free_service":
            api_key = engine_config.get("api_key", "")
            if not api_key or api_key == "free_service":
                raise Exception("未配置有效的Gemini API密钥，请在Translation_config.json中配置api_key")

        # 使用配置文件中的参数
        model = engine_config.get("model", "gemini-2.0-flash-lite")
        config_temp = engine_config.get("temperature", 0.3)
        config_max_tokens = engine_config.get("max_output_tokens", 4096)

        # 优先使用用户传入的参数，否则使用配置文件参数
        final_temperature = temperature if temperature != 0.3 else config_temp
        final_max_tokens = max_output_tokens if max_output_tokens != 4096 else config_max_tokens

        _log_info(f"🤖 使用Gemini AI翻译: {model} | 温度: {final_temperature} | 最大令牌: {final_max_tokens}")

        # 构建翻译提示词
        prompt = self._build_translation_prompt(
            text, source_lang, target_lang, quality_mode, preserve_formatting,
            context_aware, context_info, custom_instructions
        )

        # 构建生成配置
        generation_config = {
            "temperature": final_temperature,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": final_max_tokens,
            "responseModalities": ["TEXT"]
        }

        # 准备内容
        content_parts = [{"text": prompt}]

        # 使用配置的代理和基础URL，如果配置中没有代理则自动检测
        proxy = engine_config.get("proxy")
        if proxy is None:
            proxy = get_system_proxy()
        base_url = engine_config.get("base_url")

        # 使用nano-banana官方调用方式
        response_json = generate_with_priority_api(
            api_key=api_key,
            model=model,
            content_parts=content_parts,
            generation_config=generation_config,
            proxy=proxy,
            base_url=base_url
        )

        if not response_json:
            raise Exception("Gemini API调用失败")

        # 提取翻译结果
        translated_text = extract_text_from_response(response_json)
        if not translated_text:
            raise Exception("未能从Gemini响应中提取翻译结果")

        # 解析翻译结果
        detected_lang, final_translation = self._parse_translation_result(translated_text, source_lang)

        info = f"Gemini AI翻译 | 模型: {model} | 质量: {quality_mode} | 配置文件"
        return (final_translation, detected_lang, info)

    def _translate_with_openai(self, api_key: str, text: str, source_lang: str, target_lang: str,
                              quality_mode: str, preserve_formatting: bool, context_aware: bool,
                              temperature: float, max_output_tokens: int, context_info: str,
                              custom_instructions: str) -> Tuple[str, str, str]:
        """使用OpenAI GPT进行翻译（备用方案）"""

        # 构建翻译提示词
        prompt = self._build_translation_prompt(
            text, source_lang, target_lang, quality_mode, preserve_formatting,
            context_aware, context_info, custom_instructions
        )

        # 获取OpenAI配置
        engine_config = get_engine_config("openai-gpt")
        if not engine_config or not engine_config.get("enabled", False):
            raise Exception("OpenAI GPT翻译引擎未启用或配置不存在，请检查Translation_config.json")

        # 如果API密钥是免费服务标识，使用配置文件中的密钥
        if api_key == "free_service":
            api_key = engine_config.get("api_key", "")
            if not api_key or api_key == "free_service":
                raise Exception("未配置有效的OpenAI API密钥，请在Translation_config.json中配置api_key")

        # 使用配置文件中的参数
        model = engine_config.get("model", "gpt-4o-mini")
        config_temp = engine_config.get("temperature", 0.3)
        config_max_tokens = engine_config.get("max_tokens", 4096)
        base_url = engine_config.get("base_url", "https://api.openai.com/v1")
        proxy = engine_config.get("proxy")
        if proxy is None:
            proxy = get_system_proxy()

        # 优先使用用户传入的参数，否则使用配置文件参数
        final_temperature = temperature if temperature != 0.3 else config_temp
        final_max_tokens = max_output_tokens if max_output_tokens != 4096 else config_max_tokens

        _log_info(f"🤖 使用OpenAI GPT翻译: {model} | 温度: {final_temperature} | 最大令牌: {final_max_tokens}")

        try:
            import requests

            # 构建消息
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate accurately while preserving tone and style."},
                {"role": "user", "content": prompt}
            ]

            # 构建请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": model,
                "messages": messages,
                "temperature": final_temperature,
                "max_tokens": final_max_tokens
            }

            # 发送请求
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
                proxies={"http": proxy, "https": proxy} if proxy else None
            )
            response.raise_for_status()

            result = response.json()
            if "choices" in result and result["choices"]:
                translated_text = result["choices"][0]["message"]["content"].strip()
                detected_lang, final_translation = self._parse_translation_result(translated_text, source_lang)
                info = f"OpenAI GPT翻译 | 模型: {model} | 质量: {quality_mode} | 配置文件"
                return (final_translation, detected_lang, info)
            else:
                raise Exception("OpenAI API返回格式错误")

        except Exception as e:
            _log_error(f"OpenAI GPT翻译失败: {e}")
            raise Exception(f"OpenAI GPT翻译失败: {e}")

    def _translate_with_google_translate(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """使用Google免费翻译服务"""
        try:
            import requests
            import urllib.parse
            import json

            _log_info("🌐 使用Google免费翻译服务...")

            # 语言代码转换
            lang_map = {
                "zh-CN": "zh", "zh-TW": "zh-tw", "auto": "auto"
            }
            src_lang = lang_map.get(source_lang, source_lang)
            tgt_lang = lang_map.get(target_lang, target_lang)

            # Google翻译API URL
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": src_lang,
                "tl": tgt_lang,
                "dt": "t",
                "q": text
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            # 解析响应
            result = response.json()
            if result and len(result) > 0 and result[0]:
                translated_parts = []
                for part in result[0]:
                    if part and len(part) > 0:
                        translated_parts.append(part[0])

                translated_text = "".join(translated_parts)
                detected_lang = result[2] if len(result) > 2 else source_lang

                info = "Google免费翻译服务"
                return (translated_text, detected_lang, info)
            else:
                raise Exception("Google翻译返回空结果")

        except Exception as e:
            _log_error(f"Google免费翻译失败: {e}")
            raise Exception(f"Google免费翻译失败: {e}")

    def _translate_with_baidu_translate(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """使用百度免费翻译服务（改进版）"""
        try:
            import requests
            import json
            import time
            import hashlib
            import random

            _log_info("🌐 使用百度免费翻译服务...")

            # 语言代码转换
            lang_map = {
                "zh-CN": "zh", "zh-TW": "cht", "en": "en", "ja": "jp", "ko": "kor",
                "fr": "fra", "de": "de", "es": "spa", "it": "it", "pt": "pt",
                "ru": "ru", "ar": "ara", "auto": "auto"
            }
            src_lang = lang_map.get(source_lang, "auto")
            tgt_lang = lang_map.get(target_lang, "zh")

            # 尝试多种百度翻译方法
            methods = [
                self._baidu_method_1,
                self._baidu_method_2,
                self._baidu_method_3,
                self._baidu_method_4
            ]

            for i, method in enumerate(methods):
                try:
                    result = method(text, src_lang, tgt_lang)
                    if result:
                        translated_text, detected_lang = result
                        info = f"百度免费翻译服务 (方法{i+1})"
                        return (translated_text, detected_lang, info)
                except Exception as e:
                    _log_warning(f"百度翻译方法{i+1}失败: {e}")
                    continue

            # 所有接口都失败，抛出异常
            raise Exception("所有百度翻译接口都无法访问，建议使用baidu-api")

        except Exception as e:
            _log_error(f"百度免费翻译失败: {e}")
            raise Exception(f"百度免费翻译失败: {e}，建议使用baidu-api或其他翻译引擎")

    def _translate_with_baidu_api(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """使用百度翻译API（官方API）"""
        try:
            import requests
            import hashlib
            import random

            # 获取百度API配置
            engine_config = get_engine_config("baidu-api")
            if not engine_config or not engine_config.get("enabled", False):
                raise Exception("百度翻译API未启用或配置不存在，请检查Translation_config.json")

            app_id = engine_config.get("app_id", "")
            api_key = engine_config.get("api_key", "")
            base_url = engine_config.get("base_url", "https://fanyi-api.baidu.com/api/trans/vip/translate")
            proxy = engine_config.get("proxy")
            if proxy is None:
                proxy = get_system_proxy()

            if not app_id or not api_key:
                raise Exception("未配置有效的百度翻译API密钥，请在Translation_config.json中配置app_id和api_key")

            _log_info(f"🤖 使用百度翻译API: APPID={app_id[:8]}...")

            # 语言代码转换
            config = get_translation_config()
            lang_mappings = config.get("language_mappings", {}).get("baidu-api", {})
            from_lang = lang_mappings.get(source_lang, source_lang.replace("-", "").lower())
            to_lang = lang_mappings.get(target_lang, target_lang.replace("-", "").lower())

            # 生成签名
            salt = str(random.randint(32768, 65536))
            sign_str = app_id + text + salt + api_key
            sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()

            # 构建请求参数
            params = {
                'q': text,
                'from': from_lang,
                'to': to_lang,
                'appid': app_id,
                'salt': salt,
                'sign': sign
            }

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # 发送请求
            response = requests.post(
                base_url,
                data=params,
                headers=headers,
                timeout=30,
                proxies={"http": proxy, "https": proxy} if proxy else None
            )
            response.raise_for_status()

            result = response.json()

            # 检查错误
            if 'error_code' in result:
                error_code = result['error_code']
                error_msg = result.get('error_msg', f'错误代码: {error_code}')
                raise Exception(f"百度翻译API错误: {error_msg}")

            # 提取翻译结果
            if 'trans_result' in result and result['trans_result']:
                translated_text = result['trans_result'][0]['dst']
                info = f"百度翻译API | APPID: {app_id[:8]}... | 配置文件"
                return (translated_text, source_lang, info)
            else:
                raise Exception("百度翻译API返回格式错误")

        except Exception as e:
            _log_error(f"百度翻译API失败: {e}")
            raise Exception(f"百度翻译API失败: {e}")

    def _baidu_method_1(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        """百度翻译方法1: 新版API"""
        import requests
        import json

        url = "https://fanyi.baidu.com/ait/text/translate"
        data = {
            "from": src_lang,
            "to": tgt_lang,
            "query": text,
            "source": "txt"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://fanyi.baidu.com/",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Origin": "https://fanyi.baidu.com"
        }

        response = requests.post(url, data=data, headers=headers, timeout=15)
        response.raise_for_status()

        result = response.json()
        if "data" in result and isinstance(result["data"], list):
            for item in result["data"]:
                if "dst" in item:
                    return (item["dst"], src_lang)

        raise Exception("方法1响应格式错误")

    def _baidu_method_2(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        """百度翻译方法2: 传统API"""
        import requests
        import random

        url = "https://fanyi.baidu.com/v2transapi"
        data = {
            "from": src_lang,
            "to": tgt_lang,
            "query": text,
            "simple_means_flag": "3",
            "sign": str(random.randint(100000, 999999)),
            "token": "",
            "domain": "common"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://fanyi.baidu.com/",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Cookie": "BAIDUID=test"
        }

        response = requests.post(url, data=data, headers=headers, timeout=15)
        response.raise_for_status()

        result = response.json()
        if "trans_result" in result and "data" in result["trans_result"]:
            translated_parts = []
            for item in result["trans_result"]["data"]:
                if "dst" in item:
                    translated_parts.append(item["dst"])
            if translated_parts:
                detected_lang = result["trans_result"].get("from", src_lang)
                return ("".join(translated_parts), detected_lang)

        raise Exception("方法2响应格式错误")

    def _baidu_method_3(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        """百度翻译方法3: 移动端API"""
        import requests
        import urllib.parse

        encoded_text = urllib.parse.quote(text)
        url = f"https://fanyi.baidu.com/mtpe-individual/multimodal?query={encoded_text}&from={src_lang}&to={tgt_lang}"

        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Referer": "https://fanyi.baidu.com/",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        result = response.json()
        if "data" in result and "trans_result" in result["data"]:
            trans_result = result["data"]["trans_result"]
            if isinstance(trans_result, list) and trans_result:
                for item in trans_result:
                    if "dst" in item:
                        return (item["dst"], src_lang)

        raise Exception("方法3响应格式错误")

    def _baidu_method_4(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        """百度翻译方法4: 简化接口"""
        import requests
        import time
        import hashlib

        # 生成简单的签名
        ts = str(int(time.time()))
        salt = ts
        sign = hashlib.md5(f"20150320{text}{salt}2f4e00d79b1bd3a8".encode()).hexdigest()

        url = "https://fanyi.baidu.com/langdetect"
        data = {
            "query": text
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://fanyi.baidu.com/"
        }

        # 先检测语言
        response = requests.post(url, data=data, headers=headers, timeout=10)

        # 然后进行翻译
        url = "https://fanyi.baidu.com/basetrans"
        data = {
            "from": src_lang,
            "to": tgt_lang,
            "query": text,
            "sign": sign,
            "salt": salt
        }

        response = requests.post(url, data=data, headers=headers, timeout=15)
        response.raise_for_status()

        result = response.json()
        if "trans" in result and isinstance(result["trans"], list):
            for item in result["trans"]:
                if "dst" in item:
                    return (item["dst"], src_lang)

        raise Exception("方法4响应格式错误")

    def _translate_with_libre_translate(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """使用LibreTranslate开源翻译服务"""
        try:
            import requests
            import json

            _log_info("🌐 使用LibreTranslate开源翻译服务...")

            # 语言代码转换
            lang_map = {
                "zh-CN": "zh", "zh-TW": "zh", "auto": "auto",
                "en": "en", "ja": "ja", "ko": "ko", "fr": "fr", "de": "de",
                "es": "es", "it": "it", "pt": "pt", "ru": "ru", "ar": "ar"
            }
            src_lang = lang_map.get(source_lang, "auto")
            tgt_lang = lang_map.get(target_lang, "zh")

            # 尝试多个LibreTranslate实例（更新可用实例列表）
            instances = [
                # 优先使用稳定的实例
                ("https://libretranslate.de/translate", "LibreTranslate.de"),
                ("https://translate.terraprint.co/translate", "TerraPrint"),
                ("https://translate.fedilab.app/translate", "FediLab"),
                ("https://translate.astian.org/translate", "Astian"),
                # 备用实例（可能需要特殊处理）
                ("https://libretranslate.com/translate", "LibreTranslate.com"),
            ]

            last_error = None
            for i, (url, name) in enumerate(instances):
                try:
                    data = {
                        "q": text,
                        "source": src_lang,
                        "target": tgt_lang,
                        "format": "text"
                    }

                    headers = {
                        "Content-Type": "application/json",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }

                    # 对于某些实例，禁用SSL验证以避免SSL错误
                    verify_ssl = True
                    if "argosopentech" in url or "libretranslate.com" in url:
                        verify_ssl = False

                    response = requests.post(
                        url,
                        json=data,
                        headers=headers,
                        timeout=15,
                        verify=verify_ssl
                    )
                    response.raise_for_status()

                    result = response.json()
                    if "translatedText" in result:
                        translated_text = result["translatedText"]
                        info = f"LibreTranslate开源翻译服务 ({name})"
                        _log_info(f"✅ LibreTranslate成功使用: {name}")
                        return (translated_text, src_lang, info)

                except Exception as e:
                    last_error = e
                    _log_warning(f"LibreTranslate实例 {name} 失败: {e}")
                    continue

            # 如果所有实例都失败，提供更友好的错误信息
            error_msg = f"所有LibreTranslate实例都无法访问，最后错误: {last_error}"
            if "SSL" in str(last_error):
                error_msg += "。建议使用其他翻译引擎如Google翻译或有道翻译。"
            raise Exception(error_msg)

        except Exception as e:
            error_msg = f"LibreTranslate翻译失败: {e}"
            if "所有LibreTranslate实例都无法访问" in str(e):
                error_msg += "，如果确实不能用，就将它从列表中清除"
            _log_error(error_msg)
            raise Exception(error_msg)

    def _translate_with_mymemory_translate(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """使用MyMemory免费翻译服务"""
        try:
            import requests
            import urllib.parse

            _log_info("🌐 使用MyMemory免费翻译服务...")

            # 语言代码转换
            lang_map = {
                "zh-CN": "zh-CN", "zh-TW": "zh-TW", "auto": "auto",
                "en": "en", "ja": "ja", "ko": "ko", "fr": "fr", "de": "de",
                "es": "es", "it": "it", "pt": "pt", "ru": "ru", "ar": "ar"
            }
            src_lang = lang_map.get(source_lang, "auto")
            tgt_lang = lang_map.get(target_lang, "zh-CN")

            # MyMemory API
            encoded_text = urllib.parse.quote(text)
            url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair={src_lang}|{tgt_lang}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            result = response.json()
            if "responseData" in result and result["responseData"]:
                translated_text = result["responseData"]["translatedText"]

                # 检查翻译质量
                if "matches" in result and result["matches"]:
                    # 如果有更好的匹配，使用第一个匹配
                    best_match = result["matches"][0]
                    quality = best_match.get("quality", 0)
                    # 确保quality是数字类型
                    try:
                        quality = float(quality) if quality else 0
                        if quality > 70:  # 质量阈值
                            translated_text = best_match["translation"]
                    except (ValueError, TypeError):
                        pass  # 如果quality不是数字，跳过

                info = "MyMemory免费翻译服务"
                return (translated_text, src_lang, info)
            else:
                raise Exception("MyMemory翻译返回格式错误")

        except Exception as e:
            _log_error(f"MyMemory翻译失败: {e}")
            raise Exception(f"MyMemory翻译失败: {e}")

    def _translate_with_youdao_translate(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """使用有道免费翻译服务（改进版）"""
        try:
            import requests
            import json
            import time
            import hashlib
            import random

            _log_info("🌐 使用有道免费翻译服务...")

            # 语言代码转换
            lang_map = {
                "zh-CN": "zh-CHS", "zh-TW": "zh-CHT", "en": "en", "ja": "ja", "ko": "ko",
                "fr": "fr", "de": "de", "es": "es", "it": "it", "pt": "pt", "ru": "ru", "auto": "auto"
            }
            src_lang = lang_map.get(source_lang, "auto")
            tgt_lang = lang_map.get(target_lang, "zh-CHS")

            # 生成时间戳和随机数
            ts = str(int(time.time() * 1000))
            salt = str(random.randint(1, 65536))

            # 尝试多个有道翻译接口
            urls_to_try = [
                "https://fanyi.youdao.com/translate_o",
                "https://aidemo.youdao.com/trans"
            ]

            for i, url in enumerate(urls_to_try):
                try:
                    if i == 0:  # 原接口
                        data = {
                            "i": text,
                            "from": src_lang,
                            "to": tgt_lang,
                            "smartresult": "dict",
                            "client": "fanyideskweb",
                            "salt": salt,
                            "sign": hashlib.md5((f"fanyideskweb{text}{salt}Ygy_4c=r#e#4EX^NUGUc5").encode()).hexdigest(),
                            "lts": ts,
                            "bv": "bd285208169a4c0c5fca9d4e5c7b8c4c",
                            "doctype": "json",
                            "version": "2.1",
                            "keyfrom": "fanyi.web",
                            "action": "FY_BY_REALTlME"
                        }
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                            "Referer": "https://fanyi.youdao.com/",
                            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                            "Accept": "application/json, text/javascript, */*; q=0.01",
                            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
                        }
                    else:  # 备用接口
                        data = {
                            "q": text,
                            "from": src_lang,
                            "to": tgt_lang
                        }
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                            "Content-Type": "application/x-www-form-urlencoded"
                        }

                    response = requests.post(url, data=data, headers=headers, timeout=15)
                    response.raise_for_status()

                    result = response.json()

                    # 尝试解析不同的响应格式
                    translated_text = None
                    detected_lang = source_lang

                    # 格式1: 标准有道格式
                    if "translateResult" in result and result["translateResult"]:
                        translated_parts = []
                        for group in result["translateResult"]:
                            if isinstance(group, list):
                                for item in group:
                                    if isinstance(item, dict) and "tgt" in item:
                                        translated_parts.append(item["tgt"])
                        translated_text = "".join(translated_parts)

                    # 格式2: 简化格式
                    elif "translation" in result:
                        if isinstance(result["translation"], list):
                            translated_text = "".join(result["translation"])
                        else:
                            translated_text = str(result["translation"])

                    if translated_text:
                        info = f"有道免费翻译服务 (接口{i+1})"
                        return (translated_text, detected_lang, info)

                except Exception as e:
                    _log_warning(f"有道翻译接口{i+1}失败: {e}")
                    continue

            # 所有接口都失败，抛出异常
            raise Exception("所有有道翻译接口都无法访问")

        except Exception as e:
            _log_error(f"有道免费翻译失败: {e}")
            raise Exception(f"有道免费翻译失败: {e}")

    def _translate_with_bing_translate(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """使用必应免费翻译服务（已废弃）"""
        _log_warning("⚠️ 必应翻译API已失效，建议使用其他翻译引擎")
        raise Exception("必应翻译API已失效，请使用Google翻译、有道翻译或自动最佳模式")







    def _translate_with_deepl_free(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """使用DeepL免费翻译服务（改进版）"""
        try:
            import requests
            import json
            import time
            import random

            _log_info("🌐 使用DeepL免费翻译服务...")

            # DeepL语言代码
            lang_map = {
                "zh-CN": "ZH", "zh-TW": "ZH", "en": "EN", "ja": "JA", "ko": "KO",
                "fr": "FR", "de": "DE", "es": "ES", "it": "IT", "pt": "PT", "ru": "RU", "auto": "AUTO"
            }
            src_lang = lang_map.get(source_lang, "AUTO")
            tgt_lang = lang_map.get(target_lang, "ZH")

            # 尝试多种DeepL方法
            methods = [
                self._deepl_method_1,
                self._deepl_method_2,
                self._deepl_method_3
            ]

            for i, method in enumerate(methods):
                try:
                    result = method(text, src_lang, tgt_lang)
                    if result:
                        translated_text, detected_lang = result
                        info = f"DeepL免费翻译服务 (方法{i+1})"
                        return (translated_text, detected_lang, info)
                except Exception as e:
                    _log_warning(f"DeepL方法{i+1}失败: {e}")
                    # 如果是429错误，等待一下再试下一个方法
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        wait_time = random.uniform(1, 3)
                        _log_info(f"遇到限流，等待{wait_time:.1f}秒...")
                        time.sleep(wait_time)
                    continue

            raise Exception("所有DeepL翻译方法都失败")

        except Exception as e:
            _log_error(f"DeepL免费翻译失败: {e}")
            raise Exception(f"DeepL免费翻译失败: {e}")

    def _deepl_method_1(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        """DeepL方法1: 标准API"""
        import requests
        import time
        import random

        # 生成随机时间戳
        timestamp = int(time.time() * 1000)

        url = "https://www2.deepl.com/jsonrpc"
        data = {
            "jsonrpc": "2.0",
            "method": "LMT_handle_jobs",
            "params": {
                "jobs": [{
                    "kind": "default",
                    "sentences": [{"text": text, "id": 1, "prefix": ""}],
                    "raw_en_context_before": [],
                    "raw_en_context_after": [],
                    "preferred_num_beams": 4
                }],
                "lang": {
                    "source_lang_user_selected": src_lang,
                    "target_lang": tgt_lang
                },
                "priority": -1,
                "commonJobParams": {},
                "timestamp": timestamp
            },
            "id": random.randint(1000, 9999)
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.deepl.com/translator",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Origin": "https://www.deepl.com"
        }

        response = requests.post(url, json=data, headers=headers, timeout=20)
        response.raise_for_status()

        result = response.json()
        if "result" in result and "translations" in result["result"]:
            translations = result["result"]["translations"]
            if translations and len(translations) > 0:
                translated_text = translations[0]["beams"][0]["sentences"][0]["text"]
                detected_lang = result["result"]["source_lang"]
                return (translated_text, detected_lang)

        raise Exception("方法1响应格式错误")

    def _deepl_method_2(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        """DeepL方法2: 备用接口"""
        import requests
        import urllib.parse

        # 使用GET方式的简化接口
        encoded_text = urllib.parse.quote(text)
        url = f"https://api-free.deepl.com/v2/translate?auth_key=free&text={encoded_text}&source_lang={src_lang}&target_lang={tgt_lang}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=20)

        # 如果是401错误，说明需要API密钥，跳过这个方法
        if response.status_code == 401:
            raise Exception("需要API密钥")

        response.raise_for_status()

        result = response.json()
        if "translations" in result and result["translations"]:
            translated_text = result["translations"][0]["text"]
            detected_lang = result["translations"][0].get("detected_source_language", src_lang)
            return (translated_text, detected_lang)

        raise Exception("方法2响应格式错误")

    def _deepl_method_3(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        """DeepL方法3: 第三方代理"""
        import requests

        # 使用第三方DeepL代理
        url = "https://deeplx.mingming.dev/translate"

        data = {
            "text": text,
            "source_lang": src_lang,
            "target_lang": tgt_lang
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=data, headers=headers, timeout=20)
        response.raise_for_status()

        result = response.json()
        if "data" in result:
            translated_text = result["data"]
            return (translated_text, src_lang)
        elif "text" in result:
            translated_text = result["text"]
            return (translated_text, src_lang)

        raise Exception("方法3响应格式错误")

    def _build_translation_prompt(self, text: str, source_lang: str, target_lang: str,
                                 quality_mode: str, preserve_formatting: bool, context_aware: bool,
                                 context_info: str, custom_instructions: str) -> str:
        """构建翻译提示词"""

        # 语言代码映射
        lang_names = {
            "auto": "自动检测",
            "zh-CN": "简体中文", "zh-TW": "繁体中文", "en": "英语", "ja": "日语", "ko": "韩语",
            "fr": "法语", "de": "德语", "es": "西班牙语", "it": "意大利语", "pt": "葡萄牙语",
            "ru": "俄语", "ar": "阿拉伯语", "hi": "印地语", "th": "泰语", "vi": "越南语",
            "id": "印尼语", "ms": "马来语", "tr": "土耳其语", "pl": "波兰语", "nl": "荷兰语"
        }

        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        # 质量模式说明
        quality_instructions = {
            "standard": "提供准确、流畅的翻译",
            "high": "提供高质量、自然、地道的翻译，注意语言的细微差别",
            "creative": "提供创意性翻译，可以适当意译以保持原文的风格和情感",
            "formal": "提供正式、专业的翻译，适合商务或学术场合",
            "casual": "提供口语化、轻松的翻译，适合日常交流"
        }

        prompt = f"""你是一个专业的AI翻译专家，精通多种语言的翻译工作。

翻译任务:
- 源语言: {source_name}
- 目标语言: {target_name}
- 质量要求: {quality_instructions.get(quality_mode, '标准翻译')}

翻译要求:
1. 准确传达原文的意思和语调
2. 保持翻译的自然流畅
3. 注意文化差异和语言习惯
"""

        if preserve_formatting:
            prompt += "4. 保持原文的格式和结构\n"

        if context_aware:
            prompt += "5. 根据上下文选择最合适的翻译\n"

        if context_info.strip():
            prompt += f"\n上下文信息:\n{context_info.strip()}\n"

        if custom_instructions.strip():
            prompt += f"\n特殊要求:\n{custom_instructions.strip()}\n"

        prompt += f"""
请翻译以下文本，只返回翻译结果，不要添加任何解释或说明:

原文:
{text}

翻译结果:"""

        return prompt

    def _parse_translation_result(self, response: str, source_lang: str) -> Tuple[str, str]:
        """解析翻译结果"""

        # 清理响应文本
        translation = response.strip()

        # 移除可能的前缀
        prefixes_to_remove = [
            "翻译结果:", "Translation:", "翻译:", "Result:", "译文:",
            "Translated text:", "Translation result:"
        ]

        for prefix in prefixes_to_remove:
            if translation.startswith(prefix):
                translation = translation[len(prefix):].strip()

        # 检测源语言（简单检测）
        detected_lang = source_lang if source_lang != "auto" else self._detect_language(translation)

        return (detected_lang, translation)

    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""

        # 基于字符特征的简单检测
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh-CN"  # 中文
        elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return "ja"     # 日文
        elif any('\uac00' <= char <= '\ud7af' for char in text):
            return "ko"     # 韩文
        elif any('\u0600' <= char <= '\u06ff' for char in text):
            return "ar"     # 阿拉伯文
        elif any('\u0400' <= char <= '\u04ff' for char in text):
            return "ru"     # 俄文
        else:
            return "en"     # 默认英文


# 节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiBananaTextTranslation": KenChenLLMGeminiBananaTextTranslationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiBananaTextTranslation": "🌐 Gemini Banana AI Text Translation",
}

# 导出
__all__ = [
    "KenChenLLMGeminiBananaTextTranslationNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
