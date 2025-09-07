# LLM Agent Assistant Plugin for ComfyUI
# 作者: Ken-Chen
# 版本: 1.0.0

import importlib
import sys
import os

# 合并所有节点的映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Delay module import to avoid startup import errors
def load_modules():
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 尝试导入GLM模块
    try:
        # Try relative import
        try:
            from . import glm
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            sys.path.insert(0, current_dir)
            glm = importlib.import_module('glm')
            
        if hasattr(glm, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(glm.NODE_CLASS_MAPPINGS)
        if hasattr(glm, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(glm.NODE_DISPLAY_NAME_MAPPINGS)
        print("[LLM Prompt] GLM module loaded successfully")
    except Exception as e:
        print(f"[LLM Prompt] GLM module loading failed: {e}")

    # Try to import Comfly module
    try:
        # Try relative import
        try:
            from . import comfly
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            comfly = importlib.import_module('comfly')
            
        if hasattr(comfly, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(comfly.NODE_CLASS_MAPPINGS)
        if hasattr(comfly, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(comfly.NODE_DISPLAY_NAME_MAPPINGS)
        print("[LLM Prompt] Comfly module loaded successfully")
    except Exception as e:
        print(f"[LLM Prompt] Comfly module loading failed: {e}")

    # Try to import JoyCaption module
    try:
        # Try relative import
        try:
            from . import joycaption
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            joycaption = importlib.import_module('joycaption')
            
        if hasattr(joycaption, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(joycaption.NODE_CLASS_MAPPINGS)
        if hasattr(joycaption, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(joycaption.NODE_DISPLAY_NAME_MAPPINGS)
        print("[LLM Prompt] JoyCaption module loaded successfully")
    except Exception as e:
        print(f"[LLM Prompt] JoyCaption module loading failed: {e}")

    # Try to import Gemini module
    try:
        # Try relative import
        try:
            from . import gemini
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            gemini = importlib.import_module('gemini')
            
        if hasattr(gemini, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(gemini.NODE_CLASS_MAPPINGS)
        if hasattr(gemini, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(gemini.NODE_DISPLAY_NAME_MAPPINGS)
        print("[LLM Prompt] Gemini module loaded successfully")
    except Exception as e:
        print(f"[LLM Prompt] Gemini module loading failed: {e}")

    # Try to import Gemini Banana module
    try:
        # Try relative import
        try:
            from . import gemini_banana
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            gemini_banana = importlib.import_module('gemini_banana')
            
        if hasattr(gemini_banana, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(gemini_banana.NODE_CLASS_MAPPINGS)
        if hasattr(gemini_banana, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(gemini_banana.NODE_DISPLAY_NAME_MAPPINGS)
        print("[LLM Prompt] Gemini Banana module loaded successfully")
    except Exception as e:
        print(f"[LLM Prompt] Gemini Banana module loading failed: {e}")

    # Try to import Gemini Banana Mirror module
    try:
        # Try relative import
        try:
            from . import gemini_banana_mirror
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            gemini_banana_mirror = importlib.import_module('gemini_banana_mirror')
            
        if hasattr(gemini_banana_mirror, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(gemini_banana_mirror.NODE_CLASS_MAPPINGS)
        if hasattr(gemini_banana_mirror, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(gemini_banana_mirror.NODE_DISPLAY_NAME_MAPPINGS)
        print("[LLM Prompt] Gemini Banana Mirror module loaded successfully")
    except Exception as e:
        print(f"[LLM Prompt] Gemini Banana Mirror module loading failed: {e}")

    # Try to import OpenRouter Banana module
    try:
        # Try relative import
        try:
            from . import openrouter_banana
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            openrouter_banana = importlib.import_module('openrouter_banana')
            
        if hasattr(openrouter_banana, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(openrouter_banana.NODE_CLASS_MAPPINGS)
        if hasattr(openrouter_banana, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(openrouter_banana.NODE_DISPLAY_NAME_MAPPINGS)
        print("[LLM Prompt] OpenRouter Banana module loaded successfully")
    except Exception as e:
        print(f"[LLM Prompt] OpenRouter Banana module loading failed: {e}")

    # Try to import Comfly Nano Banana module
    try:
        # Try relative import
        try:
            from . import comfly_nano_banana
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            comfly_nano_banana = importlib.import_module('comfly_nano_banana')
            
        if hasattr(comfly_nano_banana, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(comfly_nano_banana.NODE_CLASS_MAPPINGS)
        if hasattr(comfly_nano_banana, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(comfly_nano_banana.NODE_DISPLAY_NAME_MAPPINGS)
        print("[LLM Prompt] Comfly Nano Banana module loaded successfully")
    except Exception as e:
        print(f"[LLM Prompt] Comfly Nano Banana module loading failed: {e}")





# Load modules immediately
load_modules()

print("[LLM Agent Assistant] Plugin loading completed!")
print(f"[LLM Agent Assistant] Registered {len(NODE_CLASS_MAPPINGS)} nodes")
print(f"[LLM Agent Assistant] Node list: {list(NODE_CLASS_MAPPINGS.keys())}")
