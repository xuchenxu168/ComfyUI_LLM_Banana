# LLM Agent Assistant Plugin for ComfyUI
# 作者: Ken-Chen
# 版本: 1.0.0

import importlib
import sys
import os

# 合并所有节点的映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 设置Web目录以加载前端资源
WEB_DIRECTORY = "./web"

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
    except Exception as e:
        pass

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
    except Exception as e:
        pass

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
    except Exception as e:
        pass

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
    except Exception as e:
        pass

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
    except Exception as e:
        pass

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
    except Exception as e:
        pass

    # Try to import Image Collage module
    try:
        # Try relative import
        try:
            from . import image_collage_node
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            image_collage_node = importlib.import_module('image_collage_node')

        if hasattr(image_collage_node, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(image_collage_node.NODE_CLASS_MAPPINGS)
        if hasattr(image_collage_node, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(image_collage_node.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        pass


    # Try to import General API module
    try:
        # Try relative import
        try:
            from . import general_api
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            general_api = importlib.import_module('general_api')

        if hasattr(general_api, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(general_api.NODE_CLASS_MAPPINGS)
        if hasattr(general_api, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(general_api.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        pass

    # Note: openrouter_banana and comfly_nano_banana modules are deprecated
    # Their functionality has been integrated into gemini_banana_mirror.py





# Load modules immediately
load_modules()

# 导出WEB_DIRECTORY供ComfyUI使用
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
