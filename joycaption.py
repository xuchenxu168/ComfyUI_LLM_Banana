import logging
import os
from pathlib import Path
import numpy as np
import torch
import uuid
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import Union

# 日志函数
def _log_info(message):
    print(f"[LLM Agent Assistant] {message}")

def _log_warning(message):
    print(f"[LLM Agent Assistant] WARNING: {message}")

def _log_error(message):
    print(f"[LLM Agent Assistant] ERROR: {message}")

# 延迟导入ComfyUI模块
def get_comfyui_modules():
    try:
        import folder_paths
        import model_management
        return folder_paths, model_management
    except ImportError:
        _log_warning("ComfyUI模块未找到，使用默认配置")
        return None, None

# 获取ComfyUI模块
folder_paths, model_management = get_comfyui_modules()

if folder_paths and model_management:
    models_dir = folder_paths.models_dir
    model_path = os.path.join(models_dir, "LLM")
    device = model_management.get_torch_device()
else:
    models_dir = "models"
    model_path = os.path.join(models_dir, "LLM")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

# PIL.Image.MAX_IMAGE_PIXELS = 933120000   # Quiets Pillow from giving warnings on really large images (WARNING: Exposes a risk of DoS from malicious images)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

preset_prompts = [
    "None",
    "Write a descriptive caption for this image in a formal tone.-以正式的语气为这张图片写一个描述性的标题。",
    "Write a descriptive caption for this image in a casual tone.-以随意的语气为这张图片写一个描述性的标题。",
    "Write a stable diffusion prompt for this image.-为这张图片写一个 stable diffusion 提示。",
    "Write a MidJourney prompt for this image.-为这张图片写一个 MidJourney 提示。",
    "Write a list of Booru tags for this image.-为这张图片写一个 Booru 标签列表。",
    "Write a list of Booru-like tags for this image.-为这张图片写一个类似 Booru 的标签列表。",
    ("Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc."
     "-像艺术评论家一样分析这张图片，提供关于其构图、风格、象征意义、色彩和光线的使用、可能属于的任何艺术运动等信息。"),
    "Write a caption for this image as though it were a product listing.-为这张图片写一个标题，就像它是一个产品列表一样。",
    "Write a caption for this image as if it were being used for a social media post.-为这张图片写一个标题，就像它被用于社交媒体帖子一样。",
    "请用中文详细描述这张图片的内容。-Please describe this image in detail in Chinese.",
    "请分析这张图片的艺术风格和构图特点。-Please analyze the artistic style and composition of this image in Chinese.",
    "请描述这张图片中的人物特征和表情。-Please describe the character features and expressions in this image in Chinese.",
    "请用中文生成适合AI绘画的提示词。-Please generate AI art prompts in Chinese for this image.",
]

class JoyCaptionModel:
    def __init__(self, model: str, nf4: bool):
        IS_NF4 = nf4
        _log_info(f"初始化JoyCaption模型: {model}, nf4={nf4}")
        
        # Load JoyCaption
        if IS_NF4:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            _log_info("使用NF4量化配置")
        else:
            nf4_config = None
            _log_info("使用标准配置")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
            assert isinstance(self.tokenizer, PreTrainedTokenizer) or isinstance(self.tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(self.tokenizer)}"
            _log_info("Tokenizer加载成功")

            if IS_NF4:
                # 使用更安全的设备映射
                self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                    model, 
                    torch_dtype=torch.bfloat16, 
                    quantization_config=nf4_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).eval()
                
                # 修复NF4模型的attention层
                try:
                    attention = self.llava_model.vision_tower.vision_model.head.attention
                    attention.out_proj = torch.nn.Linear(
                        attention.embed_dim, 
                        attention.embed_dim, 
                        device=self.llava_model.device, 
                        dtype=torch.bfloat16
                    )
                    _log_info("NF4 attention层修复完成")
                except Exception as e:
                    _log_warning(f"NF4 attention层修复失败: {e}")
            else: 
                # 非NF4版本使用特殊的加载策略
                try:
                    # 首先检查safetensors版本
                    import safetensors
                    _log_info(f"safetensors版本: {safetensors.__version__}")
                    
                    # 尝试使用更保守的加载方式
                    self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                        model, 
                        torch_dtype=torch.bfloat16, 
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_safetensors=False  # 尝试不使用safetensors
                    ).eval()
                    _log_info("使用标准配置加载成功")
                except Exception as e1:
                    _log_warning(f"标准配置加载失败: {e1}")
                    try:
                        # 如果失败，尝试使用更基础的加载方式
                        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                            model, 
                            torch_dtype=torch.float16,  # 使用float16而不是bfloat16
                            device_map="auto",
                            trust_remote_code=True
                        ).eval()
                        _log_info("使用float16配置加载成功")
                    except Exception as e2:
                        _log_warning(f"float16配置加载失败: {e2}")
                        try:
                            # 尝试使用CPU加载然后转移到GPU
                            _log_info("尝试CPU加载方式...")
                            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                                model, 
                                device_map="cpu",
                                trust_remote_code=True
                            )
                            # 手动转移到GPU
                            self.llava_model = self.llava_model.to("cuda")
                            self.llava_model = self.llava_model.eval()
                            _log_info("使用CPU加载方式成功")
                        except Exception as e3:
                            _log_warning(f"CPU加载方式失败: {e3}")
                            try:
                                # 最后尝试最基础的加载方式，不使用device_map
                                _log_info("尝试基础加载方式...")
                                self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                                    model, 
                                    trust_remote_code=True
                                )
                                # 手动转移到GPU
                                self.llava_model = self.llava_model.to("cuda")
                                self.llava_model = self.llava_model.eval()
                                _log_info("使用基础配置加载成功")
                            except Exception as e4:
                                _log_warning(f"基础加载方式失败: {e4}")
                                try:
                                    # 尝试使用更保守的safetensors处理
                                    _log_info("尝试保守safetensors加载方式...")
                                    import os
                                    os.environ['SAFETENSORS_FAST_GPU'] = '0'  # 禁用快速GPU加载
                                    self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                                        model, 
                                        torch_dtype=torch.float32,  # 使用float32
                                        device_map="cpu",
                                        trust_remote_code=True,
                                        use_safetensors=True
                                    )
                                    # 手动转移到GPU
                                    self.llava_model = self.llava_model.to("cuda")
                                    self.llava_model = self.llava_model.eval()
                                    _log_info("使用保守safetensors加载方式成功")
                                except Exception as e5:
                                    _log_error(f"所有加载方式都失败: {e5}")
                                    # 提供详细的错误信息和解决建议
                                    error_details = f"""
模型加载失败，可能的原因：
1. safetensors库版本不兼容
2. 模型文件损坏
3. 内存不足
4. CUDA版本不兼容

建议解决方案：
1. 更新safetensors: pip install --upgrade safetensors
2. 重新下载模型文件
3. 使用NF4版本模型: llama-joycaption-beta-one-hf-llava-nf4
4. 检查CUDA和PyTorch版本兼容性

详细错误: {e5}
                                    """
                                    _log_error(error_details)
                                
                                # 智能回退：尝试加载NF4版本
                                _log_info("尝试自动回退到NF4版本...")
                                try:
                                    nf4_model_path = model.replace("-llava", "-llava-nf4")
                                    if os.path.exists(nf4_model_path):
                                        _log_info(f"找到NF4版本: {nf4_model_path}")
                                        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                                            nf4_model_path, 
                                            torch_dtype=torch.bfloat16, 
                                            quantization_config=BitsAndBytesConfig(
                                                load_in_4bit=True, 
                                                bnb_4bit_quant_type="nf4", 
                                                bnb_4bit_quant_storage=torch.bfloat16,
                                                bnb_4bit_use_double_quant=True, 
                                                bnb_4bit_compute_dtype=torch.bfloat16,
                                            ),
                                            device_map="auto",
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=True
                                        ).eval()
                                        
                                        # 修复NF4模型的attention层
                                        try:
                                            attention = self.llava_model.vision_tower.vision_model.head.attention
                                            attention.out_proj = torch.nn.Linear(
                                                attention.embed_dim, 
                                                attention.embed_dim, 
                                                device=self.llava_model.device, 
                                                dtype=torch.bfloat16
                                            )
                                            _log_info("NF4 attention层修复完成")
                                        except Exception as e:
                                            _log_warning(f"NF4 attention层修复失败: {e}")
                                        
                                        _log_info("自动回退到NF4版本成功")
                                    else:
                                        raise Exception("NF4版本不存在，无法回退")
                                except Exception as e6:
                                    _log_error(f"自动回退失败: {e6}")
                                    raise Exception(f"模型加载失败，且无法回退到NF4版本: {e5}")
            
            assert isinstance(self.llava_model, LlavaForConditionalGeneration)
            _log_info("模型加载成功")
            
        except Exception as e:
            _log_error(f"模型加载失败: {e}")
            raise e
    
    @torch.inference_mode()
    def inference(
        self,
        prompt,
        # num_workers,
        batch_size,
        max_new_tokens,
        do_sample,
        use_cache,
        temperature,
        top_k,
        top_p,
        save_img_prompt_to_folder=None,
        images_dir=None,
        tagger="",
        image=None,
        ):

        _log_info(f"开始推理，提示词: {prompt}")
        _log_info(f"参数: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}")

        image_paths = []
        if images_dir and images_dir.strip():
            # Find the images
            image_paths = find_images(images_dir)
            _log_info(f"从目录加载图片: {len(image_paths)} 张")
        elif image is not None:
            from io import BytesIO
            image = tensor2pil(image)
            _log_info(f"处理输入图片，尺寸: {image.size}")
            if save_img_prompt_to_folder:
                image_path = os.path.join(save_img_prompt_to_folder, f"{str(uuid.uuid4())}.png")
                image.save(image_path)
                image_paths = [image_path]
                _log_info(f"保存图片到: {image_path}")
            else:
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                buffer.seek(0) 
                image_paths = [buffer]
                _log_info("图片保存在内存中")
        else:
            _log_error("没有提供图片或图片目录")
            return "错误：没有提供图片或图片目录"
        
        if not image_paths:
            _log_error("没有找到可处理的图片")
            return "错误：没有找到可处理的图片"
        
        tagger = tagger.strip() + ", " if tagger.strip() != "" else ""

        _log_info("创建数据集...")
        dataset = ImageDataset(prompt, image_paths, self.tokenizer, self.llava_model.config.image_token_index, self.llava_model.config.image_seq_length)
        dataloader = DataLoader(
                        dataset, 
                        collate_fn=dataset.collate_fn, 
                        # num_workers=num_workers, 
                        shuffle=False, 
                        drop_last=False, 
                        batch_size=batch_size
                        )
        
        end_of_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        assert isinstance(end_of_header_id, int) and isinstance(end_of_turn_id, int)

        pbar = tqdm(total=len(image_paths), desc="Captioning images...", dynamic_ncols=True)
        
        n = 1
        for batch in dataloader:
            vision_dtype = self.llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
            vision_device = self.llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
            language_device = self.llava_model.language_model.get_input_embeddings().weight.device
            _log_info(f"设备信息: vision={vision_device}, language={language_device}")

            # Move to GPU
            pixel_values = batch['pixel_values'].to(vision_device, non_blocking=True)
            input_ids = batch['input_ids'].to(language_device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(language_device, non_blocking=True)

            # Normalize the image
            pixel_values = pixel_values / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to(vision_dtype)

            _log_info("开始生成描述...")
            # Generate the captions
            generate_ids = self.llava_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                suppress_tokens=None,
                use_cache=use_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Trim off the prompts
            assert isinstance(generate_ids, torch.Tensor)
            generate_ids = generate_ids.tolist()
            generate_ids = [trim_off_prompt(ids, end_of_header_id, end_of_turn_id) for ids in generate_ids]

            # Decode the captions
            captions = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            captions = [c.strip() for c in captions]
            _log_info(f"生成描述: {captions[0]}")
            
            if image is not None:
                if save_img_prompt_to_folder:
                    write_caption(Path(image_paths[0]), tagger + captions[0])
                return tagger + captions[0]
            
            import shutil
            for path, caption in zip(batch['paths'], captions):
                if save_img_prompt_to_folder:
                    file_ext = os.path.splitext(path)[1]
                    img_path = os.path.join(save_img_prompt_to_folder, f"{n:0{7}d}{file_ext}")
                    shutil.copy2(path, img_path)
                    write_caption(Path(img_path), tagger + caption)
                    n += 1
                else:
                    write_caption(Path(path), tagger + caption)

            pbar.update(len(captions))

    
    def clean(self):
        import gc
        _log_info("开始清理模型...")
        
        # 清理模型
        if hasattr(self, 'llava_model') and self.llava_model is not None:
            try:
                del self.llava_model
                _log_info("模型已删除")
            except Exception as e:
                _log_warning(f"删除模型失败: {e}")
        
        # 清理tokenizer
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            try:
                del self.tokenizer
                _log_info("Tokenizer已删除")
            except Exception as e:
                _log_warning(f"删除Tokenizer失败: {e}")
        
        # 清理CUDA缓存
        try:
            torch.cuda.empty_cache()
            _log_info("CUDA缓存已清理")
        except Exception as e:
            _log_warning(f"清理CUDA缓存失败: {e}")
        
        # 强制垃圾回收
        try:
            gc.collect()
            _log_info("垃圾回收完成")
        except Exception as e:
            _log_warning(f"垃圾回收失败: {e}")
        
        # 重置属性
        self.llava_model = None
        self.tokenizer = None
        _log_info("模型清理完成")


def trim_off_prompt(input_ids: list[int], eoh_id: int, eot_id: int) -> list[int]:
    # Trim off the prompt
    while True:
        try:
            i = input_ids.index(eoh_id)
        except ValueError:
            break
        
        input_ids = input_ids[i + 1:]
    
    # Trim off the end
    try:
        i = input_ids.index(eot_id)
    except ValueError:
        return input_ids
    
    return input_ids[:i]


def write_caption(image_path: Path, caption: str):
    caption_path = image_path.with_suffix(".txt")
    try:
        f = os.open(caption_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)  # Write-only, create if not exist, fail if exists
    except FileExistsError:
        logging.warning(f"Caption file '{caption_path}' already exists")
        return
    except Exception as e:
        logging.error(f"Failed to open caption file '{caption_path}': {e}")
        return
    
    try:
        os.write(f, caption.encode("utf-8"))
        os.close(f)
    except Exception as e:
        logging.error(f"Failed to write caption to '{caption_path}': {e}")
        return


class ImageDataset(Dataset):
    def __init__(self, prompt: str, paths: list[Path], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], image_token_id: int, image_seq_length: int):
        self.prompt = prompt
        self.paths = paths
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.pad_token_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]

        # Preprocess image
        # NOTE: I don't use the Processor here and instead do it manually.
        # This is because in my testing a simple resize in Pillow yields higher quality results than the Processor,
        # and the Processor had some buggy behavior on some images.
        # And yes, with the so400m model, the model expects the image to be squished into a square, not padded.
        try:
            image = Image.open(path)
            if image.size != (384, 384):
                image = image.resize((384, 384), Image.LANCZOS)
            image = image.convert("RGB")
            pixel_values = TVF.pil_to_tensor(image)
        except Exception as e:
            logging.error(f"Failed to load image '{path}': {e}")
            pixel_values = None   # Will be filtered out later

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": self.prompt,
            },
        ]

        # Format the conversation
        convo_string = self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        convo_tokens = self.tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

        # Repeat the image tokens
        input_tokens = []
        for token in convo_tokens:
            if token == self.image_token_id:
                input_tokens.extend([self.image_token_id] * self.image_seq_length)
            else:
                input_tokens.append(token)
        
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            'path': path,
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        # Filter out images that failed to load
        batch = [item for item in batch if item['pixel_values'] is not None]

        # Pad input_ids and attention_mask
        # Have to use left padding because HF's generate can't handle right padding it seems
        max_length = max(item['input_ids'].shape[0] for item in batch)
        n_pad = [max_length - item['input_ids'].shape[0] for item in batch]
        input_ids = torch.stack([torch.nn.functional.pad(item['input_ids'], (n, 0), value=self.pad_token_id) for item, n in zip(batch, n_pad)])
        attention_mask = torch.stack([torch.nn.functional.pad(item['attention_mask'], (n, 0), value=0) for item, n in zip(batch, n_pad)])

        # Stack pixel values
        pixel_values = torch.stack([item['pixel_values'] for item in batch])

        # Paths
        paths = [item['path'] for item in batch]

        return {
            'paths': paths,
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def find_images(folder_path):
    """查找指定文件夹下的常见图片文件"""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [str(p.resolve()) for p in Path(folder_path).rglob("*") if p.suffix.lower() in extensions]

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# 检查safetensors版本
try:
    import safetensors
    safetensors_version = safetensors.__version__
    _log_info(f"检测到safetensors版本: {safetensors_version}")
    
    # 检查版本兼容性
    version_parts = safetensors_version.split('.')
    if len(version_parts) >= 2:
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major < 0 or (major == 0 and minor < 4):
            _log_warning(f"safetensors版本 {safetensors_version} 可能过低，建议升级到0.4.0或更高版本")
except ImportError:
    _log_warning("未安装safetensors库，可能影响模型加载")
except Exception as e:
    _log_warning(f"检查safetensors版本时出错: {e}")

# 全局模型缓存
MODEL_CACHE = None

class JoyCaptionNode:
    def __init__(self):
        self.model_name = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "输入自定义提示词（支持中英文，可选）"}),
                "preset_prompt": (preset_prompts, {"default": "请用中文详细描述这张图片的内容。-Please describe this image in detail in Chinese."}),
                "merge_prompt": ("BOOLEAN", {"default": False}),
                "model": (
                    [
                        "llama-joycaption-beta-one-hf-llava-nf4",
                        "llama-joycaption-beta-one-hf-llava",
                    ],
                    {"default": "llama-joycaption-beta-one-hf-llava-nf4", "description": "NF4版本：内存占用少，加载快；标准版本：精度更高，但可能加载较慢"},
                ),
                "use_cache": ("BOOLEAN", {"default": True}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 10, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "unload_model": ("BOOLEAN", {"default": True}),
                "save_img_prompt_to_folder": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
            },
            "optional": {
                "images_dir": ("STRING", {"default": "", "multiline": False}),
                "tagger": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_caption"
    CATEGORY = "Ken-Chen/LLM-Nano-Banana"

    def generate_caption(
        self,
        model,
        user_prompt,
        preset_prompt,
        merge_prompt,
        max_new_tokens,
        do_sample,
        use_cache,
        temperature,
        top_k,
        top_p,
        save_img_prompt_to_folder="",
        batch_size=1,
        images_dir=None,
        tagger="",
        image=None,
        seed=0,
        unload_model=False,
    ):
        try:
            if seed != 0:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            nf4 = False
            if model == "llama-joycaption-beta-one-hf-llava-nf4":
                nf4 = True
            
            # 构建最终提示词
            final_prompt = ""
            if preset_prompt != "None":
                # 提取预设提示词的英文部分
                preset_english = preset_prompt.rsplit("-", 1)[0]
                
                if user_prompt and user_prompt.strip():
                    # 如果用户提供了自定义提示词
                    if merge_prompt:
                        # 合并模式：用户提示词 + 预设提示词
                        # 支持中英文混合
                        final_prompt = f"{user_prompt.strip()}. {preset_english}"
                        _log_info(f"合并模式 - 用户提示词: {user_prompt.strip()}")
                        _log_info(f"合并模式 - 预设提示词: {preset_english}")
                    else:
                        # 用户提示词优先模式：只使用用户提示词
                        final_prompt = user_prompt.strip()
                        _log_info(f"用户提示词优先模式: {final_prompt}")
                else:
                    # 没有用户提示词，使用预设提示词
                    final_prompt = preset_english
                    _log_info(f"使用预设提示词: {final_prompt}")
            else:
                # 预设提示词为"None"，只使用用户提示词
                if user_prompt and user_prompt.strip():
                    final_prompt = user_prompt.strip()
                    _log_info(f"仅使用用户提示词: {final_prompt}")
                else:
                    # 没有预设提示词也没有用户提示词，使用默认提示词
                    final_prompt = "Describe this image in detail."
                    _log_info(f"使用默认提示词: {final_prompt}")
            
            _log_info(f"最终提示词: {final_prompt}")
            
            # 检测提示词语言并添加相应的语言指示
            if any('\u4e00' <= char <= '\u9fff' for char in final_prompt):
                # 包含中文字符，添加中文输出指示
                if "中文" not in final_prompt and "Chinese" not in final_prompt:
                    final_prompt = f"{final_prompt} Please respond in Chinese."
                _log_info("检测到中文提示词，将输出中文描述")
            else:
                # 英文提示词，保持原样
                _log_info("使用英文提示词")
            
            _log_info(f"处理后的最终提示词: {final_prompt}")
            
            # 构建模型路径
            model_path_full = os.path.join("D:\\Ken_ComfyUI_312\\ComfyUI\\models\\LLM", model)
            _log_info(f"使用模型: {model}")
            _log_info(f"模型路径: {model_path_full}")
            
            # 验证模型路径是否存在
            if not os.path.exists(model_path_full):
                error_msg = f"模型路径不存在: {model_path_full}"
                _log_error(error_msg)
                return (error_msg,)
            
            # 检查模型文件
            config_file = os.path.join(model_path_full, "config.json")
            if not os.path.exists(config_file):
                error_msg = f"模型配置文件不存在: {config_file}"
                _log_error(error_msg)
                return (error_msg,)
            
            _log_info("模型路径验证通过")
            
            global MODEL_CACHE
            if MODEL_CACHE is None or self.model_name != model_path_full:
                try:
                    _log_info("开始加载模型...")
                    
                    # 清理之前的模型缓存
                    if MODEL_CACHE is not None:
                        _log_info("清理之前的模型缓存")
                        try:
                            MODEL_CACHE.clean()
                        except Exception as e:
                            _log_warning(f"清理模型缓存失败: {e}")
                        MODEL_CACHE = None
                    
                    self.model_name = model_path_full
                    MODEL_CACHE = JoyCaptionModel(model_path_full, nf4)
                    _log_info("模型加载完成")
                except Exception as e:
                    error_msg = f"模型加载失败: {e}"
                    _log_error(error_msg)
                    return (error_msg,)
            
            JC = MODEL_CACHE

            # 处理保存路径
            save_path = None
            if save_img_prompt_to_folder and isinstance(save_img_prompt_to_folder, str) and save_img_prompt_to_folder.strip():
                save_path = save_img_prompt_to_folder.strip()
                _log_info(f"将保存到: {save_path}")

            # 调试参数
            _log_info(f"images_dir类型: {type(images_dir)}, 值: {repr(images_dir)}")
            _log_info(f"image类型: {type(image)}, 值: {repr(image)}")

            # 验证images_dir参数
            valid_images_dir = None
            if images_dir and isinstance(images_dir, str) and images_dir.strip():
                # 检查是否是有效的目录路径
                if not images_dir.startswith('[') and not images_dir.endswith(']'):
                    valid_images_dir = images_dir.strip()
                    _log_info(f"使用images_dir: {valid_images_dir}")
                else:
                    _log_warning(f"images_dir值无效: {images_dir}")

            if valid_images_dir:
                _log_info("使用images_dir模式")
                try:
                    result = JC.inference(
                        final_prompt, 
                        batch_size, 
                        max_new_tokens, 
                        do_sample, 
                        use_cache, 
                        temperature, 
                        top_k, 
                        top_p, 
                        save_img_prompt_to_folder=save_path, 
                        images_dir=valid_images_dir, 
                        tagger=tagger,
                        image=None,
                    )
                    return (result,)
                except Exception as e:
                    error_msg = f"推理失败: {e}"
                    _log_error(error_msg)
                    return (error_msg,)
            else:
                _log_info("使用image模式")
                try:
                    result = JC.inference(
                        final_prompt, 
                        batch_size, 
                        max_new_tokens, 
                        do_sample, 
                        use_cache, 
                        temperature, 
                        top_k, 
                        top_p, 
                        save_img_prompt_to_folder=save_path, 
                        images_dir=None, 
                        tagger=tagger,
                        image=image,
                    )
                    return (result,)
                except Exception as e:
                    error_msg = f"推理失败: {e}"
                    _log_error(error_msg)
                    return (error_msg,)
            
            if unload_model:
                import gc
                JC.clean()
                JC = None
                MODEL_CACHE = None
                gc.collect()
                torch.cuda.empty_cache()

            return (caption,)
            
        except Exception as e:
            error_msg = f"JoyCaption生成失败: {e}"
            _log_error(error_msg)
            return (error_msg,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "JoyCaption-BetaOne-Run": JoyCaptionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyCaption-BetaOne-Run": "JoyCaption-BetaOne-Run"
} 