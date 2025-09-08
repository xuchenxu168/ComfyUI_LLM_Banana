# -*- coding: utf-8 -*-
"""
通用AI放大模块：优先 Nomos8k(SwinIR) / SwinIR / RealESRGAN(2x/4x) / BSRGAN，自动回退。
模型目录：ComfyUI/models/upscale_models/
"""
import os
import subprocess
import json
import time
import shutil
from typing import Optional

from PIL import Image


def _load_gigapixel_config():
	"""加载 Gigapixel AI 配置"""
	try:
		config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Gigapixel_config.json')
		if os.path.exists(config_path):
			with open(config_path, 'r', encoding='utf-8') as f:
				return json.load(f)
	except Exception as e:
		print(f"[Config] 加载 Gigapixel 配置失败: {e}")

	# 返回默认配置
	return {
		"gigapixel_ai": {
			"enabled": True,
			"executable_paths": [
				"C:\\Program Files\\Topaz Labs LLC\\Topaz Gigapixel AI\\gigapixel.exe",
				"gigapixel"
			],
			"default_model": "std",
			"timeout_seconds": 300
		}
	}


def _models_dir() -> str:
	# 定位到 ComfyUI 根目录，再进入 models/upscale_models
	root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	dir_path = os.path.join(root, 'models', 'upscale_models')
	print(f"[Upscale] 模型目录: {dir_path}")
	return dir_path


def _exists(*names: str) -> Optional[str]:
	base = _models_dir()
	for n in names:
		p = os.path.join(base, n)
		if os.path.exists(p):
			print(f"[Upscale] 找到模型: {p}")
			return p
		else:
			print(f"[Upscale] 未找到: {p}")
	return None


def _device_is_cuda() -> bool:
	try:
		import torch
		return torch.cuda.is_available()
	except Exception:
		return False


def _pil_to_bgr_ndarray(img: Image.Image):
	import numpy as np, cv2
	if img.mode != 'RGB':
		img = img.convert('RGB')
	rgb = np.array(img)
	bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
	return bgr


def _upscale_with_realesrgan(img: Image.Image, scale_hint: int) -> Optional[Image.Image]:
	try:
		from realesrgan import RealESRGANer
		from basicsr.archs.rrdbnet_arch import RRDBNet
		model_path = None
		model = None
		# 仅使用最稳的 x4
		model_path = _exists('RealESRGAN_x4plus.pth')
		if not model_path:
			return None
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
		scale = 4
		upsampler = RealESRGANer(
			scale=scale,
			model_path=model_path,
			model=model,
			tile=0,
			tile_pad=10,
			pre_pad=0,
			half=_device_is_cuda(),
		)
		bgr = _pil_to_bgr_ndarray(img)
		if hasattr(upsampler, 'enhance'):
			res_bgr, _ = upsampler.enhance(bgr)
			import cv2
			res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
			return Image.fromarray(res_rgb)
	except Exception as e:
		print(f"⚠️ RealESRGAN 失败: {e}")
	return None


def _upscale_with_swinir(img: Image.Image, scale_hint: int) -> Optional[Image.Image]:
	# 仅在权重存在时使用；SwinIR推理较慢，但稳定
	try:
		pth = _exists('003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-M_x4_GAN.pth')
		if not pth:
			return None
		from realesrgan import RealESRGANer
		from basicsr.archs.swinir_arch import SwinIR
		model = SwinIR(
			upscale=4,
			in_chans=3,
			img_size=64,
			window_size=8,
			img_range=1.0,
			depths=[6,6,6,6],
			num_heads=[6,6,6,6],
			mlp_ratio=2,
			upsampler='pixelshuffle',
			resi_connection='1conv'
		)
		upsampler = RealESRGANer(scale=4, model_path=pth, model=model, tile=0, tile_pad=10, pre_pad=0, half=_device_is_cuda())
		bgr = _pil_to_bgr_ndarray(img)
		if hasattr(upsampler, 'enhance'):
			res_bgr, _ = upsampler.enhance(bgr)
			import cv2
			res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
			return Image.fromarray(res_rgb)
	except Exception as e:
		print(f"⚠️ SwinIR 失败: {e}")
	return None


def _upscale_with_bsrgan(img: Image.Image, scale_hint: int) -> Optional[Image.Image]:
	try:
		pth = _exists('BSRGANx4.pth')
		if not pth:
			return None
		from realesrgan import RealESRGANer
		from basicsr.archs.rrdbnet_arch import RRDBNet
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
		upsampler = RealESRGANer(scale=4, model_path=pth, model=model, tile=0, tile_pad=10, pre_pad=0, half=_device_is_cuda())
		bgr = _pil_to_bgr_ndarray(img)
		if hasattr(upsampler, 'enhance'):
			res_bgr, _ = upsampler.enhance(bgr)
			import cv2
			res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
			return Image.fromarray(res_rgb)
	except Exception as e:
		print(f"⚠️ BSRGAN 失败: {e}")
	return None


def _upscale_with_nomos(img: Image.Image, scale_hint: int) -> Optional[Image.Image]:
	"""Nomos8k 权重优先用 spandrel 方式加载；失败时按 SwinIR x4 回退"""
	# 先尝试 spandrel（与 ComfyUI 原生 nodes_upscale_model.py 一致）
	try:
		pth = _exists('完美高清放大4xNomos8kSCHAT-L.pth', '4xNomos8kSCHAT-L.pth', 'Nomos8kSCHAT-L.pth')
		if not pth:
			return None
		import torch
		import comfy.utils
		from comfy import model_management
		from spandrel import ModelLoader, ImageModelDescriptor
		try:
			from spandrel_extra_arches import EXTRA_REGISTRY
			from spandrel import MAIN_REGISTRY
			MAIN_REGISTRY.add(*EXTRA_REGISTRY)
			print('[Upscale/Nomos] spandrel_extra_arches 已启用')
		except Exception:
			pass
		# 加载权重
		sd = comfy.utils.load_torch_file(pth, safe_load=True)
		if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
			sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
		upscale_model = ModelLoader().load_from_state_dict(sd).eval()
		if not isinstance(upscale_model, ImageModelDescriptor):
			print('⚠️ 不是单图像模型，跳过 spandrel 路径')
			raise RuntimeError('not ImageModelDescriptor')
		# 设备与显存管理
		device = model_management.get_torch_device()
		memory_required = model_management.module_size(upscale_model.model)
		# 按 ComfyUI 估算方式
		memory_required += (512 * 512 * 3) * 4 * max(getattr(upscale_model, 'scale', 4.0), 1.0) * 384.0
		# 输入图像内存
		# 创建输入 tensor
		import numpy as np
		if img.mode != 'RGB':
			img_rgb = img.convert('RGB')
		else:
			img_rgb = img
		arr = np.array(img_rgb).astype('float32') / 255.0
		in_img = torch.from_numpy(arr).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
		memory_required += in_img.nelement() * in_img.element_size()
		model_management.free_memory(memory_required, device)
		upscale_model.to(device)
		in_img = in_img.to(device)
		# 平铺推理，自动降 tile 防 OOM
		tile = 512
		overlap = 32
		while True:
			try:
				s = comfy.utils.tiled_scale(
					in_img,
					lambda a: upscale_model(a),
					tile_x=tile,
					tile_y=tile,
					overlap=overlap,
					upscale_amount=getattr(upscale_model, 'scale', 4.0),
					pbar=None
				)
				break
			except model_management.OOM_EXCEPTION as e:
				tile //= 2
				if tile < 128:
					raise e
		# 回到 CPU 并转回 PIL
		upscale_model.to('cpu')
		s = torch.clamp(s, min=0.0, max=1.0)
		out = s.permute(0, 2, 3, 1)[0].detach().float().cpu().numpy()
		out = (out * 255.0 + 0.5).clip(0, 255).astype('uint8')
		from PIL import Image as _Image
		return _Image.fromarray(out)
	except Exception as e:
		print(f"⚠️ Nomos8k(spandrel) 失败，将回退至 SwinIR 方式: {e}")
	# 回退到 SwinIR x4 方式
	try:
		from realesrgan import RealESRGANer
		from basicsr.archs.swinir_arch import SwinIR
		pth = _exists('完美高清放大4xNomos8kSCHAT-L.pth', '4xNomos8kSCHAT-L.pth', 'Nomos8kSCHAT-L.pth')
		if not pth:
			return None
		model = SwinIR(
			upscale=4,
			in_chans=3,
			img_size=64,
			window_size=8,
			img_range=1.0,
			depths=[6,6,6,6],
			num_heads=[6,6,6,6],
			mlp_ratio=2,
			upsampler='pixelshuffle',
			resi_connection='1conv'
		)
		upsampler = RealESRGANer(scale=4, model_path=pth, model=model, tile=0, tile_pad=10, pre_pad=0, half=_device_is_cuda())
		bgr = _pil_to_bgr_ndarray(img)
		if hasattr(upsampler, 'enhance'):
			res_bgr, _ = upsampler.enhance(bgr)
			import cv2
			res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
			return Image.fromarray(res_rgb)
	except Exception as e:
		print(f"⚠️ Nomos8k 放大失败: {e}")
	return None


def _upscale_with_realesrgan_x2(img: Image.Image) -> Optional[Image.Image]:
	"""RealESRGAN x2（若权重存在）。优先使用 spandrel 按 ComfyUI 原生加载，避免架构不匹配。"""
	# 先尝试 spandrel 加载（支持更多架构，避免 12 通道等不匹配问题）
	try:
		pth = _exists('RealESRGAN_x2plus.pth')
		if not pth:
			return None
		import torch
		import comfy.utils
		from comfy import model_management
		from spandrel import ModelLoader, ImageModelDescriptor
		try:
			from spandrel_extra_arches import EXTRA_REGISTRY
			from spandrel import MAIN_REGISTRY
			MAIN_REGISTRY.add(*EXTRA_REGISTRY)
			print('[Upscale/x2] spandrel_extra_arches 已启用')
		except Exception:
			pass
		sd = comfy.utils.load_torch_file(pth, safe_load=True)
		if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
			sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
		upscale_model = ModelLoader().load_from_state_dict(sd).eval()
		if not isinstance(upscale_model, ImageModelDescriptor):
			raise RuntimeError('x2 不是单图像模型')
		device = model_management.get_torch_device()
		memory_required = model_management.module_size(upscale_model.model)
		memory_required += (512 * 512 * 3) * 4 * max(getattr(upscale_model, 'scale', 2.0), 1.0) * 384.0
		# 准备输入
		import numpy as np
		if img.mode != 'RGB':
			img_rgb = img.convert('RGB')
		else:
			img_rgb = img
		arr = np.array(img_rgb).astype('float32') / 255.0
		in_img = torch.from_numpy(arr).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
		memory_required += in_img.nelement() * in_img.element_size()
		model_management.free_memory(memory_required, device)
		upscale_model.to(device)
		in_img = in_img.to(device)
		tile = 512
		overlap = 32
		while True:
			try:
				s = comfy.utils.tiled_scale(
					in_img,
					lambda a: upscale_model(a),
					tile_x=tile,
					tile_y=tile,
					overlap=overlap,
					upscale_amount=getattr(upscale_model, 'scale', 2.0),
					pbar=None
				)
				break
			except model_management.OOM_EXCEPTION as e:
				tile //= 2
				if tile < 128:
					raise e
		upscale_model.to('cpu')
		s = torch.clamp(s, min=0.0, max=1.0)
		out = s.permute(0, 2, 3, 1)[0].detach().float().cpu().numpy()
		out = (out * 255.0 + 0.5).clip(0, 255).astype('uint8')
		from PIL import Image as _Image
		return _Image.fromarray(out)
	except Exception as e:
		print(f"⚠️ RealESRGAN x2(spandrel) 失败，将尝试 RRDBNet: {e}")
	# 回退：使用 RealESRGANer + RRDBNet（可能仍因权重架构不符而失败）
	try:
		pth = _exists('RealESRGAN_x2plus.pth')
		if not pth:
			return None
		from realesrgan import RealESRGANer
		from basicsr.archs.rrdbnet_arch import RRDBNet
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
		upsampler = RealESRGANer(scale=2, model_path=pth, model=model, tile=0, tile_pad=10, pre_pad=0, half=_device_is_cuda())
		bgr = _pil_to_bgr_ndarray(img)
		if hasattr(upsampler, 'enhance'):
			res_bgr, _ = upsampler.enhance(bgr)
			import cv2
			res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
			return Image.fromarray(res_rgb)
	except Exception as e:
		print(f"⚠️ RealESRGAN x2 失败: {e}")
	return None


def smart_upscale(img: Image.Image, target_w: int, target_h: int, gigapixel_model: str = "High Fidelity") -> Optional[Image.Image]:
	"""智能选择模型放大到精确目标尺寸（返回精确尺寸的图像；失败返回 None）
	策略：使用AI模型放大后，精确调整到目标尺寸，避免过度放大
	"""
	# 所需放大倍数
	s_needed = max(target_w / img.width, target_h / img.height)
	if s_needed <= 1.0:
		# 如果不需要放大，直接调整到目标尺寸
		print(f"[Upscale] 无需放大，直接调整到目标尺寸: {img.size} -> ({target_w}, {target_h})")
		return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

	print(f"[Upscale] 需要放大倍数: {s_needed:.3f}x ({img.size} -> ({target_w}, {target_h}))")

	# 🚀 优先尝试 Gigapixel AI（支持任意精确倍数）
	if _detect_gigapixel_ai():
		print(f"[Upscale] 检测到 Gigapixel AI，优先使用精确放大，模型: {gigapixel_model}")
		gigapixel_result = _upscale_with_gigapixel_ai(img, target_w, target_h, gigapixel_model)
		if gigapixel_result is not None:
			return gigapixel_result
		print(f"[Upscale] Gigapixel AI 失败，回退到其他AI模型")

	def apply_ai_upscale_and_resize(ai_result: Optional[Image.Image]) -> Optional[Image.Image]:
		"""应用AI放大后精确调整到目标尺寸"""
		if ai_result is None:
			return None

		# 如果AI放大结果已经是目标尺寸，直接返回
		if ai_result.size == (target_w, target_h):
			print(f"[Upscale] AI放大结果已是目标尺寸: {ai_result.size}")
			return ai_result

		# 否则精确调整到目标尺寸
		print(f"[Upscale] AI放大完成，精确调整: {ai_result.size} -> ({target_w}, {target_h})")
		return ai_result.resize((target_w, target_h), Image.Resampling.LANCZOS)

	# 1) 若需要 <= 2.2 倍，优先使用 x2 模型
	if s_needed <= 2.2:
		print(f"[Upscale] 尝试使用 2x 模型（需求: {s_needed:.3f}x）")
		res = _upscale_with_realesrgan_x2(img)
		if res is not None:
			return apply_ai_upscale_and_resize(res)

		# 如果 2x 不可用，检查是否适合使用 4x
		if s_needed <= 3.0:  # 放宽条件，允许适度的4x使用
			print(f"[Upscale] 2x不可用，尝试使用 4x 模型")
			for fn in (_upscale_with_nomos, _upscale_with_swinir, _upscale_with_realesrgan, _upscale_with_bsrgan):
				res = fn(img, 4)
				if res is not None:
					return apply_ai_upscale_and_resize(res)
		else:
			print(f"[Upscale] 跳过 4x（过度放大，所需 {s_needed:.3f}x）")

		# AI模型都不可用，使用高质量传统放大
		print(f"[Upscale] AI模型不可用，使用传统高质量放大")
		return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

	# 2) 若需要 > 2.2 倍，使用 4x 模型
	print(f"[Upscale] 尝试使用 4x 模型（需求: {s_needed:.3f}x）")
	for fn in (_upscale_with_nomos, _upscale_with_swinir, _upscale_with_realesrgan, _upscale_with_bsrgan):
		res = fn(img, 4)
		if res is not None:
			return apply_ai_upscale_and_resize(res)

	# 所有AI模型都失败，尝试Gigapixel AI
	print(f"[Upscale] 尝试使用 Gigapixel AI 进行精确放大")
	gigapixel_result = _upscale_with_gigapixel_ai(img, target_w, target_h, gigapixel_model)
	if gigapixel_result is not None:
		return gigapixel_result

	# 最后回退到传统放大
	print(f"[Upscale] 所有AI模型失败，使用传统高质量放大")
	return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def _upscale_with_gigapixel_ai(img: Image.Image, target_w: int, target_h: int, gigapixel_model: str = "High Fidelity") -> Optional[Image.Image]:
	"""
	使用 Gigapixel AI 进行精确倍数放大
	支持任意放大倍数，质量最高
	"""
	try:
		# 加载配置
		config = _load_gigapixel_config()
		gigapixel_config = config.get("gigapixel_ai", {})

		if not gigapixel_config.get("enabled", True):
			print(f"[Gigapixel] 已在配置中禁用")
			return None

		# 从配置获取可执行文件路径
		gigapixel_paths = gigapixel_config.get("executable_paths", [
			"C:\\Program Files\\Topaz Labs LLC\\Topaz Gigapixel AI\\gigapixel.exe",
			"gigapixel"
		])

		gigapixel_exe = None
		use_system_command = False

		# 检查系统命令
		try:
			result = subprocess.run(['gigapixel', '--help'], capture_output=True, timeout=5)
			if result.returncode == 0:
				gigapixel_exe = 'gigapixel'
				use_system_command = True
				print(f"[Gigapixel] 找到系统命令: gigapixel")
		except:
			pass

		# 检查完整路径
		if not gigapixel_exe:
			for path in gigapixel_paths:
				if path != 'gigapixel' and os.path.exists(path):
					gigapixel_exe = path
					use_system_command = False
					print(f"[Gigapixel] 找到可执行文件: {path}")
					break

		if not gigapixel_exe:
			print(f"[Gigapixel] 未找到 Gigapixel AI，跳过")
			return None

		# 计算精确的放大倍数
		scale_w = target_w / img.width
		scale_h = target_h / img.height
		scale = max(scale_w, scale_h)  # 使用较大的倍数确保覆盖目标尺寸

		print(f"[Gigapixel] 精确放大倍数: {scale:.3f}x ({img.size} -> ({target_w}, {target_h}))")

		# 创建临时目录
		temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_gigapixel')
		os.makedirs(temp_dir, exist_ok=True)

		# 保存输入图像
		timestamp = int(time.time() * 1000)
		input_path = os.path.join(temp_dir, f'input_{timestamp}.png')
		output_dir = os.path.join(temp_dir, f'output_{timestamp}')
		os.makedirs(output_dir, exist_ok=True)

		img.save(input_path, 'PNG')

		# 构建 Gigapixel AI 命令
		if use_system_command:
			command = ['gigapixel']
		else:
			command = [gigapixel_exe]

		# 模型名称映射
		model_mapping = {
			'Art & CG': 'art',
			'Lines': 'lines',
			'Very Compressed': 'vc',
			'High Fidelity': 'fidelity',
			'Low Resolution': 'lowres',
			'Standard': 'std',
			'Text & Shapes': 'text',
			'Redefine': 'redefine',
			'Recover': 'recovery'
		}

		# 获取实际的模型代码
		model_code = model_mapping.get(gigapixel_model, 'fidelity')
		model_version = gigapixel_config.get("model_version", 2)

		print(f"[Gigapixel] 使用模型: {gigapixel_model} ({model_code})")

		command.extend([
			'--scale', str(scale),
			'-i', input_path,
			'-o', output_dir,
			'--model', model_code
		])

		# 添加模型版本（如果需要）
		mv2_models = {'std', 'fidelity', 'lowres', 'recovery'}
		if model_code in mv2_models:
			command.extend(['--mv', str(model_version)])

		print(f"[Gigapixel] 执行命令: {' '.join(command)}")

		# 从配置获取超时设置
		timeout_seconds = gigapixel_config.get("timeout_seconds", 300)

		# 执行 Gigapixel AI
		result = subprocess.run(
			command,
			capture_output=True,
			text=True,
			timeout=timeout_seconds,
			check=False
		)

		if result.returncode != 0:
			print(f"[Gigapixel] 执行失败: {result.stderr}")
			return None

		# 查找输出文件
		output_files = [
			os.path.join(output_dir, f)
			for f in os.listdir(output_dir)
			if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
		]

		if not output_files:
			print(f"[Gigapixel] 未找到输出文件")
			return None

		# 加载放大后的图像
		output_path = output_files[0]
		upscaled_img = Image.open(output_path)

		# 如果放大后的尺寸不是精确的目标尺寸，进行微调
		if upscaled_img.size != (target_w, target_h):
			print(f"[Gigapixel] 微调尺寸: {upscaled_img.size} -> ({target_w}, {target_h})")
			upscaled_img = upscaled_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

		# 清理临时文件（根据配置）
		if gigapixel_config.get("temp_cleanup", True):
			try:
				shutil.rmtree(temp_dir)
				print(f"[Gigapixel] 已清理临时文件")
			except Exception as e:
				print(f"[Gigapixel] 清理临时文件失败: {e}")
		else:
			print(f"[Gigapixel] 保留临时文件: {temp_dir}")

		print(f"[Gigapixel] 放大成功: {img.size} -> {upscaled_img.size}")
		return upscaled_img

	except subprocess.TimeoutExpired:
		print(f"[Gigapixel] 执行超时")
		return None
	except Exception as e:
		print(f"[Gigapixel] 执行失败: {e}")
		return None


def _detect_gigapixel_ai() -> bool:
	"""
	检测系统中是否安装了 Gigapixel AI
	"""
	try:
		# 检查系统命令
		result = subprocess.run(['gigapixel', '--help'], capture_output=True, timeout=5)
		if result.returncode == 0:
			return True
	except:
		pass

	# 检查常见安装路径
	common_paths = [
		"C:\\Program Files\\Topaz Labs LLC\\Topaz Gigapixel AI\\gigapixel.exe",
		"C:\\Program Files (x86)\\Topaz Labs LLC\\Topaz Gigapixel AI\\gigapixel.exe",
	]

	for path in common_paths:
		if os.path.exists(path):
			return True

	return False