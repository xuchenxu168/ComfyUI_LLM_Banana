# -*- coding: utf-8 -*-
"""
通用AI放大模块：优先 Nomos8k(SwinIR) / SwinIR / RealESRGAN(2x/4x) / BSRGAN，自动回退。
模型目录：ComfyUI/models/upscale_models/
"""
import os
from typing import Optional

from PIL import Image


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


def smart_upscale(img: Image.Image, target_w: int, target_h: int) -> Optional[Image.Image]:
	"""智能选择模型放大到覆盖 target 尺寸（返回放大图；失败返回 None）
	策略：优先使用不超过需求太多的最小倍数（避免 4x→再缩小导致过度平滑）
	"""
	# 所需放大倍数
	s_needed = max(target_w / img.width, target_h / img.height)
	if s_needed <= 1.0:
		s_needed = 1.0
	
	def try_and_crop(res: Optional[Image.Image]) -> Optional[Image.Image]:
		return res
	
	# 1) 若需要 <= 2.2 倍，并且有 x2，则优先 x2
	if s_needed <= 2.2:
		res = _upscale_with_realesrgan_x2(img)
		if res is not None:
			return res
		# 没有 x2，则考虑是否用 4x；如果 4/所需 > 1.6，跳过 4x，交由外部 LANCZOS 处理
		if 4.0 / s_needed > 1.6:
			print(f"[Upscale] 跳过 4x（过度放大，所需 {s_needed:.2f}）")
			return None
		# 否则尝试 4x 路径（Nomos/spandrel → SwinIR → RealESRGAN x4 → BSRGAN）
		for fn in (_upscale_with_nomos, _upscale_with_swinir, _upscale_with_realesrgan, _upscale_with_bsrgan):
			res = fn(img, 4)
			if res is not None:
				return res
		return None
	
	# 2) 若需要 > 2.2 倍，优先 4x（最接近所需）
	for fn in (_upscale_with_nomos, _upscale_with_swinir, _upscale_with_realesrgan, _upscale_with_bsrgan):
		res = fn(img, 4)
		if res is not None:
			return res
	return None 