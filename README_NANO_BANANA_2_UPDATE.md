# ComfyUI LLM Banana v2.2 专属更新说明 — Nano Banana 2 与重大功能合集

本文件在原始 README 的基础上，专门汇总本次 Nano Banana 2（NB2）与相关重大功能更新，便于快速了解改动、完成迁移并高效使用。

## 🆕 本次更新一览（TL;DR）
- Nano Banana 2 全量接入：使用 Gemini 原生格式，支持更高画质与更稳定的生成/编辑
- 镜像站全面适配：Comfly、T8、API4GPT、OpenRouter 按模型自动选择正确端点
- 迭代优化（会话式细化）全链路打通：生成、单图编辑、多图编辑均可连续细化并自动承接图像
- 多图编辑增强：更稳定的上传与超时策略、失败回退与多路径重试
- 分辨率与参数标准化：新增 imageConfig（aspectRatio + imageSize），支持 1K / 2K / 4K 与种子 seed
- 智能AI放大：内置对接 Topaz Gigapixel（smart_upscale），可选 2x~6x
- 安全设置与系统指令：沿用 v2.1 新特性，全面兼容 NB2

## 📦 适用范围
- 节点：Gemini Banana / Gemini Banana Mirror / Comfly Nano Banana / T8 Mirror Banana / OpenRouter Banana
- 版本：v2.2 及以上
- 配置：Gemini_Banana_config.json（镜像站、默认参数）

## 🚀 快速开始（Nano Banana 2）
NB2 使用 Gemini 原生 generateContent 接口，参数更规范、解析更稳定。

- 文本生图（API4GPT 示例）
  - 端点：/v1beta/models/gemini-3-pro-image-preview:generateContent
  - 关键参数：
    - generationConfig.responseModalities = ["Image"]
    - generationConfig.imageConfig = { aspectRatio, imageSize(1K/2K/4K) }
    - temperature/topP/topK/maxOutputTokens/seed（可选）
  - 参考：gemini_banana_mirror.py 第 5266-5313、5349-5376 行

- 图像编辑（NB2）
  - 请求体：contents.parts = [ {text: 指令}, {inline_data: 输入图像(base64)} ]
  - 端点：同上（generateContent）
  - 参考：gemini_banana_mirror.py 第 7023-7099 行

- 迭代优化开启
  - 打开节点参数“启用迭代优化/连续细化”
  - 系统自动保存本轮 prompt 与输出图像（append_turn + cache_image）
  - 参考代码：图生图第 5418-5423 行；编辑第 7119-7126 / 7225-7233 / 9235-9243 行

## 🔀 镜像站与端点选择（要点）
- Comfly：nano-banana 使用 /v1/chat/completions；其他 Gemini 模型走 /v1/models/{model}:generateContent
- T8：与 Comfly 对齐，部分场景使用 /fal-ai/{model}
- API4GPT：
  - NB2 走 Gemini 原生接口（generateContent）
  - NB1 走 OpenAI 兼容 /v1/images/generations 与 /v1/images/edits
  - 代码内对 API4GPT 做了“禁用 OpenAI 分支”纠偏以强制 NB2 使用 Gemini 原生
- OpenRouter：图像相关统一走 /v1/chat/completions 并支持流式解析

详情可见：gemini_banana_mirror.py 的 build_api_url 与分支判断逻辑。

## ⚙️ 关键参数与兼容性
- aspectRatio：如 1:1 / 16:9 / 9:16 / 4:3 / 3:4（留空则由模型决定）
- imageSize："1K" / "2K" / "4K"（UI 会映射为上述枚举）
- responseModalities：统一为 ["Text", "Image"] 或 ["Image"]（按需求）
- seed：>0 时生效，复现更稳定
- NB1 与 NB2 差异：
  - NB1：OpenAI 格式，/images/generations 或 /images/edits
  - NB2：Gemini 原生 generateContent，imageConfig 支持更完善

参考：NANO_BANANA_2_IMPLEMENTATION.md、GEMINI_BANANA_RESOLUTION_UPDATE.md

## 🔁 迁移指引（从 NB1 到 NB2）
1. 模型选择切换为 gemini-3-pro-image-preview（或 NB2 对应别名）
2. 将 NB1 的 size 参数改为 imageConfig 中的 aspectRatio + imageSize
3. 若使用 API4GPT：确保 NB2 走 generateContent（本仓库已内置纠偏逻辑）
4. 工作流层面：无需改动输出解析，节点内部已兼容

## 🧠 迭代优化（连续细化）
- 生成/编辑后：自动保存 turn 历史与图像缓存
- 下轮执行：可自动承接上一轮输出图像
- 适用节点：Comfly、T8、API4GPT、OpenRouter 等镜像节点与官方节点
- 示例工作流：
  - examples/迭代优化_基础连续细化工作流.json
  - examples/迭代优化_自动承接图像工作流.json
  - examples/迭代优化_多图编辑工作流.json
  - 测试：test_iterative_refinement.py

## 🖼️ 多图编辑增强
- API4GPT 多图编辑采用分阶段重试：代理失败→直连重试
- 更长的超时窗口（默认 300s），减少大图/多图传输失败
- 失败时返回原图并携带详细提示，便于继续工作
- 参考：gemini_banana_mirror.py 第 9090-9243 行

## 🛡️ 安全设置与系统指令（延续 v2.1）
- 5 档安全预设 + 可选自定义系统指令（全节点支持）
- 见主 README 与以下文档：
  - GOOGLE_SEARCH_GROUNDING_GUIDE.md
  - ITERATIVE_REFINEMENT_GUIDE.md

## 🔧 配置示例（Gemini_Banana_config.json）
```json
{
  "mirror_sites": {
    "API4GPT": {
      "url": "https://one.api4gpt.com",
      "api_key": "sk-...",
      "api_format": "openai",
      "models": ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
      "description": "API4GPT 镜像站（NB2 使用 Gemini 原生接口，已内置纠偏）"
    }
  },
  "default_params": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "gen_control": "randomize"
  }
}
```

## 🧪 测试与示例
- 单元/集成测试：
  - test_nano_banana_2.py
  - test_mirror_auto_selection.py
  - test_openrouter_api.py
  - test_iterative_refinement.py
- 工作流示例：
  - examples/Nano-Banana官方API调用.json
  - examples/镜像API调用.json
  - examples/迭代优化_*.json（多套）

## ❗ 已知问题与排障
- 部分镜像站的 imageSize 与分辨率对应存在差异：建议优先选择 1K/2K/4K 标准值
- 代理可能导致多图上传失败：已实现“代理→直连”双策略自动回退
- 超时：大图/多图建议放宽超时至 300s 以上

## 📝 变更日志摘要（与本更新相关）
- NB2：接入 Gemini 原生接口（generateContent），统一 imageConfig 与响应解析
- 镜像：自动选择与纠偏（API4GPT NB2 强制走 Gemini）
- 迭代优化：生成/编辑/多图编辑全覆盖
- 多图编辑：稳健的上传与重试策略
- 放大：集成 smart_upscale，可选 2x~6x

---
如需把本专属更新说明加入主 README，请告知，我可为 README.md 添加入口链接。

