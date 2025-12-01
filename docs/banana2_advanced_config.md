# Banana2 高级配置选项与实践

本指南面向进阶用户，说明 Banana2-提示词模板节点的高级参数、端点与认证策略、输出控制与性能优化建议，确保跨供应商与模型的一致体验。

## 通用 API 参数（统一新增）
- `base_url`：请求基址；支持 Gemini 原生与 OpenAI 兼容路径。
  - 原生：`https://generativelanguage.googleapis.com/`
  - 兼容：`https://generativelanguage.googleapis.com/v1beta/openai/`
- `version`：API 版本；建议与实现一致（如 `v1beta`）。
- `auth_mode`：认证模式；统一策略：
  - `auto`：自动选择，按端点类型构造正确的认证头。
  - `x-goog-api-key`：使用 `x-goog-api-key: <key>`（Gemini 原生）。
  - `bearer`：使用 `Authorization: Bearer <key>`（OpenAI 兼容路径）。
- 参数优先级（回退）：
  - 用户输入优先 → 供应商配置 → 全局配置 → 默认值。

## 模型与模板
- `model` / `custom_model`：选择内置或自定义模型 ID（如 `gemini-3.5-pro-preview`、`gemini-2.5-flash`）。
- `template_type`：选择具体模板；见 `banana2_template_details.md`。

## 输出控制（GenerationConfig）
- `temperature`：创意幅度；写实/保真建议 0.3–0.6，风格化/抽象可 0.6–0.9。
- `max_output_tokens`：输出长度；短文 512–768，分格/详述 768–1200+。
- 可选（若实现支持）：`top_p`、`top_k`、`presence_penalty`、`frequency_penalty`。

## 媒体输入
- 支持最多 4 张图片作为参考；小型图片建议使用 `inline_data`（内联字节）。
- 多图时强调：
  - 风格/色板一致；
  - 透视/阴影/反射匹配；
  - 避免过多或过大图片导致不稳定。

## Sora 模板提示
- `编辑-Sora动漫3宫格/5宫格`：视频/漫画叙事模板。
- 时长约束：5 宫格示例严格 25 秒，总时长需在提示中明确说明。
- 建议在提示中清晰标注每一格的镜头、动作与节奏。
- 参阅：`SORA_3Panel_Template_使用指南.md` 与 `SORA_5Panel_Template_使用指南.md`。

## 端点与认证（统一规则）
- 原生生成：`/v1beta/models/{model}:generateContent`
- 原生批量：`/v1beta/models/{model}:batchGenerateContent`
- 兼容路径：`/v1beta/openai/...`（认证改为 Bearer）
- 统一构造：
  - 端点通过 `base_url + version + path` 规则拼接；
  - 认证头通过 `auth_mode` 决定，`auto` 模式自动匹配。

## 批量处理（可行性说明）
- 小批量内嵌（Inline Requests）：同构的 `GenerateContentRequest` 列表；总大小 ≤ 20MB。
- 大批量文件模式：JSONL + 文件 API；最大 2GB。
- 适用场景：非紧急高吞吐；计费约标准费用的 50%。
- 参考（官方）：Gemini Batch API 与 API 参考。

## 性能与稳定性建议
- 写实/保真：降低 `temperature`，明确光照与镜头参数；控制输出长度。
- 风格化/抽象：提高 `temperature`，明确风格与色板；避免过度抽象的描述。
- 多图合成：重点强调阴影/透视与反射；必要时降低 `temperature` 以减少风格漂移。
- 文字内容：中英文分别用引号标注；指定字体与版式层级。
- Token 管理：复杂分格或详细说明时提高 `max_output_tokens`，并尽量结构化输入。

## 故障排除
- 输出不相关 → 补充结构化要点；明确风格/色板与镜头。
- 细节丢失 → 使用保真模板；降低 `temperature` 并明确关键区域。
- 风格偏差 → 增加风格参考图；提示中强调“一致性”。
- 认证错误 → 检查 `auth_mode` 与 `base_url` 路径（原生 vs 兼容）。

## 参考文档
- 模板详解：`docs/banana2_template_details.md`
- 实战案例：`docs/banana2_use_cases.md`
- 快速开始：`docs/banana2_quick_start.md`
- Sora 系列指南与示例：`docs/SORA_*.md`