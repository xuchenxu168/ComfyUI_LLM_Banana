# Banana2 提示词模板节点使用指南

## 📖 概述

**Banana2 提示词模板节点**是一个强大的提示词生成工具，它内置了 Gemini-3-Pro-Image-Preview 模型的强大能力，能够根据用户的需求和输入的媒体（图片、视频等），智能生成高质量的 Banana 图像生成/编辑提示词。

如需逐个模板的详细使用说明，请参见《提示词模板详解》：`banana2_template_details.md`。

更多参考：
- 实战案例集：`banana2_use_cases.md`
- 高级配置选项：`banana2_advanced_config.md`

## 🌟 核心特性

 - **内置 Gemini-3-Pro-Image-Preview**：利用最先进的多模态 AI 模型
 - **多种专业提示词模板**：覆盖常见的图像生成、编辑及分镜场景（含 Webtoon 9 宫格、Sora 3/5 宫格、TVC 9 宫格）
 - **多媒体输入支持**：支持最多 4 张图片同时输入
 - **智能提示词优化**：自动将用户简单的描述转化为详细的专业提示词
 - **基于 Google 官方指南**：遵循 Gemini API 图像生成最佳实践

## 🎯 提示词模板类型

### 1. 文本到图片基础模板
**适用场景**：从零开始创建图像
**特点**：
- 详细的主题/对象描述
- 场景和环境设定
- 光照、色彩和氛围控制
- 风格和艺术方向指导
- 构图和视角建议

**示例输入**：
```
用户提示词: 一只可爱的橙色猫咪在星空下
```

**优化后输出**：
```
Create a photorealistic image of an adorable orange tabby cat sitting on a grassy hill under a starry night sky. The cat has bright green eyes and fluffy fur with distinct stripes. The Milky Way is visible overhead with countless stars twinkling against a deep blue-black sky. Soft moonlight illuminates the cat from the left, creating gentle highlights on its fur. The composition uses a low angle perspective, making the cat appear majestic against the cosmic backdrop. Style: cinematic photography, f/2.8, golden hour lighting transitioning to night.
```

### 2. 图片编辑模板
**适用场景**：修改现有图片
**特点**：
- 明确保留的元素
- 详细的修改指令
- 风格一致性保证
- 光照和阴影调整

**示例输入**：
```
用户提示词: 给这只猫戴上一顶魔法师帽子
图片输入: [猫咪照片]
```

**优化后输出**：
```
Edit this image by adding a wizard's hat to the cat. Keep the cat's facial features, fur color, and pose exactly as shown. Add a tall, pointed wizard hat in deep purple with golden stars and moons embroidered on it. The hat should sit naturally on the cat's head, slightly tilted to the right. Adjust the lighting to create a subtle shadow from the hat's brim across the cat's forehead. The hat's texture should be velvet-like, with realistic folds and creases. Maintain the original image's color grading and atmosphere.
```

### 3. 多图合成模板
**适用场景**：组合多张图片的元素
**特点**：
- 明确每张图片的使用元素
- 元素组合和布局指导
- 统一的风格和氛围
- 自然的过渡和融合

**示例输入**：
```
用户提示词: 将第一张图的连衣裙穿在第二张图的模特身上
图片输入: [连衣裙图片] + [模特图片]
```

### 4. 高保真细节保留模板
**适用场景**：需要精确保留关键细节的编辑
**特点**：
- 强调关键细节保护（面部、徽标等）
- 精确的编辑区域定义
- 自然的元素集成
- 质量和风格保持

**示例输入**：
```
用户提示词: 在这个人的T恤上添加公司徽标
图片输入: [人物照片] + [徽标图片]
```

### 5. 草图优化模板
**适用场景**：将草图/简笔画转化为成品
**特点**：
- 保留草图的核心设计
- 添加细节、材质和表面处理
- 风格和色彩增强
- 创意概念实现

**示例输入**：
```
用户提示词: 将这个汽车草图变成逼真的概念车照片
图片输入: [汽车草图]
```

### 6. 角色一致性模板
**适用场景**：生成角色的不同视角/姿势
**特点**：
- 详细的角色特征描述
- 新视角/姿势要求
- 身份和风格一致性
- 表情和情绪控制

**示例输入**：
```
用户提示词: 生成这个角色的侧面视图
图片输入: [角色正面照]
```

### 7. 自定义模板
**适用场景**：特殊需求或自定义工作流
**特点**：
- 完全自定义的模板内容
- 灵活的提示词结构
- 适应特定项目需求

## 🔧 节点参数说明

### 必需参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| api_key | STRING | "" | Gemini API密钥（留空自动从配置读取） |
| user_prompt | STRING | "" | 用户的提示词描述 |
| template_type | COMBO | "文本到图片基础模板" | 选择提示词模板类型 |
| custom_template | STRING | "" | 自定义模板内容（仅在选择"自定义模板"时使用） |
| model | COMBO | "gemini-3-pro-image-preview" | 使用的模型 |
| custom_model | STRING | "" | 自定义模型名称 |
| temperature | FLOAT | 0.7 | 生成的随机性（0.0-1.0） |
| max_output_tokens | INT | 2048 | 最大输出token数 |

### 可选参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 第一张输入图片 |
| image_2 | IMAGE | 第二张输入图片 |
| image_3 | IMAGE | 第三张输入图片 |
| image_4 | IMAGE | 第四张输入图片 |
| system_instruction | STRING | 自定义系统指令 |

### 通用 API 参数（新增）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| base_url | STRING | "" | 自定义/镜像基础地址；留空走配置/默认 |
| version | COMBO | "Auto" | 接口版本：Auto/v1/v1alpha/v1beta |
| auth_mode | COMBO | "auto" | 认证模式：auto/google_xgoog/bearer |

说明：以上三参数位置在 `api_key` 与 `max_output_tokens` 之间；`Gemini-Multimodal` 节点同步支持相同参数。

### 参数优先级与认证说明

- API Key：用户输入 → 供应商/镜像配置（`Gemini_config.json`/`Gemini_Banana_config.json`）→ 全局配置 → 错误提示（缺失）。
- Base URL：用户输入 → 供应商/镜像配置 → 全局配置 → 供应商默认（如 Google 为 `https://generativelanguage.googleapis.com`）。
- Custom：选择自定义供应商/镜像时，必须同时填写 `api_key` 与 `base_url`。
- 认证模式：
  - `auto`：Google 域使用 `x-goog-api-key`，其他域使用 `Authorization: Bearer`；
  - `google_xgoog`：强制 `x-goog-api-key`；
  - `bearer`：强制 `Authorization: Bearer`。
- 版本选择（`version=Auto`）：Google 域在 `media_resolution=Auto` 使用 `v1beta`，否则 `v1alpha`；非 Google 域默认使用 `v1`。

### 端点构建规则

- 完整端点直通：当 `base_url` 已包含 `/models/<model>:generateContent` 时原样使用；
- 通用拼接：
  - `base_url/<version>/models/<model>:generateContent`；
  - 当 `base_url` 以 `/v1|/v1alpha|/v1beta` 结尾时，拼接为 `.../models/<model>:generateContent`。

### 输出

| 输出名 | 类型 | 说明 |
|--------|------|------|
| banana_prompt | STRING | 优化后的Banana提示词 |
| raw_response | STRING | 原始API响应 |

## 💡 使用示例

### 示例 1：文本生成图片

```
模板类型: 文本到图片基础模板
用户提示词: 一个未来主义的城市天际线，霓虹灯闪烁
Temperature: 0.7
```

### 示例 2：图片编辑

```
模板类型: 图片编辑模板
用户提示词: 将背景改为日落海滩
图片输入: [人物照片]
Temperature: 0.6
```

### 示例 3：多图合成

```
模板类型: 多图合成模板
用户提示词: 创建一个专业的电商时尚照片
图片输入: [服装图] + [模特图]
Temperature: 0.5
```

## 📝 最佳实践

### 1. 提示词编写技巧
- **具体化**：使用具体的描述而非抽象概念
- **提供上下文**：说明图片的用途和目标
- **迭代优化**：根据输出结果逐步调整
- **分步指令**：复杂场景分解为多个步骤

### 2. 参数调整建议
- **Temperature**：
  - 0.3-0.5：需要精确控制和一致性
  - 0.6-0.8：平衡创意和控制
  - 0.8-1.0：最大创意和多样性

- **Max Output Tokens**：
  - 1024：简单提示词
  - 2048：标准使用（推荐）
  - 4096+：复杂场景和详细描述

### 3. 图片输入建议
- 使用高质量、清晰的图片
- 确保图片内容与提示词相关
- 最多使用3张图片以获得最佳效果
- 图片顺序很重要，按重要性排列

## 🔗 工作流集成

### 典型工作流

```
1. [Load Image] → [Banana2 Prompt Template]
2. [Banana2 Prompt Template] → [Gemini Banana Image Generation]
3. [Gemini Banana Image Generation] → [Save Image]
```

### 高级工作流

```
1. [Load Image] × 2 → [Banana2 Prompt Template]
2. [Banana2 Prompt Template] → [Text Combine]
3. [Text Combine] → [Gemini Banana Image Generation]
4. [Gemini Banana Image Generation] → [Image Compare] → [Save Image]
```

## ⚙️ 配置说明

在 `config.json` 中配置 API 密钥：

```json
{
  "multimodal_api_key": "your-gemini-api-key-here"
}
```

## 🐛 故障排除

### 问题：API密钥错误
**解决方案**：
1. 检查 `config.json` 中的 `multimodal_api_key` 配置
2. 或在节点中手动输入有效的 API 密钥

### 问题：生成的提示词不够详细
**解决方案**：
1. 增加 `max_output_tokens` 值
2. 在用户提示词中提供更多细节
3. 尝试不同的 `temperature` 值

### 问题：图片输入无法识别
**解决方案**：
1. 确保图片格式正确（PNG、JPEG等）
2. 检查图片大小是否合理
3. 尝试使用更清晰的图片

## 📚 参考资源

- [Google Gemini API 图像生成文档](https://ai.google.dev/gemini-api/docs/image-generation)
- [Banana 图像生成最佳实践](https://ai.google.dev/gemini-api/docs/image-generation#template)
- [提示词工程指南](https://ai.google.dev/gemini-api/docs/prompting-strategies)

## 📌 模板新增与使用示例

### 编辑-9宫格TVC广告分镜头

```
模板类型: 编辑-9宫格TVC广告分镜头
用户提示词: 一支强调城市清晨能量的运动饮料TVC，强调动感与清爽氛围
图片输入: [形象参考：角色造型/场景风格/色彩基调]
```

输出采用严格的中文“提示词”九镜头格式（3×3，16:9），结构如下：

---

**提示词**：

### 镜头1（全景）：
- 场景：……
- 角色：……
- 镜头语言：……
- 光影风格：……

### 镜头2（中景）：
- 场景：……
- 角色动作：……
- 镜头语言：……
- 光影风格：……

### 镜头3（近景）：
- 场景：……
- 镜头语言：……
- 光影风格：……

### 镜头4（全景）：
- 场景：……
- 角色动作：……
- 镜头语言：……
- 光影风格：……

### 镜头5（中景）：
- 场景：……
- 角色动作：……
- 镜头语言：……
- 光影风格：……

### 镜头6（特写）：
- 场景：……
- 镜头语言：……
- 光影风格：……

### 镜头7（全景）：
- 场景：……
- 角色动作：……
- 镜头语言：……
- 光影风格：……

### 镜头8（近景）：
- 场景：……
- 镜头语言：……
- 光影风格：……

### 镜头9（定格）：
- 场景：……
- 文字元素：品牌名与广告语（如需）
- 镜头语言：……
- 光影风格：……

适用场景：需要商业级统一风格的广告分镜，强调角色/产品一致性与连贯叙事。

### 编辑-9宫格TVC广告分镜头线稿

```
模板类型: 编辑-9宫格TVC广告分镜头线稿
用户提示词: 一支强调城市清晨能量的运动饮料TVC，画面以黑白线稿呈现
图片输入: [形象参考：角色造型/场景风格/构图基调]
```

输出采用严格的中文“提示词”九镜头格式（3×3，16:9），结构与「编辑-9宫格TVC广告分镜头」完全一致，仅将风格限定为线稿：

---

**提示词**（线稿风格）：

### 镜头1（全景）：
- 场景：……
- 角色：……
- 镜头语言：……
- 光影风格：线性排线/网点表现，不使用色块填充。

### 镜头2（中景）：
- 场景：……
- 角色动作：……
- 镜头语言：……
- 光影风格：轮廓线清晰，疏密排线体现层次。

### 镜头3（近景）：
- 场景：……
- 镜头语言：……
- 光影风格：方向性排线强化体积与动感。

### 镜头4（全景）：
- 场景：……
- 角色动作：……
- 镜头语言：……
- 光影风格：以线条密度区分空间层级，避免色块。

### 镜头5（中景）：
- 场景：……
- 角色动作：……
- 镜头语言：……
- 光影风格：排线对比明确，形体立体。

### 镜头6（特写）：
- 场景：……
- 镜头语言：……
- 光影风格：交叉排线塑形，细节清晰。

### 镜头7（全景）：
- 场景：……
- 角色动作：……
- 镜头语言：……
- 光影风格：线条节奏表现灯光反射与空间。

### 镜头8（近景）：
- 场景：……
- 镜头语言：……
- 光影风格：柔和排线过渡，突出轮廓与质感。

### 镜头9（定格）：
- 场景：……
- 文字元素：品牌名与广告语（线稿呈现）
- 镜头语言：……
- 光影风格：轮廓强化，形成记忆点。

线稿专用要求：
- 风格统一为黑白线描/线稿，不着色；明暗仅用排线或网点表达。
- 避免渐变、真实材质渲染、复杂色彩；仅保留线条与结构。
- 画面保持商业级分镜规范，干净有力，连贯叙事。

适用场景：需要输出线稿风格的分镜提案、前期分镜评审、快速视觉草图。

## 🎉 更新日志

### v1.0.0 (2025-11-25)
- 初始版本发布
- 支持7种提示词模板
- 集成 Gemini-3-Pro-Image-Preview
- 支持多图输入
- 基于 Google 官方提示词指南

