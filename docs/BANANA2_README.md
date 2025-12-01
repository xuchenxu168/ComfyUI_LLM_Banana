# Banana2 提示词模板节点

> 基于 Gemini-3-Pro-Image-Preview 的智能提示词生成工具

## 🌟 项目简介

**Banana2 提示词模板节点**是一个专为 ComfyUI 设计的高级提示词生成工具。它内置了 Google 最先进的 **Gemini-3-Pro-Image-Preview** 多模态 AI 模型，能够根据用户的简单描述和输入的图片，自动生成专业、详细、高质量的 Banana 图像生成提示词。

### 为什么选择 Banana2？

- 🚀 **强大的 AI 引擎**：基于 Gemini-3-Pro-Image-Preview，理解力和创造力[object Object]业模板库**：7 种精心设计的提示词模板，覆盖各种使用场景
- 🎨 **多模态输入**：支持文本 + 最多 4 张图片的组合输入
- 🎯 **遵循最佳实践**：基于 Google 官方图像生成提示词指南
- 💡 **智能优化**：自动将简单描述转化为详细的专业提示词
-[object Object]成**：完美融入 ComfyUI 工作流

## 📦 功能特性

### 核心功能

| 功能 | 描述 |
|------|------|
| **智能提示词生成** | 利用 Gemini-3-Pro 的强大能力，自动优化和扩展用户提示词 |
| **多种模板支持** | 7 种专业模板：文本生成、图片编辑、多图合成、细节保留、草图优化、角色一致性、自定义 |
| **多模态输入** | 支持纯文本、单图、多图（最多4张）等多种输入方式 |
| **参数可调** | Temperature、Max Tokens 等参数可自由调整 |
| **自动配置** | 支持从配置文件自动读取 API 密钥 |
| **备用机制** | 当 API 不可用时提供基础提示词生成 |

### 提示词模板

#### 1️⃣ 文本到图片基础模板
将简单的文本描述转化为详细的图像生成提示词
```
输入: "一只猫在月光下"
输出: 详细的场景描述，包含光照、构图、风格等专业要素
```

#### 2️⃣ 图片编辑模板
为图片编辑任务生成精确的指令
```
输入: 图片 + "改变背景为海滩"
输出: 保留原图元素，详细的编辑指令
```

#### 3️⃣ 多图合成模板
组合多张图片的元素创建新图像
```
输入: 服装图 + 模特图 + "电商照片"
输出: 详细的合成和布局指令
```

#### 4️⃣ 高保真细节保留模板
在编辑时精确保留关键细节
```
输入: 人物图 + Logo图 + "添加到T恤上"
输出: 强调细节保护的精确编辑指令
```

#### 5️⃣ 草图优化模板
将草图转化为成品图像
```
输入: 汽车草图 + "概念车渲染"
输出: 保留设计，添加细节和材质的指令
```

#### 6️⃣ 角色一致性模板
生成角色的不同视角和姿势
```
输入: 角色正面图 + "侧面视图"
输出: 保持角色特征一致的生成指令
```

#### 7️⃣ 自定义模板
完全自定义的模板内容
```
输入: 自定义模板 + 用户提示词
输出: 基于自定义逻辑的提示词
```

## 🚀 快速开始

### 安装

节点已集成在 ComfyUI-Gemini-3 插件中，无需额外安装。

### 配置

在 `ComfyUI-Gemini-3/config.json` 中添加 API 密钥：

```json
{
  "multimodal_api_key": "your-gemini-api-key-here"
}
```

### 基础使用

1. **添加节点**：在 ComfyUI 中右键 → `Add Node` → `🍌 Banana` → `Banana2-提示词模板`

2. **配置参数**：
   - 选择模板类型
   - 输入用户提示词
   - （可选）连接图片输入

3. **连接下游节点**：将输出连接到 Gemini Banana 图像生成节点

4. **运行工作流**：生成优化后的提示词

## 📖 使用示例

### 示例 1：简单文本生成

```
节点配置：
- 模板类型: 文本到图片基础模板
- 用户提示词: "一个未来主义的城市"
- Temperature: 0.7

输出：详细的城市场景描述，包含建筑风格、光照、氛围等
```

### 示例 2：产品图编辑

```
节点配置：
- 模板类型: 图片编辑模板
- 用户提示词: "改善光照，添加阴影"
- 图片输入: [产品照片]
- Temperature: 0.5

输出：保留产品特征的详细编辑指令
```

### 示例 3：时尚电商合成

```
节点配置：
- 模板类型: 多图合成模板
- 用户提示词: "专业电商时尚照片"
- 图片输入: [连衣裙] + [模特]
- Temperature: 0.6

输出：详细的服装合成和拍摄指令
```

## 🎯 应用场景

| 场景 | 推荐模板 | Temperature |
|------|----------|-------------|
| 社交媒体内容创作 | 文本到图片 | 0.7-0.8 |
| 电商产品图优化 | 图片编辑 | 0.4-0.6 |
| 概念设计可视化 | 草图优化 | 0.6-0.8 |
| 角色设计 | 角色一致性 | 0.5-0.7 |
| 品牌营销素材 | 高保真细节 | 0.3-0.5 |
| 艺术创作 | 自定义模板 | 0.7-0.9 |

## 📊 技术架构

```
用户输入 → 模板选择 → 提示词构建 → Gemini-3-Pro → 优化引擎 → 输出
    ↓          ↓            ↓              ↓            ↓         ↓
  文本      7种模板      上下文组合      AI理解      智能优化   专业提示词
  图片                   系统指令       多模态       结构化
```

## 🔧 参数说明

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| api_key | String | - | Gemini API密钥 |
| user_prompt | String | - | 用户的提示词描述 |
| template_type | Combo | 7种 | 提示词模板类型 |
| model | Combo | 2种 | gemini-3-pro-image-preview / custom |
| temperature | Float | 0.0-1.0 | 生成的随机性 |
| max_output_tokens | Int | 256-8192 | 最大输出长度 |
| image (1-4) | Image | - | 可选的图片输入 |
| system_instruction | String | - | 自定义系统指令 |
| base_url | String | - | 自定义/镜像基础地址（留空走配置/默认） |
| version | Combo | Auto/v1/v1alpha/v1beta | 接口版本选择，Auto 智能匹配 |
| auth_mode | Combo | auto/google_xgoog/bearer | 认证模式（Auto按域自动选择） |

## 📚 文档

- [完整使用指南](banana2_prompt_template_guide.md)
- 🚀 [快速开始教程](banana2_quick_start.md)
- 💡 [最佳实践](banana2_best_practices.md)
- 🎨 [实战案例](banana2_use_cases.md)

## 🌐 通用 API 与新参数（新增）

- 新增参数：`base_url`、`version`、`auth_mode` 位于 `api_key` 与 `max_output_tokens` 之间。
- 端点构建：
  - 完整端点（含 `/models/<model>:generateContent`）直通使用；
  - 否则按 `base_url/<version>/models/<model>:generateContent` 拼接；若 `base_url` 以 `/v1|/v1alpha|/v1beta` 结尾，则拼为 `.../models/<model>:generateContent`。
- 认证模式：
  - `auto`：Google 域使用 `x-goog-api-key`，其他域使用 `Authorization: Bearer`；
  - `google_xgoog`：强制 `x-goog-api-key`；
  - `bearer`：强制 `Authorization: Bearer`。
- 版本选择（`version=Auto`）：
  - Google 域：`media_resolution=Auto` 用 `v1beta`；否则 `v1alpha`；
  - 非 Google 域：默认 `v1`。
- 参数优先级：用户输入 → 供应商/镜像配置（`Gemini_config.json` / `Gemini_Banana_config.json`）→ 全局配置 → 默认。
- 说明：`Gemini-Multimodal` 节点也同步支持这三参数，体验一致。

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

## 📄 许可证

本项目遵循 MIT 许可证。

## 🙏 致谢

- Google Gemini API 团队提供的强大 AI 能力
- ComfyUI 社区的支持和反馈
- 所有贡献者的辛勤工作

## 📞 联系方式

- **作者**：Ken-Chen
- **项目**：ComfyUI-Gemini-3
- **版本**：v1.0.0

---

**Made with ❤️ for the ComfyUI Community[object Object] **Banana2 - 让 AI 图像生成更简单、更专业！** 🍌
