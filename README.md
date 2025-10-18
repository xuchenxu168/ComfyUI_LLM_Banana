# ComfyUI LLM Banana - 多模态AI助手节点集合

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

一个功能强大的ComfyUI自定义节点集合，集成了多种大语言模型和AI服务，支持文本生成、图像生成、图像分析、翻译等多种AI功能。

## 🌟 主要特性

- **🍌 Banana系列节点**: 专业的图像编辑和多图处理能力，支持批量操作
- **🎨 强大的图像编辑**: 智能图像到图像编辑、多图像同时编辑、内容感知处理
- **🔄 双API架构**: 官方API节点 + 镜像API节点，确保服务稳定性和可用性
- **🚀 多模型支持**: Gemini、GLM、OpenRouter、Nano-Banana等多种AI模型
- **📸 专业图像生成**: 支持文本到图像生成，多种尺寸和质量选项
- **🔍 智能图像分析**: 图像描述、内容分析、视觉问答、场景理解
- **🌐 多语言翻译**: 支持50+语言的专业翻译服务
- **⚡ 高性能处理**: AI超分辨率、智能裁剪、批量质量优化
- **🛠️ 灵活配置**: 支持多种API端点、代理设置和自定义参数
- **🛡️ 安全设置**: 5个安全级别预设，完全符合Google Gemini官方标准 ⭐ NEW
- **🎯 系统指令**: 7个专业预设模板 + 自定义指令支持 ⭐ NEW
- **📐 标准化API**: Response Modalities格式统一，完全兼容官方API ⭐ NEW
- **🧹 优化体验**: 启动输出精简68%，加载更快更清晰 ⭐ NEW

## 📸 功能展示

### 视频反推及扩写
![ComfyUI-GLM4](/image/PixPin_2025-06-28_15-59-36.png)

### Flux 提示词生成
#### 反推
![](/image/PixPin_2025-06-21_23-09-55.png)

#### 扩写
![](/image/PixPin_2025-06-21_23-09-47.png)

## 📦 安装指南

### 方法一：Git克隆（推荐）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI_LLM_Banana.git
cd ComfyUI_LLM_Banana
pip install -r requirements.txt
```

### 方法二：手动下载

1. 下载项目压缩包并解压到 `ComfyUI/custom_nodes/ComfyUI_LLM_Banana`
2. 安装依赖：
```bash
cd ComfyUI/custom_nodes/ComfyUI_LLM_Banana
pip install -r requirements.txt
```

### 依赖要求

- Python 3.8+
- PyTorch
- Pillow
- requests
- numpy
- google-generativeai (可选，用于官方Gemini API)

## 🔧 配置设置

### API密钥配置

在使用前，您需要配置相应的API密钥：

1. **Gemini API**: 获取Google AI Studio API密钥
2. **GLM API**: 获取智谱AI API密钥 - [bigmodel.cn](https://www.bigmodel.cn/)
3. **OpenRouter API**: 获取OpenRouter API密钥
4. **Nano-Banana**: 获取相应服务的API密钥

### 代理设置

如果需要使用代理，可以在节点中配置：
- HTTP代理：`http://proxy:port`
- SOCKS代理：`socks5://proxy:port`

## 📋 节点说明

### 🍌 Banana系列核心节点

ComfyUI LLM Banana的核心优势在于其强大的**Banana系列节点**，提供了业界领先的图像编辑和多图处理能力。

#### 🎯 双API架构设计

**官方API节点 (3个核心节点):**
- **Gemini Banana** - Google官方API调用
- **GLM Banana** - 智谱AI官方API调用
- **OpenRouter Banana** - OpenRouter官方API调用

**镜像API节点 (3个镜像节点):**
- **Gemini Banana Mirror** - 镜像站点调用，突破网络限制
- **Comfly Nano Banana** - Comfly镜像服务，支持Nano-Banana模型
- **T8 Mirror Banana** - T8镜像站点，提供稳定的备用服务

---

### 🤖 官方API节点详解

#### Gemini Banana (Google官方API)
- **功能**: 使用Google官方Gemini API进行多模态AI处理
- **核心特性**:
  - ✅ **图像到图像编辑**: 智能修改现有图像内容
  - ✅ **多图像批量处理**: 同时处理多张图像
  - ✅ **视觉理解**: 深度图像内容分析
  - ✅ **文本生成**: 高质量文本创作
  - ✅ **安全设置**: 5个安全级别预设，精确控制内容过滤 ⭐ NEW
  - ✅ **系统指令**: 7个专业模板 + 自定义指令支持 ⭐ NEW
- **输入参数**:
  - `prompt`: 文本提示词或编辑指令
  - `image`: 输入图像(支持多图)
  - `api_key`: Google API密钥
  - `model`: 模型选择 (gemini-1.5-pro, gemini-1.5-flash等)
  - `temperature`: 创造性控制 (0.0-2.0)
  - `max_output_tokens`: 最大输出长度
  - `safety_level`: 安全级别 (default, strict, moderate, permissive, off) ⭐ NEW
  - `system_instruction_preset`: 系统指令预设 (image_generation, creative_artist等) ⭐ NEW
  - `custom_system_instruction`: 自定义系统指令 ⭐ NEW
- **输出**: 编辑后的图像 + 描述文本
- **特性**:
  - 🔥 **强大的图像编辑能力**: 支持局部修改、风格转换、内容替换
  - 🔥 **多图像协同编辑**: 可同时编辑多张相关图像
  - 🔥 **智能内容理解**: 准确识别图像内容并进行精确编辑
  - 🔥 **安全可控**: 灵活的安全设置，适应不同应用场景 ⭐ NEW
  - 🔥 **行为定制**: 系统指令让AI扮演特定角色，提升输出质量 ⭐ NEW

#### GLM Banana (智谱AI官方API)
- **功能**: 使用智谱AI官方GLM模型，专门优化中文处理
- **核心特性**:
  - ✅ **中文图像理解**: 专业的中文图像描述和分析
  - ✅ **中文文本生成**: 高质量中文内容创作
  - ✅ **多模态处理**: 图文结合的智能处理
- **输入参数**:
  - `prompt`: 中文提示词
  - `image`: 输入图像(可选)
  - `api_key`: 智谱AI API密钥
  - `model`: GLM模型版本 (GLM-4, GLM-4-Flash, GLM-4V等)
  - `temperature`: 温度参数
- **输出**: 中文文本内容或图像分析结果
- **特性**:
  - 🔥 **中文优化**: 专门针对中文语境优化
  - 🔥 **免费额度**: GLM-4-Flash提供大量免费调用
  - 🔥 **多模态能力**: GLM-4V支持图像理解

#### OpenRouter Banana (多模型聚合API)
- **功能**: 通过OpenRouter平台访问多种顶级AI模型
- **核心特性**:
  - ✅ **模型选择丰富**: Claude、GPT、Llama、Mistral等
  - ✅ **统一接口**: 一个API密钥访问多种模型
  - ✅ **成本优化**: 根据需求选择最适合的模型
- **输入参数**:
  - `prompt`: 提示词
  - `model`: 模型选择 (claude-3-opus, gpt-4, llama-3等)
  - `api_key`: OpenRouter API密钥
  - `temperature`: 创造性参数
- **输出**: 生成的文本内容
- **特性**:
  - 🔥 **模型多样性**: 支持20+种主流AI模型
  - 🔥 **灵活计费**: 按使用量付费，成本可控
  - 🔥 **高可用性**: 多个模型提供商确保服务稳定

---

### 🌐 镜像API节点详解

#### Gemini Banana Mirror (镜像站点)
- **功能**: 通过镜像站点调用Gemini API，突破网络访问限制
- **核心特性**:
  - ✅ **网络优化**: 解决官方API访问困难的问题
  - ✅ **完整功能**: 保持与官方API相同的功能特性
  - ✅ **图像编辑**: 支持图像到图像的智能编辑
  - ✅ **多图处理**: 批量处理多张图像
- **输入参数**:
  - `prompt`: 编辑指令或描述
  - `image`: 输入图像(支持多图)
  - `api_key`: 镜像站点API密钥
  - `base_url`: 镜像站点地址
  - `model`: Gemini模型选择
- **输出**: 编辑后的图像 + 处理说明
- **特性**:
  - 🔥 **突破限制**: 解决网络访问问题
  - 🔥 **稳定服务**: 多个镜像站点备选
  - 🔥 **完整兼容**: 与官方API功能完全一致

#### Comfly Nano Banana (Comfly镜像服务)
- **功能**: 专门的图像生成和编辑服务，支持Nano-Banana模型
- **核心特性**:
  - ✅ **专业图像生成**: 高质量文本到图像生成
  - ✅ **图像编辑**: 基于现有图像进行智能修改
  - ✅ **多尺寸支持**: 支持各种图像尺寸和比例
  - ✅ **种子控制**: 可重现的图像生成结果
- **输入参数**:
  - `prompt`: 图像描述或编辑指令
  - `image`: 输入图像(编辑模式)
  - `api_url`: Comfly服务地址
  - `api_key`: API密钥
  - `size`: 图像尺寸 (512x512, 1024x1024等)
  - `seed`: 随机种子值
- **输出**: 生成或编辑后的图像
- **特性**:
  - 🔥 **快速生成**: 优化的图像生成速度
  - 🔥 **高质量输出**: 专业级图像质量
  - 🔥 **灵活控制**: 丰富的参数调节选项

#### T8 Mirror Banana (T8镜像站点)
- **功能**: T8镜像站点服务，提供稳定的AI图像处理能力
- **核心特性**:
  - ✅ **多模型支持**: 支持多种图像生成模型
  - ✅ **稳定服务**: 高可用性的镜像服务
  - ✅ **批量处理**: 支持多图像批量操作
  - ✅ **格式兼容**: 支持多种图像格式输入输出
- **输入参数**:
  - `prompt`: 处理指令
  - `image`: 输入图像
  - `api_url`: T8镜像地址
  - `model`: 模型选择
  - `quality`: 质量设置
- **输出**: 处理后的图像
- **特性**:
  - 🔥 **服务稳定**: 专业的镜像服务提供商
  - 🔥 **多模型**: 集成多种AI图像处理模型
  - 🔥 **高性能**: 优化的处理速度和质量

---

### 🎨 强大的图像编辑和生成功能

#### 🔥 图像到图像编辑 (Image-to-Image Editing)
**Banana系列节点的核心优势**

**支持的编辑类型:**
- **局部内容修改**: 替换图像中的特定对象或区域
- **风格转换**: 改变图像的艺术风格或色调
- **内容增强**: 提升图像质量、细节和清晰度
- **场景重构**: 修改背景、环境或整体构图
- **多图协同**: 同时编辑多张相关图像，保持一致性

**编辑工作流程:**
1. **输入原图**: 加载需要编辑的图像
2. **编辑指令**: 提供详细的修改描述
3. **智能处理**: AI理解图像内容和编辑需求
4. **精确修改**: 保持原图结构，精确执行编辑
5. **质量优化**: 自动优化编辑后的图像质量

#### 🔥 多图像批量编辑 (Multi-Image Editing)
**业界领先的批量处理能力**

**批量编辑特性:**
- **同时处理**: 一次性编辑多张图像
- **一致性保持**: 确保多图编辑结果风格统一
- **关联处理**: 理解图像间的关联关系
- **批量优化**: 统一的质量和风格标准
- **效率提升**: 大幅提高工作效率

**适用场景:**
- 📸 **摄影后期**: 批量调色、风格统一
- 🎨 **设计工作**: 系列图像的风格调整
- 🛍️ **电商产品**: 产品图片的批量优化
- 📱 **社交媒体**: 内容创作的批量处理

#### Gemini Banana Image Generator (专业图像生成)
- **功能**: 高级文本到图像生成 + 图像编辑
- **核心特性**:
  - ✅ **文本生图**: 根据描述生成高质量图像
  - ✅ **图像编辑**: 基于现有图像进行智能修改
  - ✅ **多图处理**: 同时处理多张图像
  - ✅ **专业控制**: 丰富的摄影和艺术参数
- **输入参数**:
  - `prompt`: 图像描述或编辑指令
  - `image`: 输入图像(编辑模式，可多图)
  - `size`: 图像尺寸 (512x512, 1024x1024, 1024x1792等)
  - `quality`: 质量等级 (Basic, Professional, Premium, Masterpiece)
  - `style`: 风格控制 (写实、艺术、卡通、电影等)
  - `detail_level`: 细节等级控制
  - `camera_control`: 专业相机参数
  - `lighting_control`: 灯光设置
  - `template_selection`: 预设模板选择
- **输出**: 生成或编辑后的高质量图像
- **特性**:
  - 🔥 **专业级质量**: 支持4K+超高清输出
  - 🔥 **智能编辑**: 理解图像内容，精确修改
  - 🔥 **批量处理**: 高效的多图像处理能力

#### Nano-Banana Image Generator (快速生成)
- **功能**: 快速图像生成和编辑服务
- **核心特性**:
  - ✅ **高速生成**: 优化的生成速度
  - ✅ **种子控制**: 可重现的生成结果
  - ✅ **多尺寸**: 灵活的尺寸选择
  - ✅ **镜像支持**: 多个服务节点备选
- **输入参数**:
  - `prompt`: 生成或编辑指令
  - `image`: 输入图像(编辑模式)
  - `size`: 尺寸选择
  - `seed`: 随机种子值
  - `api_url`: 服务端点选择
- **输出**: 快速生成的高质量图像
- **特性**:
  - 🔥 **极速处理**: 秒级图像生成
  - 🔥 **稳定输出**: 种子控制确保结果可重现
  - 🔥 **多节点**: 负载均衡，服务稳定

### 🔍 图像分析节点

#### Gemini Vision Analyzer
- **功能**: 图像内容分析和描述
- **输入**:
  - `image`: 输入图像
  - `prompt`: 分析提示词
  - `api_key`: Gemini API密钥
  - `model`: 视觉模型选择
- **输出**: 图像描述文本
- **特性**:
  - 智能场景识别
  - 详细内容描述
  - 支持多语言输出

#### JoyCaption (图像描述)
- **功能**: 专业图像标注和描述
- **输入**:
  - `image`: 输入图像
  - `caption_type`: 描述类型
  - `caption_length`: 描述长度
- **输出**: 结构化图像描述
- **特性**: 适合训练数据标注，支持多种描述风格

### 🌐 翻译节点

#### Gemini Banana Translation
- **功能**: 多语言文本翻译
- **输入**:
  - `text`: 源文本
  - `source_language`: 源语言
  - `target_language`: 目标语言
  - `api_key`: API密钥
- **输出**: 翻译后的文本
- **支持语言**: 中文、英文、日文、韩文、法文、德文等50+语言
- **特性**:
  - 上下文感知翻译
  - 专业术语处理
  - 批量翻译支持

### 🎯 图像处理节点

#### AI Image Upscaler
- **功能**: AI超分辨率图像放大
- **输入**:
  - `image`: 低分辨率图像
  - `scale_factor`: 放大倍数
  - `model_type`: 放大算法选择
- **输出**: 高分辨率图像
- **特性**:
  - 多种放大算法 (Real-ESRGAN, ESRGAN, Waifu2x)
  - 细节保持优化
  - 批量处理支持

#### Smart Image Resizer
- **功能**: 智能图像尺寸调整
- **输入**:
  - `image`: 原图像
  - `target_size`: 目标尺寸
  - `fill_strategy`: 填充策略
  - `fill_color`: 填充颜色
- **输出**: 调整后的图像
- **特性**:
  - 智能裁剪
  - 内容感知填充
  - 宽高比保持

### 🔧 工具节点

#### Universal Subject Detection
- **功能**: 通用主体检测
- **输入**: 图像
- **输出**: 主体检测结果
- **特性**: 自动识别图像中的主要对象

#### Adaptive Enhancement
- **功能**: 自适应图像增强
- **输入**: 图像、增强参数
- **输出**: 增强后的图像
- **特性**: 智能调整对比度、亮度、饱和度

## 🚀 使用示例

### 🆕 安全设置与系统指令使用 (v2.1新功能)

**使用安全设置控制内容:**
1. 添加 "Gemini Banana" 节点
2. 设置 `safety_level` 参数：
   - `strict` - 严格模式，适合儿童内容
   - `moderate` - 中等模式，适合一般应用
   - `permissive` - 宽松模式，适合创意工作
   - `off` - 关闭过滤，适合专业用途
3. 执行工作流，享受精确的内容控制

**使用系统指令提升质量:**
1. 添加 "Gemini Banana Image Generator" 节点
2. 选择 `system_instruction_preset`：
   - `image_generation` - 图像生成助手
   - `creative_artist` - 创意艺术家
   - `professional_photographer` - 专业摄影师
   - `technical_expert` - 技术专家
3. 或者使用 `custom_system_instruction` 输入自定义指令
4. 执行工作流，获得更专业的输出

**组合使用示例 - 儿童教育应用:**
```
safety_level: strict
system_instruction_preset: friendly_helper
prompt: "创建一个友好的卡通角色教孩子们学习数字"
```

**组合使用示例 - 专业摄影:**
```
safety_level: moderate
system_instruction_preset: professional_photographer
custom_system_instruction: "你是一位专业的人像摄影师，擅长捕捉自然光线和情感表达"
prompt: "拍摄一张温馨的家庭肖像照"
```

### 🔥 图像到图像编辑工作流 (核心功能)

**单图编辑示例:**
1. 使用 "Load Image" 节点加载原始图像
2. 添加 "Gemini Banana" 或 "Gemini Banana Mirror" 节点
3. 连接图像输入
4. 输入编辑指令："将这张照片的背景改为海滩日落场景，保持人物不变"
5. 设置模型为 "gemini-1.5-pro"
6. 配置API密钥或镜像地址
7. 连接到图像预览节点
8. 执行工作流，获得编辑后的图像

**多图批量编辑示例:**
1. 使用多个 "Load Image" 节点加载图像序列
2. 添加 "Gemini Banana" 节点
3. 连接所有图像输入
4. 输入批量编辑指令："统一将这些产品图片的背景改为纯白色，保持产品细节"
5. 设置批量处理参数
6. 执行工作流，同时获得多张编辑后的图像

### 🎨 专业图像生成工作流

**高质量图像生成:**
1. 添加 "Gemini Banana Image Generator" 节点
2. 输入详细提示词："一只优雅的白猫坐在古典书房里，温暖的壁炉光线，油画风格，专业摄影构图"
3. 设置参数：
   - 尺寸: "1024x1024"
   - 质量: "Masterpiece Level"
   - 风格: "Classical Oil Painting"
   - 细节等级: "Premium Quality"
   - 相机控制: "Professional Portrait"
   - 灯光控制: "Soft Glow"
4. 连接到图像预览节点
5. 执行工作流

**快速图像生成 (Nano-Banana):**
1. 添加 "Comfly Nano Banana" 节点
2. 输入提示词："现代简约风格的客厅设计"
3. 设置尺寸为 "1024x1024"
4. 设置种子值 (可选，用于重现结果)
5. 选择合适的API端点
6. 执行工作流，快速获得结果

### 🌐 镜像API使用工作流

**突破网络限制:**
1. 添加 "Gemini Banana Mirror" 节点
2. 配置镜像站点地址
3. 输入处理指令
4. 设置代理参数 (如需要)
5. 执行工作流，享受稳定的服务

### 🔍 智能图像分析工作流

1. 使用 "Load Image" 节点加载图像
2. 连接到 "Gemini Vision Analyzer" 节点
3. 输入分析提示："详细分析这张图片的构图、色彩、情感表达和艺术价值"
4. 连接到文本输出节点
5. 执行工作流，获得专业的图像分析

### 🌍 多语言翻译工作流

1. 添加 "Gemini Banana Translation" 节点
2. 输入要翻译的文本
3. 设置源语言和目标语言
4. 选择翻译质量等级
5. 连接到文本输出节点
6. 执行工作流

### 📈 批量处理优化工作流

**大批量图像处理:**
1. 准备多张需要处理的图像
2. 使用 "Gemini Banana" 节点的批量模式
3. 设置统一的处理参数
4. 配置输出格式和质量
5. 执行批量处理，提高工作效率

## ⚙️ 高级配置

### 🆕 安全设置与系统指令配置 (v2.1)

**安全设置预设:**
```json
{
  "safety_presets": {
    "default": "使用Google默认安全设置",
    "strict": "严格模式 - 阻止大多数不安全内容",
    "moderate": "中等模式 - 阻止中等及以上不安全内容",
    "permissive": "宽松模式 - 仅阻止高度不安全内容",
    "off": "关闭模式 - 不进行安全过滤"
  }
}
```

**系统指令预设模板:**
```json
{
  "system_instruction_presets": {
    "none": "无系统指令",
    "image_generation": "专业图像生成助手",
    "image_editing": "专业图像编辑助手",
    "creative_artist": "创意艺术家",
    "technical_expert": "技术专家",
    "friendly_helper": "友好助手",
    "professional_photographer": "专业摄影师"
  }
}
```

**自定义系统指令示例:**
```json
{
  "custom_system_instruction": "你是一位专业的产品摄影师，擅长电商产品拍摄。请确保产品细节清晰，背景简洁，光线均匀。"
}
```

### Banana节点专用配置

**官方API节点配置:**
```json
{
  "gemini_api_base": "https://generativelanguage.googleapis.com",
  "glm_api_base": "https://open.bigmodel.cn/api/paas/v4/",
  "openrouter_api_base": "https://openrouter.ai/api/v1",
  "image_editing_settings": {
    "max_batch_size": 5,
    "quality_enhancement": true,
    "smart_resize": true
  },
  "safety_settings": {
    "default_level": "moderate",
    "custom_categories": {
      "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
      "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
      "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
      "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
    }
  },
  "system_instruction": {
    "default_preset": "image_generation",
    "allow_custom": true
  }
}
```

**镜像API节点配置:**
```json
{
  "mirror_endpoints": {
    "gemini_mirror": "https://your-gemini-mirror.com",
    "comfly_endpoint": "https://api.comfly.ai",
    "t8_mirror": "https://ai.t8star.cn"
  },
  "fallback_settings": {
    "auto_fallback": true,
    "retry_attempts": 3,
    "timeout_seconds": 60
  },
  "proxy_settings": {
    "http": "http://proxy:port",
    "https": "https://proxy:port"
  }
}
```

**图像编辑优化配置:**
```json
{
  "image_processing": {
    "max_resolution": "2048x2048",
    "compression_quality": 95,
    "format_preference": "PNG",
    "batch_processing": {
      "enabled": true,
      "max_concurrent": 3,
      "memory_limit": "4GB"
    }
  }
}
```

### 环境变量配置

支持通过环境变量配置API密钥：

```bash
export GEMINI_API_KEY="your_gemini_api_key"
export ZHIPUAI_API_KEY="your_glm_api_key"
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

### Banana节点性能优化

**图像编辑性能优化:**
- 🚀 **智能批量处理**: 同时处理多达10张图像，效率提升10倍
- 🚀 **内存优化**: 智能内存管理，支持大尺寸图像编辑
- 🚀 **并发处理**: 多线程异步处理，充分利用系统资源
- 🚀 **缓存机制**: 智能结果缓存，相似请求秒级响应

**API调用优化:**
- 🌐 **双重保障**: 官方API + 镜像API，确保服务可用性
- 🌐 **自动切换**: 检测到官方API不可用时自动切换到镜像
- 🌐 **负载均衡**: 多个镜像节点分担负载，提高稳定性
- 🌐 **智能重试**: 失败自动重试，指数退避算法优化

**质量与速度平衡:**
- ⚡ **快速模式**: Nano-Banana节点，秒级图像生成
- 🎨 **质量模式**: Gemini节点，专业级图像编辑
- 🔄 **自适应**: 根据任务复杂度自动选择最优处理方式
- 📊 **实时监控**: 处理进度实时反馈，透明化处理过程

## 🐛 故障排除

### 常见问题

#### 1. API密钥错误
- **问题**: "Invalid API key" 或 "Authentication failed"
- **解决**: 检查API密钥是否正确配置，确认密钥有效且有足够额度

#### 2. 网络连接问题
- **问题**: "Connection timeout" 或 "Network error"
- **解决**: 检查网络连接，配置正确的代理设置

#### 3. 模型不可用
- **问题**: "Model not found" 或 "Model not supported"
- **解决**: 确认选择的模型在当前API中可用，检查模型名称拼写

#### 4. 内存不足
- **问题**: "Out of memory" 或系统卡顿
- **解决**: 降低批量大小、减少图像分辨率、关闭不必要的应用

#### 5. 图像生成失败
- **问题**: 生成黑色或异常图像
- **解决**: 检查提示词是否合适，尝试不同的模型或参数

### 调试模式

**v2.1 优化说明:**
- ✅ 启动输出已精简 68%，减少控制台噪音
- ✅ 移除了不存在模块的错误提示
- ✅ 保留了所有必要的错误日志

**如需启用详细调试信息:**

1. 找到节点文件中的日志函数 `_log_info()`, `_log_warning()`
2. 将函数内的 `pass` 改为相应的 `print` 语句
3. 重启ComfyUI查看详细日志

**调试输出控制:**
```python
# 在 gemini_banana.py 中
def _log_info(message):
    # pass  # 默认关闭
    print(f"[INFO] {message}")  # 启用调试

def _log_warning(message):
    # pass  # 默认关闭
    print(f"[WARNING] {message}")  # 启用调试
```

### 免费模型推荐

为了节省成本，推荐使用以下免费或低成本模型：

- **GLM-4-Flash**: 智谱AI的免费模型，适合文本生成
- **GLM-4V-Flash**: 支持图像理解的免费模型
- **Gemini-1.5-Flash**: Google的快速模型，有免费额度

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交问题报告和功能请求！如果您想贡献代码，请：

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## � Support the Project

### ☕ Buy Me a Coffee

如果您觉得 ComfyUI LLM Banana 对您有帮助，让您的 AI 图像生成项目更加轻松，请考虑支持开发！

#### 💜 Your support helps:

- 🚀 **Accelerate new feature development** - 加速新功能开发
- 🧠 **Enhance AI capabilities** - 增强 AI 能力
- 🔧 **Improve system stability** - 提升系统稳定性
- 📚 **Create better documentation** - 创建更好的文档
- 🌍 **Support the open source community** - 支持开源社区

---

<table>
<tr>
<td align="center" width="50%">

### 💬 WeChat Contact

<img src="https://github.com/xuchenxu168/images/blob/main/%E5%BE%AE%E4%BF%A1%E5%8F%B7.jpg" alt="WeChat QR Code" width="200px" />

*Scan to add WeChat 扫码添加微信*

**WeChat ID**: Kenchen7168

</td>
<td align="center" width="50%">

### ☕ Support Development

<img src="https://github.com/xuchenxu168/images/blob/main/%E6%94%B6%E6%AC%BE%E7%A0%81.jpg" alt="Buy Me a Coffee" width="200px" />

*Scan to buy me a coffee 扫码请我喝咖啡*

#### ☕ Every coffee counts! 每一杯咖啡都是支持！

</td>
</tr>
</table>

---

### 🙏 Thank you for your support!

*Your contributions, whether through code, feedback, or coffee, make ComfyUI LLM Banana better for everyone!*

**谢谢您的支持！无论是代码贡献、反馈建议还是请我喝咖啡，都让 ComfyUI LLM Banana 变得更好！**

---

## 📞 Community & Support

如果您遇到问题或需要帮助：

- 💬 **WeChat**: Kenchen7168 (扫描上方二维码添加)
- 🐛 **Issues**: [GitHub Issue](https://github.com/your-repo/ComfyUI_LLM_Banana/issues)
- 📖 **Documentation**: 查看项目文档和示例
- 🌟 **Star**: 如果觉得有用，请给项目一个星标！

## 🔄 更新日志

### v2.1.0 (2025-10-12) - 质量与体验全面提升 🎉
**🛡️ 安全设置与系统指令 (Safety Settings & System Instruction):**
- ✅ **安全设置支持**: 新增 5 个安全级别预设（default, strict, moderate, permissive, off）
- ✅ **系统指令功能**: 7 个专业预设模板（图像生成助手、创意艺术家、专业摄影师等）
- ✅ **自定义指令**: 支持用户自定义系统指令，优先级高于预设
- ✅ **完全兼容**: 符合 Google Gemini 官方 API 标准
- ✅ **全节点覆盖**: 所有 7 个 Gemini Banana 节点均已支持

**📐 Response Modalities 标准化:**
- ✅ **格式统一**: 统一使用 `["Text", "Image"]` 格式，符合官方标准
- ✅ **智能转换**: 自动规范化不同格式的输入
- ✅ **向后兼容**: 完全兼容旧版工作流
- ✅ **代码优化**: 创建 `normalize_response_modalities()` 辅助函数

**🧹 启动体验优化:**
- ✅ **输出精简**: 启动输出减少 68%（从 22 行减少到 7 行）
- ✅ **移除冗余**: 清理 19 个无用的调试输出
- ✅ **错误清理**: 移除不存在模块的错误提示
- ✅ **代码精简**: 删除 53 行无用代码
- ✅ **加载更快**: 优化模块加载逻辑，启动更流畅

**📚 文档完善:**
- ✅ **详细指南**: 新增安全设置和系统指令使用指南
- ✅ **快速参考**: 创建快速参考文档，方便查阅
- ✅ **实现总结**: 完整的技术实现文档
- ✅ **测试覆盖**: 100% 测试通过，功能稳定可靠

**🔧 技术改进:**
- ✅ **代码质量**: 移除重复代码，提升可维护性
- ✅ **性能优化**: 优化 AI 模型检测逻辑
- ✅ **错误处理**: 改进异常处理机制
- ✅ **模块清理**: 移除已废弃的模块引用

### v1.0.0 - Banana系列节点重大更新
**🍌 核心功能突破:**
- ✅ **图像到图像编辑**: 业界领先的图像编辑能力
- ✅ **多图批量处理**: 同时编辑多张图像，效率提升10倍
- ✅ **双API架构**: 官方API + 镜像API，确保服务稳定性
- ✅ **6大核心节点**: 3个官方API节点 + 3个镜像API节点

**🚀 技术创新:**
- ✅ **智能图像理解**: 深度理解图像内容，精确执行编辑指令
- ✅ **内容感知编辑**: 保持图像结构，智能修改指定区域
- ✅ **批量一致性**: 多图编辑保持风格和质量一致性
- ✅ **专业级控制**: 丰富的摄影和艺术参数控制

**🌐 服务优化:**
- ✅ **网络突破**: 镜像节点解决网络访问限制
- ✅ **自动切换**: 官方API不可用时自动切换镜像
- ✅ **负载均衡**: 多节点分担负载，提高稳定性
- ✅ **性能优化**: 内存管理、并发处理、智能缓存

**🛠️ 用户体验:**
- ✅ **简化操作**: 一键式图像编辑，降低使用门槛
- ✅ **实时反馈**: 处理进度实时显示
- ✅ **错误处理**: 智能错误恢复和提示
- ✅ **文档完善**: 详细的使用指南和示例

### 历史更新
- **6-28**: 新增翻译节点，支持50+语言翻译
- **6-20**: 集成FLUX提示词模板，优化提示词生成
- **持续优化**: 性能提升、稳定性改进、功能扩展

## 🙏 致谢

感谢以下项目和开发者的贡献：

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的AI工作流平台
- [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party) - FLUX提示词模板参考
- [ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet) - 翻译节点参考

---

**⚠️ 重要提示**:
- 使用本项目需要相应的API密钥和网络连接
- 请确保遵守各AI服务提供商的使用条款和限制
- 建议在生产环境中使用前进行充分测试

**💡 提示**: 如果您觉得这个项目有用，请给我们一个⭐星标支持！

<img src="/image/微信图片_20250620124237.png" alt="ComfyUI LLM Banana" width="200px" />

