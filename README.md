# ComfyUI LLM Banana - 多模态AI助手节点集合

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

一个功能强大的ComfyUI自定义节点集合，集成了多种大语言模型和AI服务，支持文本生成、图像生成、图像分析、翻译等多种AI功能。

## 🌟 主要特性

- **多模型支持**: Gemini、GLM、OpenRouter、Nano-Banana等多种AI模型
- **图像生成**: 支持文本到图像生成，多种尺寸和质量选项
- **图像分析**: 智能图像描述、内容分析、视觉问答
- **多语言翻译**: 支持多种翻译服务和语言对
- **图像增强**: AI超分辨率、智能裁剪、质量优化
- **灵活配置**: 支持多种API端点和代理设置

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

### 🤖 文本生成节点

#### Gemini Banana (文本生成)
- **功能**: 使用Gemini模型进行文本生成和对话
- **输入**:
  - `prompt`: 文本提示词
  - `api_key`: Gemini API密钥
  - `model`: 模型选择 (gemini-pro, gemini-1.5-pro等)
  - `temperature`: 创造性控制 (0.0-2.0)
  - `max_output_tokens`: 最大输出长度
- **输出**: 生成的文本内容
- **特性**: 支持多种Gemini模型，可配置生成参数

#### GLM Banana (智谱AI)
- **功能**: 使用GLM模型进行中文文本生成
- **输入**:
  - `prompt`: 文本提示词
  - `api_key`: 智谱AI API密钥
  - `model`: GLM模型版本 (GLM-4, GLM-4-Flash等)
  - `temperature`: 温度参数
- **输出**: 生成的文本内容
- **特性**: 专门优化中文处理，支持GLM-4等模型

#### OpenRouter Banana
- **功能**: 通过OpenRouter访问多种AI模型
- **输入**: 提示词、模型选择、API密钥
- **输出**: 生成的文本内容
- **特性**: 支持Claude、GPT、Llama等多种模型

### 🎨 图像生成节点

#### Gemini Banana Image Generator
- **功能**: 文本到图像生成
- **输入**:
  - `prompt`: 图像描述提示词
  - `api_key`: API密钥
  - `size`: 图像尺寸 (512x512, 1024x1024, 1024x1792等)
  - `quality`: 质量设置 (标准、高清、超高清)
  - `style`: 风格控制 (写实、艺术、卡通等)
  - `detail_level`: 细节等级
  - `camera_control`: 相机控制
  - `lighting_control`: 灯光控制
- **输出**: 生成的图像tensor
- **特性**:
  - 支持多种图像尺寸
  - 智能提示词增强
  - 专业摄影参数控制

#### Nano-Banana Image Generator
- **功能**: 使用Nano-Banana模型生成图像
- **输入**:
  - `prompt`: 提示词
  - `size`: 尺寸选择
  - `seed`: 种子值控制
  - `api_url`: API端点
- **输出**: 高质量生成图像
- **特性**: 快速生成，支持种子控制，兼容多个镜像站点

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

### 基础文本生成工作流

1. 添加 "Gemini Banana" 节点
2. 输入提示词："写一首关于春天的诗"
3. 设置模型为 "gemini-pro"
4. 配置API密钥
5. 连接到文本输出节点
6. 执行工作流

### 图像生成工作流

1. 添加 "Gemini Banana Image Generator" 节点
2. 输入提示词："一只可爱的小猫在花园里玩耍，阳光明媚，花朵盛开"
3. 设置尺寸为 "1024x1024"
4. 选择质量为 "Premium Quality"
5. 设置细节等级为 "Professional Detail"
6. 连接到图像预览节点
7. 执行工作流

### 图像分析工作流

1. 使用 "Load Image" 节点加载图像
2. 连接到 "Gemini Vision Analyzer" 节点
3. 输入分析提示："详细描述这张图片的内容，包括场景、人物、物体和氛围"
4. 连接到文本输出节点
5. 执行工作流

### 翻译工作流

1. 添加 "Gemini Banana Translation" 节点
2. 输入要翻译的文本
3. 设置源语言为 "中文"，目标语言为 "英文"
4. 连接到文本输出节点
5. 执行工作流

### 图像增强工作流

1. 加载低分辨率图像
2. 连接到 "AI Image Upscaler" 节点
3. 选择放大算法 (推荐 Real-ESRGAN)
4. 设置放大倍数 (2x 或 4x)
5. 连接到图像预览节点
6. 执行工作流

## ⚙️ 高级配置

### 自定义API端点

可以在配置文件中设置自定义API端点：

```json
{
  "gemini_api_base": "https://your-custom-endpoint.com",
  "glm_api_base": "https://open.bigmodel.cn/api/paas/v4/",
  "proxy_settings": {
    "http": "http://proxy:port",
    "https": "https://proxy:port"
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

### 性能优化

- **批量处理**: 支持多图像批量处理
- **缓存机制**: 智能结果缓存，提高响应速度
- **异步处理**: 支持异步API调用，提高并发性能
- **内存管理**: 自动释放不需要的图像数据

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

如需启用调试信息进行问题诊断：

1. 找到节点文件中的日志函数
2. 将 `pass` 改为相应的 `print` 语句
3. 重启ComfyUI查看详细日志

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

## 📞 支持

如果您遇到问题或需要帮助：

- 提交 [GitHub Issue](https://github.com/your-repo/ComfyUI_LLM_Banana/issues)
- 查看项目文档和示例
- 加入社区讨论

## 🔄 更新日志

### v1.0.0 (最新)
- ✅ 初始版本发布
- ✅ 支持Gemini、GLM、OpenRouter多种模型
- ✅ 图像生成和分析功能
- ✅ 多语言翻译支持
- ✅ AI图像增强功能
- ✅ 优化调试信息输出
- ✅ 简化图像处理流程

### 历史更新
- 6-28: 新增翻译节点，支持中英文翻译
- 6-20: 集成FLUX提示词模板功能
- 持续优化性能和稳定性

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
