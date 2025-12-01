# Banana2 提示词模板节点 - 快速开始

## [object Object]分钟快速上手

### 步骤 1：配置 API 密钥

在 `ComfyUI-Gemini-3/config.json` 中添加你的 Gemini API 密钥：

```json
{
  "multimodal_api_key": "your-gemini-api-key-here"
}
```

### 步骤 2：在 ComfyUI 中添加节点

1. 在 ComfyUI 中右键点击空白处
2. 选择 `Add Node` → `🍌 Banana` → `Banana2-提示词模板`
3. 节点将出现在画布上

### 通用 API 参数（新增）

在 Banana2-提示词模板节点的 `api_key` 与 `max_output_tokens` 之间新增了三个通用 API 参数：

- `base_url`：自定义/镜像基础地址（留空走配置/默认）；
- `version`：`Auto`、`v1`、`v1alpha`、`v1beta`；
- `auth_mode`：`auto`（按域自动）、`google_xgoog`、`bearer`。

说明：`Gemini-Multimodal` 节点也同步支持以上三参数，端点与认证逻辑保持一致。

### 步骤 3：基础使用 - 文本生成图片

**场景**：从文本描述生成图片

```
1. 添加 Banana2-提示词模板 节点
2. 设置参数：
   - 模板类型: "文本到图片基础模板"
   - 用户提示词: "一只在月光下的神秘黑猫"
   - Temperature: 0.7
3. 连接到 Gemini Banana 图像生成节点
4. 运行工作流

（可选）若使用镜像/代理：设置 `base_url`、选择 `auth_mode=bearer`；`version=Auto` 下非 Google 域默认使用 `v1`。
```

**预期输出**：
```
Create a mystical scene featuring a sleek black cat sitting elegantly under moonlight. 
The cat has piercing emerald green eyes that glow softly in the darkness. The setting 
is a quiet garden at night with silvery moonbeams filtering through tree branches, 
creating dramatic shadows. The cat's fur has a subtle blue sheen from the moonlight. 
Composition: medium shot, slightly low angle to emphasize the cat's regal posture. 
Style: cinematic photography with high contrast, f/2.8, cool color temperature.
```

### 步骤 4：进阶使用 - 图片编辑

**场景**：给现有图片添加元素

```
1. 添加 Load Image 节点，加载一张猫咪照片
2. 添加 Banana2-提示词模板 节点
3. 设置参数：
   - 模板类型: "图片编辑模板"
   - 用户提示词: "给猫咪戴上一顶圣诞帽"
   - Temperature: 0.6
4. 连接：Load Image → Banana2 (image输入)
5. 连接：Banana2 → Gemini Banana 图像生成
6. 运行工作流
```

### 步骤 5：高级使用 - 多图合成

**场景**：组合多张图片的元素

```
1. 添加两个 Load Image 节点
   - 图片1: 一件连衣裙
   - 图片2: 一位模特
2. 添加 Banana2-提示词模板 节点
3. 设置参数：
   - 模板类型: "多图合成模板"
   - 用户提示词: "让模特穿上这件连衣裙，拍摄专业电商照片"
   - Temperature: 0.5
4. 连接：
   - Load Image 1 → Banana2 (image)
   - Load Image 2 → Banana2 (image_2)
   - Banana2 → Gemini Banana 图像生成
5. 运行工作流

（可选）若使用官方域：`base_url` 留空，填写 `api_key`，`auth_mode=auto`；`version=Auto` 将根据任务选择 `v1beta/v1alpha`。
```

## 📊 完整工作流示例

### 示例 1：AI 头像生成器

```
[用户输入文本]
    ↓
[Banana2-提示词模板]
(模板: 文本到图片基础模板)
    ↓
[Gemini Banana 图像生成]
    ↓
[图像增强处理]
    ↓
[保存图片]
```

### 示例 2：产品照片编辑器

```
[加载产品照片] → [Banana2-提示词模板] → [Gemini Banana 图像生成]
                  (模板: 图片编辑模板)              ↓
                                              [对比原图]
                                                    ↓
                                              [保存最佳结果]
```

### 示例 3：时尚电商合成器

```
[服装图片] ──┐
            ├→ [Banana2-提示词模板] → [Gemini Banana] → [保存]
[模特图片] ──┘   (模板: 多图合成)
```

## 💡 实用技巧

### 技巧 1：提高提示词质量

**问题**：生成的提示词太简单
**解决**：
```
在用户提示词中添加更多细节：
❌ "一只猫"
✅ "一只橙色虎斑猫，有着明亮的绿色眼睛，坐在木质窗台上，阳光从左侧照射进来"
```

### 技巧 2：控制创意程度

**需要精确控制**：Temperature = 0.3-0.5
```
适用场景：
- 产品照片编辑
- 品牌元素添加
- 精确的颜色匹配
```

**需要创意发挥**：Temperature = 0.7-0.9
```
适用场景：
- 艺术创作
- 概念设计
- 风格探索
```

### 技巧 3：多轮迭代优化

```
第一轮：使用基础提示词生成
    ↓
查看结果，识别需要改进的地方
    ↓
第二轮：添加具体的修改指令
    ↓
继续迭代直到满意
```

### 技巧 4：模板组合使用

```
步骤1：使用"文本到图片"生成基础图像
步骤2：使用"图片编辑"添加细节
步骤3：使用"高保真细节保留"精修关键部分
```

## 🎯 常见使用场景

### 场景 1：社交媒体内容创作
```
模板：文本到图片基础模板
提示词：创建吸引眼球的社交媒体配图
Temperature：0.7-0.8
```

### 场景 2：电商产品图优化
```
模板：图片编辑模板
提示词：改善产品背景、光照和构图
Temperature：0.4-0.6
```

### 场景 3：概念设计可视化
```
模板：草图优化模板
提示词：将手绘草图转化为专业渲染图
Temperature：0.6-0.8
```

### 场景 4：角色设计
```
模板：角色一致性模板
提示词：生成角色的多个视角和姿势
Temperature：0.5-0.7
```

### 场景 5：品牌营销素材
```
模板：高保真细节保留模板
提示词：在产品图上添加品牌logo和文字
Temperature：0.3-0.5
```

## 🔍 故障排除速查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 提示词太简单 | Temperature太低 | 提高到0.7-0.8 |
| 提示词不相关 | 用户输入不够具体 | 添加更多细节描述 |
| API错误 | 密钥未配置 | 检查config.json |
| 图片无法识别 | 格式不支持 | 使用PNG/JPEG格式 |
| 输出被截断 | Token限制 | 增加max_output_tokens |

## 📚 下一步学习

1. [完整使用指南](banana2_prompt_template_guide.md)
2. 🎨 探索[提示词模板详解](banana2_template_details.md)
3. 💼 查看[实战案例集](banana2_use_cases.md)
4. 🔧 了解[高级配置选项](banana2_advanced_config.md)

## 🆘 获取帮助

- **文档**：查看 `docs/` 目录下的详细文档
- **示例**：参考 `examples/` 目录下的工作流示例
- **问题反馈**：在 GitHub Issues 中提交问题

## 🎉 开始创作！

现在你已经掌握了 Banana2 提示词模板节点的基础使用，开始创作你的第一个 AI 图像吧！

记住：
- ✅ 提示词越具体，效果越好
- ✅ 多尝试不同的模板和参数
- ✅ 迭代优化是关键
- ✅ 享受创作的乐趣！

Happy Creating! 🚀🎨✨

