#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 通用智能主体检测（人物、物品、动物等）
适用于任何类型的主体检测
"""

def detect_image_foreground_subject(image):
    """
    🎯 通用智能主体检测（人物、物品、动物等）
    返回主体边界框 (x, y, width, height) 和中心点
    """
    try:
        import cv2
        import numpy as np
        
        # 转换为OpenCV格式
        if hasattr(image, "convert"):
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_array = np.array(image)
        else:
            img_array = image
        
        # 转换为BGR格式（OpenCV使用BGR）
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        height, width = img_bgr.shape[:2]
        print(f"🔍 图像尺寸: {width}x{height}")
        
        # 🎯 方法1：多区域GrabCut检测（通用主体检测）
        try:
            print(f"🔍 开始多区域GrabCut检测...")
            
            # 将图像分为9个区域进行检测
            regions = [
                # 左中右三列
                (0, 0, width//3, height),                    # 左列
                (width//3, 0, width//3, height),             # 中列
                (2*width//3, 0, width//3, height),           # 右列
                # 上中下三行
                (0, 0, width, height//3),                    # 上行
                (0, height//3, width, height//3),            # 中行
                (0, 2*height//3, width, height//3),          # 下行
                # 四个角落
                (0, 0, width//2, height//2),                 # 左上
                (width//2, 0, width//2, height//2),          # 右上
                (0, height//2, width//2, height//2),         # 左下
                (width//2, height//2, width//2, height//2),  # 右下
            ]
            
            best_contour = None
            best_area = 0
            best_region = None
            
            for i, (x, y, w, h) in enumerate(regions):
                try:
                    # 创建掩码
                    mask = np.zeros(img_bgr.shape[:2], np.uint8)
                    
                    # 创建前景矩形
                    rect = (x, y, w, h)
                    
                    # 创建临时数组
                    bgdModel = np.zeros((1,65), np.float64)
                    fgdModel = np.zeros((1,65), np.float64)
                    
                    # 应用GrabCut算法
                    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
                    
                    # 创建掩码
                    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
                    
                    # 找到前景轮廓
                    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # 找到最大的轮廓
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        
                        # 记录最大的轮廓
                        if area > best_area:
                            best_area = area
                            best_contour = largest_contour
                            best_region = i
                            
                except Exception as e:
                    continue
            
            if best_contour is not None and best_area > (width * height) * 0.01:
                x, y, w, h = cv2.boundingRect(best_contour)
                subject_center_x = x + w // 2
                subject_center_y = y + h // 2

                # 🚀 关键修复：检查检测到的主体是否过大
                detected_area = w * h
                image_area = width * height
                detected_ratio = detected_area / image_area

                print(f"🎯 多区域GrabCut检测到主体: 区域{best_region}, 位置({x}, {y}), 尺寸({w}x{h}), 中心({subject_center_x}, {subject_center_y})")
                print(f"🔍 检测主体占比: {detected_ratio:.3f}")

                # 如果检测到的主体过大（超过35%），调整主体尺寸
                if detected_ratio > 0.35:
                    print(f"⚠️ 检测到的主体过大(占比{detected_ratio:.1%})，调整主体尺寸...")

                    # 重新计算合理的主体尺寸
                    if width > height:
                        new_w = int(width * 0.2)
                        new_h = int(height * 0.3)
                    else:
                        new_w = int(width * 0.3)
                        new_h = int(height * 0.2)

                    # 确保主体不会太小
                    new_w = max(new_w, width // 10)
                    new_h = max(new_h, height // 10)

                    # 重新计算主体位置，保持中心不变
                    new_x = subject_center_x - new_w // 2
                    new_y = subject_center_y - new_h // 2

                    # 确保主体在图像范围内
                    new_x = max(0, min(new_x, width - new_w))
                    new_y = max(0, min(new_y, height - new_h))

                    print(f"🔧 调整后主体: ({new_x}, {new_y}, {new_w}x{new_h}), 新占比{(new_w*new_h)/image_area:.3f}")
                    return (new_x, new_y, new_w, new_h), (subject_center_x, subject_center_y)

                return (x, y, w, h), (subject_center_x, subject_center_y)
        
        except Exception as e:
            print(f"⚠️ 多区域GrabCut检测失败: {e}")
        
        # 🎯 方法2：基于边缘密度的主体检测
        try:
            print(f"🔍 开始基于边缘密度的主体检测...")
            
            # 转换为灰度图
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 计算边缘密度
            kernel_size = 20
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
            
            # 找到边缘密度最高的区域
            max_density = np.max(edge_density)
            if max_density > 0:
                threshold = max_density * 0.3
                mask = edge_density > threshold
                
                # 形态学操作
                kernel = np.ones((10,10), np.uint8)
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                
                # 找到轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    subject_center_x = x + w // 2
                    subject_center_y = y + h // 2

                    # 🚀 关键修复：检查检测到的主体是否过大
                    detected_area = w * h
                    image_area = width * height
                    detected_ratio = detected_area / image_area

                    print(f"🎯 边缘密度检测到主体: 位置({x}, {y}), 尺寸({w}x{h}), 中心({subject_center_x}, {subject_center_y})")
                    print(f"🔍 检测主体占比: {detected_ratio:.3f}")

                    # 如果检测到的主体过大（超过35%），调整主体尺寸
                    if detected_ratio > 0.35:
                        print(f"⚠️ 检测到的主体过大(占比{detected_ratio:.1%})，调整主体尺寸...")

                        # 重新计算合理的主体尺寸
                        if width > height:
                            new_w = int(width * 0.2)
                            new_h = int(height * 0.3)
                        else:
                            new_w = int(width * 0.3)
                            new_h = int(height * 0.2)

                        # 确保主体不会太小
                        new_w = max(new_w, width // 10)
                        new_h = max(new_h, height // 10)

                        # 重新计算主体位置，保持中心不变
                        new_x = subject_center_x - new_w // 2
                        new_y = subject_center_y - new_h // 2

                        # 确保主体在图像范围内
                        new_x = max(0, min(new_x, width - new_w))
                        new_y = max(0, min(new_y, height - new_h))

                        print(f"🔧 调整后主体: ({new_x}, {new_y}, {new_w}x{new_h}), 新占比{(new_w*new_h)/image_area:.3f}")
                        return (new_x, new_y, new_w, new_h), (subject_center_x, subject_center_y)

                    return (x, y, w, h), (subject_center_x, subject_center_y)
        
        except Exception as e:
            print(f"⚠️ 基于边缘密度的主体检测失败: {e}")
        
        # 🎯 方法3：智能网格分析
        try:
            print(f"🔍 开始智能网格分析...")
            
            # 将图像分为网格，分析每个网格的特征
            grid_size = 32
            rows = height // grid_size
            cols = width // grid_size
            
            # 计算每个网格的特征
            grid_scores = np.zeros((rows, cols))
            
            for i in range(rows):
                for j in range(cols):
                    y1, y2 = i * grid_size, (i + 1) * grid_size
                    x1, x2 = j * grid_size, (j + 1) * grid_size
                    
                    if y2 <= height and x2 <= width:
                        grid = img_bgr[y1:y2, x1:x2]
                        
                        # 计算网格的方差（纹理复杂度）
                        gray_grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
                        variance = np.var(gray_grid)
                        
                        # 计算颜色丰富度
                        hsv_grid = cv2.cvtColor(grid, cv2.COLOR_BGR2HSV)
                        color_richness = np.std(hsv_grid[:,:,0]) + np.std(hsv_grid[:,:,1]) + np.std(hsv_grid[:,:,2])
                        
                        # 综合评分
                        grid_scores[i, j] = variance * 0.7 + color_richness * 0.3
            
            # 找到得分最高的网格
            max_score = np.max(grid_scores)
            if max_score > 0:
                # 找到得分阈值以上的区域
                threshold = max_score * 0.5
                mask = grid_scores > threshold
                
                # 找到轮廓
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # 调整到原图坐标
                    x *= grid_size
                    y *= grid_size
                    w *= grid_size
                    h *= grid_size
                    
                    subject_center_x = x + w // 2
                    subject_center_y = y + h // 2

                    # 🚀 关键修复：检查检测到的主体是否过大
                    detected_area = w * h
                    image_area = width * height
                    detected_ratio = detected_area / image_area

                    print(f"🎯 智能网格分析检测到主体: 位置({x}, {y}), 尺寸({w}x{h}), 中心({subject_center_x}, {subject_center_y})")
                    print(f"🔍 检测主体占比: {detected_ratio:.3f}")

                    # 如果检测到的主体过大（超过35%），调整主体尺寸
                    if detected_ratio > 0.35:
                        print(f"⚠️ 检测到的主体过大(占比{detected_ratio:.1%})，调整主体尺寸...")

                        # 重新计算合理的主体尺寸
                        if width > height:
                            new_w = int(width * 0.2)
                            new_h = int(height * 0.3)
                        else:
                            new_w = int(width * 0.3)
                            new_h = int(height * 0.2)

                        # 确保主体不会太小
                        new_w = max(new_w, width // 10)
                        new_h = max(new_h, height // 10)

                        # 重新计算主体位置，保持中心不变
                        new_x = subject_center_x - new_w // 2
                        new_y = subject_center_y - new_h // 2

                        # 确保主体在图像范围内
                        new_x = max(0, min(new_x, width - new_w))
                        new_y = max(0, min(new_y, height - new_h))

                        print(f"🔧 调整后主体: ({new_x}, {new_y}, {new_w}x{new_h}), 新占比{(new_w*new_h)/image_area:.3f}")
                        return (new_x, new_y, new_w, new_h), (subject_center_x, subject_center_y)

                    return (x, y, w, h), (subject_center_x, subject_center_y)
        
        except Exception as e:
            print(f"⚠️ 智能网格分析失败: {e}")
        
        # 🎯 备用策略：基于图像中心的主体位置估计
        try:
            print(f"🔍 使用备用策略：基于图像中心的主体位置估计...")

            # 分析图像中心区域的特征
            center_x = width // 2
            center_y = height // 2

            # 🚀 修复：使用极保守的主体尺寸估计，确保主体完整显示
            # 根据图像尺寸动态调整主体大小，确保主体不会占满整个画面
            if width > height:
                # 横向图像：主体宽度为图像宽度的20%，高度为图像高度的30%
                estimated_w = int(width * 0.2)
                estimated_h = int(height * 0.3)
            else:
                # 纵向图像：主体宽度为图像宽度的30%，高度为图像高度的20%
                estimated_w = int(width * 0.3)
                estimated_h = int(height * 0.2)

            # 确保主体尺寸不会太小
            estimated_w = max(estimated_w, width // 10)
            estimated_h = max(estimated_h, height // 10)

            # 确保主体尺寸不会太大（不超过图像的35%）
            estimated_w = min(estimated_w, int(width * 0.35))
            estimated_h = min(estimated_h, int(height * 0.35))

            estimated_center_x = center_x
            estimated_center_y = center_y

            print(f"🎯 备用策略估计主体位置: 中心({estimated_center_x}, {estimated_center_y}), 尺寸({estimated_w}x{estimated_h})")
            print(f"🔍 主体占比: 宽度{estimated_w/width:.1%}, 高度{estimated_h/height:.1%}")
            return (center_x - estimated_w//2, center_y - estimated_h//2, estimated_w, estimated_h), (estimated_center_x, estimated_center_y)
        
        except Exception as e:
            print(f"⚠️ 备用策略失败: {e}")
        
        # 如果所有方法都失败，返回图像中心作为默认主体位置
        default_center_x = width // 2
        default_center_y = height // 2

        # 🚀 修复：使用极保守的默认主体尺寸
        if width > height:
            default_w = int(width * 0.2)
            default_h = int(height * 0.3)
        else:
            default_w = int(width * 0.3)
            default_h = int(height * 0.2)

        default_w = max(default_w, width // 10)
        default_h = max(default_h, height // 10)
        default_w = min(default_w, int(width * 0.35))
        default_h = min(default_h, int(height * 0.35))

        default_x = default_center_x - default_w // 2
        default_y = default_center_y - default_h // 2

        print(f"⚠️ 所有检测方法失败，使用图像中心作为默认主体位置: ({default_center_x}, {default_center_y})")
        print(f"🔍 默认主体尺寸: {default_w}x{default_h}, 占比: 宽度{default_w/width:.1%}, 高度{default_h/height:.1%}")
        return (default_x, default_y, default_w, default_h), (default_center_x, default_center_y)
        
    except ImportError:
        print("⚠️ OpenCV未安装，无法进行智能主体检测")
        # 返回图像中心作为默认主体位置
        width, height = image.size
        default_center_x = width // 2
        default_center_y = height // 2

        # 🚀 修复：使用极保守的默认主体尺寸
        if width > height:
            default_w = int(width * 0.2)
            default_h = int(height * 0.3)
        else:
            default_w = int(width * 0.3)
            default_h = int(height * 0.2)

        default_w = max(default_w, width // 10)
        default_h = max(default_h, height // 10)
        default_w = min(default_w, int(width * 0.35))
        default_h = min(default_h, int(height * 0.35))

        default_x = default_center_x - default_w // 2
        default_y = default_center_y - default_h // 2

        print(f"🔍 OpenCV未安装，使用默认主体尺寸: {default_w}x{default_h}, 占比: 宽度{default_w/width:.1%}, 高度{default_h/height:.1%}")
        return (default_x, default_y, default_w, default_h), (default_center_x, default_center_y)
    except Exception as e:
        print(f"❌ 智能主体检测失败: {e}")
        # 返回图像中心作为默认主体位置
        width, height = image.size
        default_center_x = width // 2
        default_center_y = height // 2

        # 🚀 修复：使用合理的默认主体尺寸
        if width > height:
            default_w = int(width * 0.3)
            default_h = int(height * 0.45)
        else:
            default_w = int(width * 0.45)
            default_h = int(height * 0.3)

        default_w = max(default_w, width // 8)
        default_h = max(default_h, height // 8)
        default_w = min(default_w, int(width * 0.5))
        default_h = min(default_h, int(height * 0.5))

        default_x = default_center_x - default_w // 2
        default_y = default_center_y - default_h // 2

        print(f"🔍 异常处理，使用默认主体尺寸: {default_w}x{default_h}, 占比: 宽度{default_w/width:.1%}, 高度{default_h/height:.1%}")
        return (default_x, default_y, default_w, default_h), (default_center_x, default_center_y)


def test_universal_detection():
    """测试通用主体检测功能"""
    print("🧪 开始测试通用主体检测功能...")
    
    from PIL import Image
    import numpy as np
    
    # 创建一个测试图像
    width, height = 1024, 768
    test_image = Image.new("RGB", (width, height), (255, 255, 255))
    
    print(f"📸 创建测试图像: {width}x{height}")
    
    try:
        # 测试主体检测
        subject_bbox, subject_center = detect_image_foreground_subject(test_image)
        subject_x, subject_y, subject_w, subject_h = subject_bbox
        subject_center_x, subject_center_y = subject_center
        
        print(f"✅ 主体检测成功!")
        print(f"📊 主体边界框: ({subject_x}, {subject_y}, {subject_w}, {subject_h})")
        print(f"🎯 主体中心: ({subject_center_x}, {subject_center_y})")
        
        # 测试智能裁剪计算
        target_width, target_height = 512, 384
        
        # 计算主体中心应该在新图像中的位置
        target_center_x = target_width // 2
        target_center_y = target_height // 2
        
        # 计算裁剪起始位置，使主体中心对齐到目标中心
        crop_x = subject_center_x - target_center_x
        crop_y = subject_center_y - target_center_y
        
        print(f"\n🔧 智能裁剪计算:")
        print(f"🎯 目标尺寸: {target_width}x{target_height}")
        print(f"🎯 目标中心: ({target_center_x}, {target_center_y})")
        print(f"🔧 理论裁剪位置: ({crop_x}, {crop_y})")
        
        # 边界检查和调整
        if crop_x < 0:
            print(f"🔧 调整：裁剪X坐标过小，调整为0")
            crop_x = 0
        elif crop_x + target_width > width:
            print(f"🔧 调整：裁剪X坐标过大，调整为{width - target_width}")
            crop_x = width - target_width
        
        if crop_y < 0:
            print(f"🔧 调整：裁剪Y坐标过小，调整为0")
            crop_y = 0
        elif crop_y + target_height > height:
            print(f"🔧 调整：裁剪Y坐标过大，调整为{height - target_height}")
            crop_y = height - target_height
        
        print(f"🔧 最终裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x + target_width}, {crop_y + target_height})")
        
        # 验证主体是否在裁剪区域内
        subject_in_crop = (crop_x <= subject_x and 
                          crop_x + target_width >= subject_x + subject_w and
                          crop_y <= subject_y and 
                          crop_y + target_height >= subject_y + subject_h)
        
        print(f"🔍 主体完整性检查: {\"✅ 主体完全包含在裁剪区域内\" if subject_in_crop else \"❌ 主体部分丢失\"}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    test_universal_detection()
