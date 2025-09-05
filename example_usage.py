#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComboOCR SDK 使用示例

展示如何使用 ComboOCR SDK 进行外观增强和扭曲矫正
"""

import cv2
import os
from combo_ocr_sdk import ComboOCRSDK, create_sdk, quick_enhance, quick_dewarp, quick_process


def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("=" * 60)
    print("示例1: 基本使用方法")
    print("=" * 60)
    
    # 创建 SDK 实例
    sdk = create_sdk(device="cpu")  # 使用 CPU，如果有 GPU 可以改为 "cuda"
    
    # 加载所有模型
    print("正在加载模型...")
    results = sdk.load_all_models()
    
    # 检查加载状态
    if results['enhancement']:
        print("✅ 外观增强模型加载成功")
    else:
        print("❌ 外观增强模型加载失败")
        
    if results['dewarp']:
        print("✅ 扭曲矫正模型加载成功")
    else:
        print("❌ 扭曲矫正模型加载失败")
    
    # 显示 SDK 状态
    status = sdk.get_status()
    print(f"\nSDK 状态: {status}")


def example_2_process_single_image():
    """示例2: 处理单张图像"""
    print("\n" + "=" * 60)
    print("示例2: 处理单张图像")
    print("=" * 60)
    
    # 检查测试图像是否存在
    test_image_path = "./images/1.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        print("请确保 images 目录下有测试图像")
        return
    
    # 创建 SDK 并加载模型
    sdk = create_sdk(device="cpu")
    sdk.load_all_models()
    
    try:
        # 处理图像 - 同时启用外观增强和扭曲矫正
        output_path, info = sdk.process_image_file(
            input_path=test_image_path,
            output_path="./temp_uploads/processed_example.jpg",
            use_enhancement=True,
            use_dewarp=True
        )
        
        print(f"✅ 图像处理完成!")
        print(f"   输入: {info['input_path']}")
        print(f"   输出: {info['output_path']}")
        print(f"   外观增强: {'✅ 已应用' if info['enhancement_applied'] else '❌ 未应用'}")
        print(f"   扭曲矫正: {'✅ 已应用' if info['dewarp_applied'] else '❌ 未应用'}")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


def example_3_process_with_opencv():
    """示例3: 使用 OpenCV 读取图像进行处理"""
    print("\n" + "=" * 60)
    print("示例3: 使用 OpenCV 读取图像进行处理")
    print("=" * 60)
    
    test_image_path = "./images/2.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    # 读取图像
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"❌ 无法读取图像: {test_image_path}")
        return
    
    print(f"📷 原始图像尺寸: {image.shape}")
    
    # 创建 SDK
    sdk = create_sdk(device="cpu")
    sdk.load_all_models()
    
    try:
        # 处理图像
        processed_image, info = sdk.process_image(
            image=image,
            use_enhancement=True,
            use_dewarp=True
        )
        
        # 保存结果
        output_path = "./temp_uploads/opencv_processed.jpg"
        cv2.imwrite(output_path, processed_image)
        
        print(f"✅ 处理完成!")
        print(f"   原始尺寸: {info['original_shape']}")
        print(f"   最终尺寸: {info['final_shape']}")
        print(f"   已保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


def example_4_enhancement_only():
    """示例4: 仅使用外观增强"""
    print("\n" + "=" * 60)
    print("示例4: 仅使用外观增强")
    print("=" * 60)
    
    test_image_path = "./images/111.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    # 创建 SDK 并仅加载外观增强模型
    sdk = create_sdk(device="cpu")
    enhancement_loaded = sdk.load_enhancement_models()
    
    if not enhancement_loaded:
        print("❌ 外观增强模型加载失败")
        return
    
    try:
        # 仅进行外观增强
        output_path, info = sdk.process_image_file(
            input_path=test_image_path,
            output_path="./temp_uploads/enhanced_only.jpg",
            use_enhancement=True,
            use_dewarp=False  # 不使用扭曲矫正
        )
        
        print(f"✅ 外观增强完成: {output_path}")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


def example_5_dewarp_only():
    """示例5: 仅使用扭曲矫正"""
    print("\n" + "=" * 60)
    print("示例5: 仅使用扭曲矫正")
    print("=" * 60)
    
    test_image_path = "./images/222.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    # 创建 SDK 并仅加载扭曲矫正模型
    sdk = create_sdk(device="cpu")
    dewarp_loaded = sdk.load_dewarp_model()
    
    if not dewarp_loaded:
        print("❌ 扭曲矫正模型加载失败")
        return
    
    try:
        # 仅进行扭曲矫正
        output_path, info = sdk.process_image_file(
            input_path=test_image_path,
            output_path="./temp_uploads/dewarped_only.jpg",
            use_enhancement=False,  # 不使用外观增强
            use_dewarp=True
        )
        
        print(f"✅ 扭曲矫正完成: {output_path}")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


def example_6_quick_functions():
    """示例6: 使用便捷函数"""
    print("\n" + "=" * 60)
    print("示例6: 使用便捷函数")
    print("=" * 60)
    
    test_images = ["./images/1.jpg", "./images/2.jpg"]
    
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"❌ 图像不存在: {image_path}")
            continue
            
        print(f"\n处理图像 {i+1}: {image_path}")
        
        try:
            # 使用便捷函数进行快速处理
            if i == 0:
                # 仅外观增强
                output_path = quick_enhance(
                    image_path, 
                    f"./temp_uploads/quick_enhanced_{i+1}.jpg"
                )
                print(f"✅ 快速外观增强完成: {output_path}")
                
            elif i == 1:
                # 仅扭曲矫正
                output_path = quick_dewarp(
                    image_path,
                    f"./temp_uploads/quick_dewarped_{i+1}.jpg"
                )
                print(f"✅ 快速扭曲矫正完成: {output_path}")
                
        except Exception as e:
            print(f"❌ 快速处理失败: {str(e)}")


def example_7_batch_processing():
    """示例7: 批量处理"""
    print("\n" + "=" * 60)
    print("示例7: 批量处理")
    print("=" * 60)
    
    # 获取 images 目录下的所有图像文件
    images_dir = "./images"
    if not os.path.exists(images_dir):
        print(f"❌ 图像目录不存在: {images_dir}")
        return
    
    # 支持的图像格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for filename in os.listdir(images_dir):
        if os.path.splitext(filename.lower())[1] in supported_formats:
            image_files.append(os.path.join(images_dir, filename))
    
    if not image_files:
        print(f"❌ 在 {images_dir} 中未找到支持的图像文件")
        return
    
    print(f"📁 找到 {len(image_files)} 个图像文件")
    
    # 创建 SDK
    sdk = create_sdk(device="cpu")
    results = sdk.load_all_models()
    
    if not any(results.values()):
        print("❌ 没有可用的模型")
        return
    
    # 批量处理
    success_count = 0
    for i, image_path in enumerate(image_files[:3]):  # 限制处理前3个文件作为示例
        try:
            print(f"\n处理 {i+1}/{min(3, len(image_files))}: {os.path.basename(image_path)}")
            
            output_path, info = sdk.process_image_file(
                input_path=image_path,
                use_enhancement=results['enhancement'],
                use_dewarp=results['dewarp']
            )
            
            print(f"✅ 完成: {os.path.basename(output_path)}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 失败: {str(e)}")
    
    print(f"\n📊 批量处理完成: {success_count}/{min(3, len(image_files))} 成功")


def main():
    """主函数 - 运行所有示例"""
    print("🎯 ComboOCR SDK 使用示例")
    print("本示例将演示 SDK 的各种使用方法")
    
    # 确保输出目录存在
    os.makedirs("./temp_uploads", exist_ok=True)
    
    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_process_single_image()
        example_3_process_with_opencv()
        example_4_enhancement_only()
        example_5_dewarp_only()
        example_6_quick_functions()
        example_7_batch_processing()
        
        print("\n" + "=" * 60)
        print("🎉 所有示例运行完成!")
        print("=" * 60)
        print("💡 提示:")
        print("  - 检查 temp_uploads 目录查看处理结果")
        print("  - 根据你的硬件配置调整 device 参数 ('cpu' 或 'cuda')")
        print("  - 确保模型文件路径正确")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行出错: {str(e)}")


if __name__ == "__main__":
    main()
