#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图像增强处理 Demo

给定输入文件夹，对其中所有图片进行外观增强和扭曲矫正处理，
并将结果保存到输出文件夹中。
"""

import os
import cv2
import time
from pathlib import Path
from tqdm import tqdm
from combo_ocr_sdk import create_sdk


class BatchEnhanceDemo:
    """批量图像增强处理类"""
    
    def __init__(self, device="cpu"):
        """
        初始化批量处理器
        
        Args:
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.device = device
        self.sdk = None
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def initialize_sdk(self):
        """初始化 SDK 和模型"""
        print("🚀 正在初始化 ComboOCR SDK...")
        self.sdk = create_sdk(device=self.device)
        
        print("📦 正在加载模型...")
        results = self.sdk.load_all_models()
        
        if not results['enhancement'] and not results['dewarp']:
            raise RuntimeError("❌ 没有可用的模型，请检查模型文件路径")
        
        print(f"✅ 模型加载完成:")
        print(f"   - 外观增强: {'✅ 可用' if results['enhancement'] else '❌ 不可用'}")
        print(f"   - 扭曲矫正: {'✅ 可用' if results['dewarp'] else '❌ 不可用'}")
        
        return results
    
    def scan_images(self, input_folder):
        """
        扫描输入文件夹中的所有图像文件
        
        Args:
            input_folder: 输入文件夹路径
            
        Returns:
            list: 图像文件路径列表
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"❌ 输入文件夹不存在: {input_folder}")
        
        image_files = []
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def process_batch(self, input_folder, output_folder, 
                     use_enhancement=True, use_dewarp=True,
                     preserve_structure=True):
        """
        批量处理文件夹中的图像
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            use_enhancement: 是否启用外观增强
            use_dewarp: 是否启用扭曲矫正
            preserve_structure: 是否保持文件夹结构
            
        Returns:
            dict: 处理结果统计
        """
        # 初始化 SDK
        model_results = self.initialize_sdk()
        
        # 检查功能可用性
        if use_enhancement and not model_results['enhancement']:
            print("⚠️  外观增强模型不可用，将跳过外观增强")
            use_enhancement = False
            
        if use_dewarp and not model_results['dewarp']:
            print("⚠️  扭曲矫正模型不可用，将跳过扭曲矫正")
            use_dewarp = False
        
        if not use_enhancement and not use_dewarp:
            raise ValueError("❌ 没有启用任何处理功能")
        
        # 扫描图像文件
        print(f"📁 正在扫描输入文件夹: {input_folder}")
        image_files = self.scan_images(input_folder)
        
        if not image_files:
            print(f"❌ 在输入文件夹中未找到支持的图像文件")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        
        print(f"📊 找到 {len(image_files)} 个图像文件")
        
        # 创建输出文件夹
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理统计
        stats = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': time.time(),
            'failed_files': []
        }
        
        # 批量处理
        print(f"🔄 开始批量处理...")
        print(f"   - 外观增强: {'✅ 启用' if use_enhancement else '❌ 禁用'}")
        print(f"   - 扭曲矫正: {'✅ 启用' if use_dewarp else '❌ 禁用'}")
        print(f"   - 保持结构: {'✅ 是' if preserve_structure else '❌ 否'}")
        
        for i, image_path in enumerate(tqdm(image_files, desc="处理进度")):
            try:
                # 计算相对路径和输出路径
                input_path_obj = Path(input_folder)
                image_path_obj = Path(image_path)
                
                if preserve_structure:
                    # 保持文件夹结构
                    relative_path = image_path_obj.relative_to(input_path_obj)
                    output_file_path = output_path / relative_path
                else:
                    # 所有文件放在输出根目录
                    output_file_path = output_path / image_path_obj.name
                
                # 创建输出子目录
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 检查输出文件是否已存在
                if output_file_path.exists():
                    print(f"⏭️  跳过已存在的文件: {output_file_path.name}")
                    stats['skipped'] += 1
                    continue
                
                # 处理图像
                processed_path, info = self.sdk.process_image_file(
                    input_path=str(image_path),
                    output_path=str(output_file_path),
                    use_enhancement=use_enhancement,
                    use_dewarp=use_dewarp
                )
                
                stats['success'] += 1
                
                # 每处理10个文件显示一次进度
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - stats['start_time']
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (len(image_files) - i - 1)
                    print(f"📈 进度: {i+1}/{len(image_files)} | "
                          f"成功: {stats['success']} | "
                          f"预计剩余: {remaining:.1f}秒")
                
            except Exception as e:
                stats['failed'] += 1
                stats['failed_files'].append({
                    'file': image_path,
                    'error': str(e)
                })
                print(f"❌ 处理失败: {Path(image_path).name} - {str(e)}")
        
        # 计算总耗时
        total_time = time.time() - stats['start_time']
        stats['total_time'] = total_time
        
        return stats
    
    def print_summary(self, stats):
        """打印处理结果摘要"""
        print("\n" + "=" * 60)
        print("📊 批量处理完成!")
        print("=" * 60)
        print(f"总文件数: {stats['total']}")
        print(f"处理成功: {stats['success']} ✅")
        print(f"处理失败: {stats['failed']} ❌")
        print(f"跳过文件: {stats['skipped']} ⏭️")
        print(f"总耗时: {stats['total_time']:.1f} 秒")
        
        if stats['success'] > 0:
            avg_time = stats['total_time'] / stats['success']
            print(f"平均处理时间: {avg_time:.2f} 秒/图")
        
        if stats['failed'] > 0:
            print(f"\n❌ 失败文件列表:")
            for failed in stats['failed_files']:
                print(f"   - {Path(failed['file']).name}: {failed['error']}")


def main():
    """主函数"""
    print("🎯 批量图像增强处理 Demo")
    print("=" * 60)
    
    # 配置参数
    input_folder = r"I:\pdfai_serv\classify\seal_extract\extracted_figures"
    output_folder = r"I:\pdfai_serv\classify\seal_extract\extracted_figures2"
    
    # 处理选项
    use_enhancement = True   # 启用外观增强
    use_dewarp = False      # 禁用扭曲矫正
    preserve_structure = True  # 保持文件夹结构
    device = "cpu"          # 使用设备 ("cpu" 或 "cuda")
    
    print(f"📁 输入文件夹: {input_folder}")
    print(f"📁 输出文件夹: {output_folder}")
    print(f"🎨 外观增强: {'启用' if use_enhancement else '禁用'}")
    print(f"📐 扭曲矫正: {'启用' if use_dewarp else '禁用'}")
    print(f"🏗️  保持结构: {'是' if preserve_structure else '否'}")
    print(f"💻 计算设备: {device}")
    
    try:
        # 创建批量处理器
        processor = BatchEnhanceDemo(device=device)
        
        # 执行批量处理
        stats = processor.process_batch(
            input_folder=input_folder,
            output_folder=output_folder,
            use_enhancement=use_enhancement,
            use_dewarp=use_dewarp,
            preserve_structure=preserve_structure
        )
        
        # 打印结果摘要
        processor.print_summary(stats)
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
