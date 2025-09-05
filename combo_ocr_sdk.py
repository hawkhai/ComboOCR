#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComboOCR SDK - 外观增强与扭曲矫正 SDK

提供两个核心功能：
1. 外观增强 (Appearance Enhancement) - 去阴影/去噪
2. 扭曲矫正 (Distortion Correction) - 文档扭曲矫正

基于 GCDRNet 和 DocTr++ 模型实现
"""

import os
import cv2
import torch
import numpy as np
import gc
from typing import Optional, Tuple, Dict, Any
import torch.nn.functional as F

# 导入必要的工具函数和模型
from utils.utils_doctr_plus import DocTr_Plus
from utils.utils_gcdrnet import stride_integral, convert_state_dict
from model.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L


class ComboOCRSDK:
    """
    ComboOCR SDK 主类
    
    提供外观增强和扭曲矫正功能的统一接口
    """
    
    def __init__(self, device: str = "cpu"):
        """
        初始化 SDK
        
        Args:
            device: 计算设备，支持 "cpu", "cuda", "cuda:0" 等
        """
        self.device = torch.device(device)
        self.models = {
            'gcnet': None,      # 外观增强模型1 - 阴影检测
            'drnet': None,      # 外观增强模型2 - 去噪增强
            'dewarp_model': None  # 扭曲矫正模型
        }
        self._enhancement_available = False
        self._dewarp_available = False
        
    def load_enhancement_models(self, 
                              gcnet_path: str = './models/gcdr_net/gcnet/checkpoint.pkl',
                              drnet_path: str = './models/gcdr_net/drnet/checkpoint.pkl') -> bool:
        """
        加载外观增强模型
        
        Args:
            gcnet_path: GCNet 模型路径 (阴影检测)
            drnet_path: DRNet 模型路径 (去噪增强)
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if not os.path.exists(gcnet_path) or not os.path.exists(drnet_path):
                print(f"外观增强模型文件未找到:")
                print(f"  GCNet: {gcnet_path} - {'存在' if os.path.exists(gcnet_path) else '不存在'}")
                print(f"  DRNet: {drnet_path} - {'存在' if os.path.exists(drnet_path) else '不存在'}")
                return False
            
            # 加载 GCNet (阴影检测网络)
            self.models['gcnet'] = UNext_full_resolution_padding(
                num_classes=3, 
                input_channels=3, 
                img_size=512
            ).to(self.device)
            
            state = convert_state_dict(
                torch.load(gcnet_path, map_location=self.device)['model_state']
            )
            self.models['gcnet'].load_state_dict(state)
            self.models['gcnet'].eval()
            print("✅ GCNet 模型加载成功")
            
            # 加载 DRNet (去噪增强网络)
            self.models['drnet'] = UNext_full_resolution_padding_L_py_L(
                num_classes=3, 
                input_channels=6, 
                img_size=512
            ).to(self.device)
            
            state = convert_state_dict(
                torch.load(drnet_path, map_location=self.device)['model_state']
            )
            self.models['drnet'].load_state_dict(state)
            self.models['drnet'].eval()
            print("✅ DRNet 模型加载成功")
            
            self._enhancement_available = True
            return True
            
        except Exception as e:
            print(f"❌ 外观增强模型加载失败: {str(e)}")
            self.models['gcnet'] = None
            self.models['drnet'] = None
            self._enhancement_available = False
            return False
    
    def load_dewarp_model(self, model_path: str = "./models/doctr_plus/model.pt") -> bool:
        """
        加载扭曲矫正模型
        
        Args:
            model_path: DocTr++ 模型路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if not os.path.exists(model_path):
                print(f"❌ 扭曲矫正模型文件不存在: {model_path}")
                return False
            
            self.models['dewarp_model'] = DocTr_Plus(weights=model_path, device=self.device)
            print("✅ 扭曲矫正模型加载成功")
            
            self._dewarp_available = True
            return True
            
        except Exception as e:
            print(f"❌ 扭曲矫正模型加载失败: {str(e)}")
            self.models['dewarp_model'] = None
            self._dewarp_available = False
            return False
    
    def load_all_models(self, 
                       gcnet_path: str = './models/gcdr_net/gcnet/checkpoint.pkl',
                       drnet_path: str = './models/gcdr_net/drnet/checkpoint.pkl',
                       dewarp_path: str = "./models/doctr_plus/model.pt") -> Dict[str, bool]:
        """
        加载所有模型
        
        Args:
            gcnet_path: GCNet 模型路径
            drnet_path: DRNet 模型路径  
            dewarp_path: DocTr++ 模型路径
            
        Returns:
            Dict[str, bool]: 各模型加载状态
        """
        results = {
            'enhancement': self.load_enhancement_models(gcnet_path, drnet_path),
            'dewarp': self.load_dewarp_model(dewarp_path)
        }
        
        print(f"\n📊 模型加载总结:")
        print(f"  外观增强: {'✅ 可用' if results['enhancement'] else '❌ 不可用'}")
        print(f"  扭曲矫正: {'✅ 可用' if results['dewarp'] else '❌ 不可用'}")
        
        return results
    
    def enhance_appearance(self, image: np.ndarray) -> np.ndarray:
        """
        应用外观增强处理 (去阴影/去噪)
        
        Args:
            image: 输入图像 (BGR格式, uint8)
            
        Returns:
            np.ndarray: 增强后的图像
            
        Raises:
            RuntimeError: 当模型未加载时
        """
        if not self._enhancement_available:
            raise RuntimeError("外观增强模型未加载，请先调用 load_enhancement_models()")
        
        print("🎨 开始外观增强处理...")
        
        # 图像预处理 - 确保尺寸是32的倍数
        im_org, padding_h, padding_w = stride_integral(image)
        h, w = im_org.shape[:2]
        im = im_org
        
        # 预先声明变量，便于清理
        im_tensor = None
        im_org_tensor = None
        shadow = None
        model1_im = None
        pred = None
        
        try:
            with torch.no_grad():
                # 创建张量并移动到设备
                im_tensor = torch.from_numpy(im.transpose(2, 0, 1) / 255).unsqueeze(0).float().to(self.device)
                im_org_tensor = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0).float().to(self.device)
                
                # 第一步：阴影检测 (GCNet)
                shadow = self.models['gcnet'](im_tensor)
                shadow = F.interpolate(shadow, (h, w))
                
                # 第二步：去噪增强 (DRNet)
                model1_im = torch.clamp(im_org_tensor / shadow, 0, 1)
                pred, _, _, _ = self.models['drnet'](torch.cat((im_org_tensor, model1_im), 1))
                
                # 转换为 numpy，立即移动到 CPU
                pred_np = pred[0].permute(1, 2, 0).detach().cpu().numpy()
                pred_np = (pred_np * 255).astype(np.uint8)
                pred_np = pred_np[padding_h:, padding_w:]
                
        finally:
            # 显式删除GPU张量
            for tensor in [im_tensor, im_org_tensor, shadow, model1_im, pred]:
                if tensor is not None:
                    del tensor
            # 强制清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 触发垃圾回收
            gc.collect()
        
        enhanced_image = np.clip(pred_np, 0, 255).astype(np.uint8)
        print("✅ 外观增强处理完成")
        return enhanced_image
    
    def correct_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        应用扭曲矫正处理
        
        Args:
            image: 输入图像 (BGR格式, uint8)
            
        Returns:
            np.ndarray: 矫正后的图像
            
        Raises:
            RuntimeError: 当模型未加载时
        """
        if not self._dewarp_available:
            raise RuntimeError("扭曲矫正模型未加载，请先调用 load_dewarp_model()")
        
        print("📐 开始扭曲矫正处理...")
        
        # 图像预处理
        img_ori = image / 255.0
        h_, w_, c_ = img_ori.shape
        img_ori = cv2.resize(img_ori, (2560, 2560))
        h, w, _ = img_ori.shape
        img = cv2.resize(img_ori, (288, 288))
        img = img.transpose(2, 0, 1)
        
        # 预先声明变量
        img_tensor = None
        bm = None
        
        try:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                bm = self.models['dewarp_model'](img_tensor)
                bm_np = bm.detach().cpu().numpy()[0]  # 立即移动到 CPU
                
            bm0 = bm_np[0, :, :]
            bm1 = bm_np[1, :, :]
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            
            img_geo = cv2.remap(img_ori, bm0, bm1, cv2.INTER_LINEAR) * 255
            img_geo = cv2.resize(img_geo, (w_, h_))
            
        finally:
            # 显式删除GPU张量  
            for tensor in [img_tensor, bm]:
                if tensor is not None:
                    del tensor  
            # 强制清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 触发垃圾回收
            gc.collect()
        
        # 确保图像是uint8类型
        corrected_image = np.clip(img_geo, 0, 255).astype(np.uint8)
        print("✅ 扭曲矫正处理完成")
        return corrected_image
    
    def process_image(self, 
                     image: np.ndarray, 
                     use_enhancement: bool = False, 
                     use_dewarp: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        综合处理图像 (可选择启用外观增强和扭曲矫正)
        
        Args:
            image: 输入图像 (BGR格式, uint8)
            use_enhancement: 是否启用外观增强
            use_dewarp: 是否启用扭曲矫正
            
        Returns:
            Tuple[np.ndarray, Dict]: (处理后的图像, 处理信息)
        """
        if image is None:
            raise ValueError("输入图像不能为空")
        
        processed_image = image.copy()
        processing_info = {
            'use_enhancement': use_enhancement,
            'use_dewarp': use_dewarp,
            'enhancement_applied': False,
            'dewarp_applied': False,
            'original_shape': image.shape,
            'final_shape': None
        }
        
        print(f"🚀 开始图像处理:")
        print(f"   - 外观增强: {'启用' if use_enhancement else '禁用'}")
        print(f"   - 扭曲矫正: {'启用' if use_dewarp else '禁用'}")
        
        # 1. 可选：外观增强
        if use_enhancement:
            if self._enhancement_available:
                processed_image = self.enhance_appearance(processed_image)
                processing_info['enhancement_applied'] = True
            else:
                print("⚠️  外观增强模型未加载，跳过外观增强步骤")
        else:
            print("⏭️  跳过外观增强步骤")
        
        # 2. 可选：扭曲矫正
        if use_dewarp:
            if self._dewarp_available:
                processed_image = self.correct_distortion(processed_image)
                processing_info['dewarp_applied'] = True
            else:
                print("⚠️  扭曲矫正模型未加载，跳过扭曲矫正步骤")
        else:
            print("⏭️  跳过扭曲矫正步骤")
        
        processing_info['final_shape'] = processed_image.shape
        print(f"✅ 图像处理完成! 原始尺寸: {processing_info['original_shape'][:2]} -> 最终尺寸: {processing_info['final_shape'][:2]}")
        
        return processed_image, processing_info
    
    def process_image_file(self, 
                          input_path: str, 
                          output_path: Optional[str] = None,
                          use_enhancement: bool = False, 
                          use_dewarp: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        处理图像文件
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径 (可选，默认为输入路径加后缀)
            use_enhancement: 是否启用外观增强
            use_dewarp: 是否启用扭曲矫正
            
        Returns:
            Tuple[str, Dict]: (输出文件路径, 处理信息)
        """
        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {input_path}")
        
        # 处理图像
        processed_image, processing_info = self.process_image(
            image, use_enhancement, use_dewarp
        )
        
        # 生成输出路径
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            ext = os.path.splitext(input_path)[1]
            suffix = []
            if use_enhancement:
                suffix.append("enhanced")
            if use_dewarp:
                suffix.append("dewarped")
            suffix_str = "_" + "_".join(suffix) if suffix else "_processed"
            output_path = f"{base_name}{suffix_str}{ext}"
        
        # 保存图像
        cv2.imwrite(output_path, processed_image)
        processing_info['input_path'] = input_path
        processing_info['output_path'] = output_path
        
        print(f"💾 图像已保存到: {output_path}")
        return output_path, processing_info
    
    @property
    def enhancement_available(self) -> bool:
        """外观增强功能是否可用"""
        return self._enhancement_available
    
    @property  
    def dewarp_available(self) -> bool:
        """扭曲矫正功能是否可用"""
        return self._dewarp_available
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取 SDK 状态信息
        
        Returns:
            Dict: 状态信息
        """
        return {
            'device': str(self.device),
            'enhancement_available': self._enhancement_available,
            'dewarp_available': self._dewarp_available,
            'models_loaded': {
                'gcnet': self.models['gcnet'] is not None,
                'drnet': self.models['drnet'] is not None,
                'dewarp_model': self.models['dewarp_model'] is not None
            }
        }


# 便捷函数
def create_sdk(device: str = "cpu") -> ComboOCRSDK:
    """
    创建 ComboOCR SDK 实例的便捷函数
    
    Args:
        device: 计算设备
        
    Returns:
        ComboOCRSDK: SDK 实例
    """
    return ComboOCRSDK(device=device)


def quick_enhance(image_path: str, 
                 output_path: Optional[str] = None,
                 device: str = "cpu") -> str:
    """
    快速外观增强处理
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        device: 计算设备
        
    Returns:
        str: 输出文件路径
    """
    sdk = create_sdk(device)
    sdk.load_enhancement_models()
    output_path, _ = sdk.process_image_file(
        image_path, output_path, use_enhancement=True, use_dewarp=False
    )
    return output_path


def quick_dewarp(image_path: str, 
                output_path: Optional[str] = None,
                device: str = "cpu") -> str:
    """
    快速扭曲矫正处理
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径  
        device: 计算设备
        
    Returns:
        str: 输出文件路径
    """
    sdk = create_sdk(device)
    sdk.load_dewarp_model()
    output_path, _ = sdk.process_image_file(
        image_path, output_path, use_enhancement=False, use_dewarp=True
    )
    return output_path


def quick_process(image_path: str,
                 output_path: Optional[str] = None, 
                 use_enhancement: bool = True,
                 use_dewarp: bool = True,
                 device: str = "cpu") -> str:
    """
    快速综合处理 (外观增强 + 扭曲矫正)
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        use_enhancement: 是否启用外观增强
        use_dewarp: 是否启用扭曲矫正
        device: 计算设备
        
    Returns:
        str: 输出文件路径
    """
    sdk = create_sdk(device)
    sdk.load_all_models()
    output_path, _ = sdk.process_image_file(
        image_path, output_path, use_enhancement, use_dewarp
    )
    return output_path


if __name__ == "__main__":
    # 示例用法
    print("🎯 ComboOCR SDK 示例")
    
    # 创建 SDK 实例
    sdk = create_sdk(device="cpu")  # 或 "cuda" 如果有GPU
    
    # 加载所有模型
    results = sdk.load_all_models()
    
    # 检查状态
    status = sdk.get_status()
    print(f"\n📊 SDK 状态: {status}")
    
    # 示例：处理图像文件
    # output_path = sdk.process_image_file(
    #     "input.jpg", 
    #     "output.jpg",
    #     use_enhancement=True, 
    #     use_dewarp=True
    # )
    
    print("\n📚 使用说明:")
    print("1. 创建 SDK: sdk = create_sdk()")
    print("2. 加载模型: sdk.load_all_models()")  
    print("3. 处理图像: sdk.process_image_file('input.jpg', use_enhancement=True, use_dewarp=True)")
    print("4. 或使用便捷函数: quick_process('input.jpg')")
