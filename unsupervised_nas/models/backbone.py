"""
骨干网络定义
支持多种backbone用于NAS搜索
"""

import torch
import torch.nn as nn
from typing import Dict, List
import timm


class BackboneFactory:
    """骨干网络工厂"""
    
    SUPPORTED_BACKBONES = {
        # ===== 2023-2024 最新SOTA CNN ⭐⭐⭐ (顶刊必备) =====
        
        # ConvNeXt V2系列 (2023) - 使用GRN归一化
        'convnextv2_tiny': {'features': 768, 'timm_name': 'convnextv2_tiny'},
        'convnextv2_base': {'features': 1024, 'timm_name': 'convnextv2_base'},
        
        # EfficientNet V2系列 (2021-2024) - 快速训练
        'efficientnetv2_s': {'features': 1280, 'timm_name': 'efficientnetv2_s'},
        
        # MobileOne (2022-2023) - 超快推理
        'mobileone_s0': {'features': 1024, 'timm_name': 'mobileone_s0'},  # 修正: 实际维度1024
        'mobileone_s3': {'features': 512, 'timm_name': 'mobileone_s3'},
        'mobileone_s4': {'features': 640, 'timm_name': 'mobileone_s4'},
        
        # ===== 2022-2023 现代CNN ⭐⭐ =====
        
        # ConvNeXt原版 (2022)
        'convnext_tiny': {'features': 768, 'timm_name': 'convnext_tiny'},
        'convnext_small': {'features': 768, 'timm_name': 'convnext_small'},
        
        # ===== Vision Transformers (2020-2023) ⭐⭐⭐ (顶刊关键) =====
        
        # ViT (Vision Transformer, 2020)
        'vit_tiny_patch16_224': {'features': 192, 'timm_name': 'vit_tiny_patch16_224'},
        
        # DeiT (Data-efficient ViT, 2021)
        'deit_tiny_patch16_224': {'features': 192, 'timm_name': 'deit_tiny_patch16_224'},
        
        # Swin Transformer (2021, 层次化设计)
        'swin_tiny_patch4_window7_224': {'features': 768, 'timm_name': 'swin_tiny_patch4_window7_224'},
        
        # ===== 2019-2021 经典高效架构 ⭐⭐ =====
        
        # EfficientNet原版 (2019)
        'efficientnet_b0': {'features': 1280, 'timm_name': 'efficientnet_b0'},
        'efficientnet_b1': {'features': 1280, 'timm_name': 'efficientnet_b1'},
        
        # MobileNet V3 (2019, 移动端优化)
        'mobilenetv3_small_100': {'features': 1024, 'timm_name': 'mobilenetv3_small_100'},  # 修正: 实际维度1024
        
        # ===== 2015-2016 经典基线 ⭐ (用于对比) =====
        
        # ResNet系列 (2015-2016)
        'resnet18': {'features': 512, 'timm_name': 'resnet18'},
        'resnet34': {'features': 512, 'timm_name': 'resnet34'},
        'resnet50': {'features': 2048, 'timm_name': 'resnet50'},
    }
    
    @staticmethod
    def create(backbone_name: str, pretrained: bool = True) -> nn.Module:
        """创建骨干网络"""
        if backbone_name not in BackboneFactory.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        config = BackboneFactory.SUPPORTED_BACKBONES[backbone_name]
        model = timm.create_model(
            config['timm_name'],
            pretrained=pretrained,
            num_classes=0,  # 移除分类头，返回全局池化后的特征
            global_pool='avg'  # 使用全局平均池化
        )
        
        return model
    
    @staticmethod
    def get_feature_dim(backbone_name: str) -> int:
        """获取骨干网络的输出特征维度"""
        if backbone_name not in BackboneFactory.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        return BackboneFactory.SUPPORTED_BACKBONES[backbone_name]['features']
    
    @staticmethod
    def list_supported() -> List[str]:
        """列出支持的骨干网络"""
        return list(BackboneFactory.SUPPORTED_BACKBONES.keys())
