"""
异常检测器模型
用于阶段3的最终异常检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .backbone import BackboneFactory


class AnomalyDetector(nn.Module):
    """
    基于特征重构的异常检测器
    
    架构:
    Input -> Backbone -> Feature Processor -> Reconstruction Head
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        feature_dim: int = 256,
        pretrained: bool = True
    ):
        super().__init__()
        
        # 骨干网络
        self.backbone = BackboneFactory.create(backbone_name, pretrained)
        backbone_feat_dim = BackboneFactory.get_feature_dim(backbone_name)
        
        # 特征处理器（2D Conv，用于特征图）
        self.feature_processor = nn.Sequential(
            nn.Conv2d(backbone_feat_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 特征处理器（1D Linear，用于池化后的向量）
        self.feature_processor_1d = nn.Sequential(
            nn.Linear(backbone_feat_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 重构头（2D Conv，用于特征图）
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, backbone_feat_dim, kernel_size=1)
        )
        
        # 重构头（1D Linear，用于池化后的向量）
        self.reconstruction_head_1d = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, backbone_feat_dim)
        )
        
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            dict包含:
                - features: 处理后的特征
                - reconstructed: 重构的特征
        """
        # 提取特征
        features = self.backbone(x)
        
        # 如果backbone返回的是列表，取最后一个元素
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # 如果是1D向量（已池化），直接处理
        if features.dim() == 2:
            # 对于1D特征，使用Linear层而非Conv2d
            processed = self.feature_processor_1d(features) if hasattr(self, 'feature_processor_1d') else features
            reconstructed = self.reconstruction_head_1d(processed) if hasattr(self, 'reconstruction_head_1d') else processed
        else:
            # 对于2D特征图，使用Conv2d
            processed = self.feature_processor(features)
            reconstructed = self.reconstruction_head(processed)
        
        return {
            'features': processed,
            'reconstructed': reconstructed
        }
    
    def compute_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算异常图
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            anomaly_map: 异常图 [B, 1, H, W]
        """
        output = self.forward(x)
        features = output['features']
        reconstructed = output['reconstructed']
        
        # 计算逐像素的L2距离
        anomaly_map = torch.norm(features - reconstructed, dim=1, keepdim=True)
        
        # 归一化到[0, 1]
        B, _, H, W = anomaly_map.shape
        anomaly_map = anomaly_map.view(B, -1)
        min_val = anomaly_map.min(dim=1, keepdim=True)[0]
        max_val = anomaly_map.max(dim=1, keepdim=True)[0]
        anomaly_map = (anomaly_map - min_val) / (max_val - min_val + 1e-8)
        anomaly_map = anomaly_map.view(B, 1, H, W)
        
        return anomaly_map
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        features = self.backbone(x)
        
        # 如果backbone返回的是列表，取最后一个元素
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # 根据维度选择处理器
        if features.dim() == 2:
            processed = self.feature_processor_1d(features)
        else:
            processed = self.feature_processor(features)
        
        return processed


class MultiScaleAnomalyDetector(nn.Module):
    """
    多尺度异常检测器
    
    使用多个stage的特征进行异常检测
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        feature_dims: list = [64, 128, 256],
        pretrained: bool = True
    ):
        super().__init__()
        
        # 骨干网络（返回多个stage的特征）
        self.backbone = BackboneFactory.create(backbone_name, pretrained)
        # 修改为返回多个stage
        self.backbone = timm.create_model(
            BackboneFactory.SUPPORTED_BACKBONES[backbone_name]['timm_name'],
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3)  # 返回3个stage的特征
        )
        
        # 特征融合
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, feature_dims[i], kernel_size=1),
                nn.BatchNorm2d(feature_dims[i]),
                nn.ReLU(inplace=True)
            )
            for i, dim in enumerate([64, 128, 256])  # ResNet的典型维度
        ])
        
        self.feature_dims = feature_dims
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取多尺度特征
        multi_scale_features = self.backbone(x)
        
        # 处理每个尺度的特征
        processed_features = []
        for i, (feat, fusion) in enumerate(zip(multi_scale_features, self.feature_fusion)):
            processed = fusion(feat)
            processed_features.append(processed)
        
        return {
            'multi_scale_features': processed_features
        }
    
    def compute_anomaly_score(self, x: torch.Tensor, normal_prototypes: torch.Tensor) -> torch.Tensor:
        """
        计算异常分数
        
        Args:
            x: 输入图像
            normal_prototypes: 正常样本的原型特征
        
        Returns:
            anomaly_scores: 异常分数
        """
        features = self.get_features(x)
        
        # 计算与正常原型的距离
        distances = torch.norm(features - normal_prototypes, dim=1)
        
        return distances
