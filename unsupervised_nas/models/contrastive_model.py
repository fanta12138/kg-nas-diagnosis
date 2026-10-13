"""
对比学习模型 (SimCLR)
用于阶段1的无监督预训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .backbone import BackboneFactory


class ProjectionHead(nn.Module):
    """投影头：将特征映射到对比学习空间"""
    
    def __init__(self, input_dim: int, projection_dim: int = 128, hidden_dim: int = 2048):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ContrastiveModel(nn.Module):
    """
    SimCLR对比学习模型
    
    架构:
    Input -> Backbone -> Projector -> Embedding
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        projection_dim: int = 128,
        pretrained: bool = True
    ):
        super().__init__()
        
        # 骨干网络
        self.backbone = BackboneFactory.create(backbone_name, pretrained)
        backbone_feat_dim = BackboneFactory.get_feature_dim(backbone_name)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 投影头
        self.projector = ProjectionHead(
            input_dim=backbone_feat_dim,
            projection_dim=projection_dim
        )
        
        self.backbone_name = backbone_name
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            z: 投影后的特征 [B, projection_dim]
        """
        # 提取特征（backbone已经包含全局池化）
        features = self.backbone(x)
        
        # 如果backbone返回的是列表，取最后一个元素
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # 确保是2D张量 [B, D]
        if features.dim() > 2:
            features = self.global_pool(features).flatten(1)
        
        # 投影
        z = self.projector(features)
        
        return z
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取骨干网络的特征（不经过投影头）"""
        features = self.backbone(x)
        
        # 如果backbone返回的是列表，取最后一个元素
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # 确保是2D张量 [B, D]
        if features.dim() > 2:
            features = self.global_pool(features).flatten(1)
        
        return features
    
    @torch.no_grad()
    def extract_features(self, dataloader, device: torch.device) -> torch.Tensor:
        """从数据加载器中提取所有样本的特征"""
        self.eval()
        all_features = []
        
        for images, _ in dataloader:
            images = images.to(device)
            features = self.get_features(images)
            all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)


class SimCLRLoss(nn.Module):
    """
    SimCLR对比损失
    
    使用NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            z_i: 第一个增强视图的投影特征 [B, D]
            z_j: 第二个增强视图的投影特征 [B, D]
        
        Returns:
            loss: 对比损失值
        """
        batch_size = z_i.size(0)
        
        # 拼接两个视图的特征 [2B, D]
        representations = torch.cat([z_i, z_j], dim=0)
        
        # 计算相似度矩阵 [2B, 2B]
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        
        # 移除对角线元素（自身相似度）
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # 正样本对的位置
        # z_i的第i个样本与z_j的第i个样本是正对
        positives = torch.cat([z_j, z_i], dim=0)  # [2B, D]
        
        # 计算logits
        logits = similarity_matrix / self.temperature
        
        # 标签：每个样本的正对位置
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels])  # [2B]
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        return loss
