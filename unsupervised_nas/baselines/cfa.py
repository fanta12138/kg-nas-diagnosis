"""
CFA (Coupled Hypersphere-based Feature Adaptation) 基线实现

参考: "CFA: Coupled Hypersphere-based Feature Adaptation for Unsupervised Anomaly Localization"
https://arxiv.org/abs/2206.04325

核心思想:
1. 使用预训练模型提取特征
2. 学习一个超球面边界包围正常样本
3. 到超球面中心的距离作为异常分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple


class CFAModel:
    """
    CFA异常检测模型
    
    简化版实现：
    1. 使用预训练ResNet18提取特征
    2. 学习正常样本的特征中心
    3. 马氏距离作为异常分数
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        
        # 骨干网络（冻结）
        self.backbone = self._build_backbone(backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 特征统计量
        self.feature_mean = None
        self.feature_cov_inv = None
        
        # 变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def _build_backbone(self, backbone_name: str) -> nn.Module:
        """构建骨干网络"""
        if 'resnet18' in backbone_name:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif 'resnet34' in backbone_name:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # 移除分类头
        modules = list(model.children())[:-2]
        model = nn.Sequential(*modules)
        model.eval()
        
        return model.to(self.device)
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取全局平均池化后的特征"""
        with torch.no_grad():
            features = self.backbone(images)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        return features
    
    def train(self, train_loader):
        """训练：计算正常样本的特征统计量"""
        print("CFA: 提取正常样本特征...")
        
        self.backbone.eval()
        
        all_features = []
        
        for images, _ in tqdm(train_loader, desc="Extracting features"):
            images = images.to(self.device)
            features = self.extract_features(images)
            all_features.append(features.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        
        # 计算均值和协方差
        print("CFA: 计算特征统计量...")
        all_features = all_features.to(self.device)
        self.feature_mean = all_features.mean(dim=0)
        
        # 计算协方差矩阵的逆（添加正则化）
        centered = all_features - self.feature_mean
        cov = torch.mm(centered.T, centered) / (all_features.size(0) - 1)
        # 添加小的对角项以保证可逆
        reg = torch.eye(cov.size(0), device=self.device) * 1e-6
        self.feature_cov_inv = torch.inverse(cov + reg)
        
        print("CFA: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算异常分数（马氏距离）"""
        self.backbone.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # 提取特征
            features = self.extract_features(images)
            
            # 计算马氏距离
            diff = features - self.feature_mean
            left = torch.mm(diff, self.feature_cov_inv)
            mahalanobis_dist = torch.sqrt(torch.sum(left * diff, dim=1))
            
            anomaly_scores = mahalanobis_dist
            
            # 生成伪异常图
            batch_size = images.size(0)
            anomaly_map = anomaly_scores.view(batch_size, 1, 1, 1).expand(
                batch_size, 1, images.size(2), images.size(3)
            )
        
        return anomaly_map.cpu(), anomaly_scores.cpu()
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """评估模型"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_scores = []
        all_labels = []
        
        print("CFA: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"CFA Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        torch.save({
            'feature_mean': self.feature_mean.cpu(),
            'feature_cov_inv': self.feature_cov_inv.cpu(),
        }, save_path)
        print(f"CFA模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.feature_mean = checkpoint['feature_mean'].to(self.device)
        self.feature_cov_inv = checkpoint['feature_cov_inv'].to(self.device)
        print(f"CFA模型已从 {load_path} 加载")
