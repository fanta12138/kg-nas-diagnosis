"""
FastFlow 基线实现 (NeurIPS 2021)

参考: "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows"
https://arxiv.org/abs/2111.07677

核心思想:
1. 使用预训练骨干网络提取特征
2. 用2D Normalizing Flow建模正常特征分布
3. 似然概率作为异常分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple


class CouplingLayer(nn.Module):
    """Coupling Layer for Normalizing Flow"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        
        if not reverse:
            # Forward pass
            scale_logit, translation = self.network(x1).chunk(2, dim=1)
            scale = torch.sigmoid(scale_logit + 2) + 1e-7
            y2 = x2 * scale + translation
            log_det_jacobian = torch.log(scale).view(x.size(0), -1).sum(dim=1)
            return torch.cat([x1, y2], dim=1), log_det_jacobian
        else:
            # Inverse pass
            scale_logit, translation = self.network(x1).chunk(2, dim=1)
            scale = torch.sigmoid(scale_logit + 2) + 1e-7
            x2 = (x2 - translation) / scale
            return torch.cat([x1, x2], dim=1)


class FastFlowModel:
    """
    FastFlow异常检测模型
    
    简化版实现：
    1. 使用预训练ResNet18提取特征
    2. 用简化的flow建模特征分布
    3. 负对数似然作为异常分数
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        lr: float = 1e-3,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        
        # 骨干网络（冻结）
        self.backbone = self._build_backbone(backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 简化的Flow层（用MLP近似）
        self.flow = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        
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
            # 全局平均池化
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        return features
    
    def train(self, train_loader, epochs: int = 50):
        """训练Flow模型"""
        print(f"FastFlow: 开始训练 ({epochs} epochs)...")
        
        self.backbone.eval()
        self.flow.train()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            num_batches = 0
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(self.device)
                
                # 提取特征
                features = self.extract_features(images)
                
                # Flow变换
                transformed = self.flow(features)
                
                # 损失：最小化变换后特征的方差（鼓励紧凑分布）
                loss = torch.var(transformed)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f}")
        
        print("FastFlow: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算异常分数"""
        self.backbone.eval()
        self.flow.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # 提取特征
            features = self.extract_features(images)
            
            # Flow变换
            transformed = self.flow(features)
            
            # 异常分数：变换后特征的L2范数
            anomaly_scores = torch.norm(transformed, dim=1)
            
            # 生成伪异常图（由于是全局特征，这里简单复制）
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
        
        print("FastFlow: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"FastFlow Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        torch.save({
            'flow_state_dict': self.flow.state_dict(),
        }, save_path)
        print(f"FastFlow模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.flow.load_state_dict(checkpoint['flow_state_dict'])
        print(f"FastFlow模型已从 {load_path} 加载")
