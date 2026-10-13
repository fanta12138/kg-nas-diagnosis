"""
AutoEncoder 基线实现

经典的自编码器异常检测方法：
1. 在正常样本上训练AE
2. 重构误差作为异常分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import numpy as np


class SimpleAutoEncoder(nn.Module):
    """简单的卷积自编码器"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # [B, 3, 224, 224] -> [B, 32, 112, 112]
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # [B, 32, 112, 112] -> [B, 64, 56, 56]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # [B, 64, 56, 56] -> [B, 128, 28, 28]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # [B, 128, 28, 28] -> [B, 256, 14, 14]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # [B, 256, 14, 14] -> [B, latent_dim, 7, 7]
            nn.Conv2d(256, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # [B, latent_dim, 7, 7] -> [B, 256, 14, 14]
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # [B, 256, 14, 14] -> [B, 128, 28, 28]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # [B, 128, 28, 28] -> [B, 64, 56, 56]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # [B, 64, 56, 56] -> [B, 32, 112, 112]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # [B, 32, 112, 112] -> [B, 3, 224, 224]
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # 输出限制在[0, 1]
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class AutoEncoderModel:
    """
    AutoEncoder异常检测模型
    
    核心思想:
    1. 在正常样本上训练自编码器
    2. 测试时计算重构误差
    3. 重构误差大的样本为异常
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        lr: float = 1e-3,
        device: str = 'cuda'
    ):
        """
        Args:
            latent_dim: 潜在空间维度
            lr: 学习率
            device: 计算设备
        """
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        
        # 创建模型
        self.model = SimpleAutoEncoder(latent_dim=latent_dim).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 自动归一化到[0, 1]
        ])
        
    def train(self, train_loader, epochs: int = 50):
        """
        训练自编码器
        
        Args:
            train_loader: 正常样本数据加载器
            epochs: 训练轮数
        """
        print(f"AutoEncoder: 开始训练 ({epochs} epochs)...")
        
        self.model.train()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            num_batches = 0
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(self.device)
                
                # 前向传播
                reconstructed = self.model(images)
                
                # 计算重构损失（MSE）
                loss = F.mse_loss(reconstructed, images)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f}")
        
        print("AutoEncoder: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> tuple:
        """
        计算异常分数
        
        Args:
            images: 输入图像 [B, C, H, W]
        
        Returns:
            anomaly_maps: 逐像素重构误差 [B, 1, H, W]
            anomaly_scores: 全局异常分数 [B]
        """
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            reconstructed = self.model(images)
            
            # 逐像素重构误差
            per_pixel_error = torch.mean((images - reconstructed) ** 2, dim=1, keepdim=True)
            
            # 全局异常分数（平均重构误差）
            anomaly_scores = per_pixel_error.view(per_pixel_error.size(0), -1).mean(dim=1)
        
        return per_pixel_error.cpu(), anomaly_scores.cpu()
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            metrics: {'auroc': float, 'ap': float}
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_scores = []
        all_labels = []
        
        print("AutoEncoder: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # 计算指标
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"AutoEncoder Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'latent_dim': self.latent_dim,
        }, save_path)
        print(f"AutoEncoder模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.latent_dim = checkpoint['latent_dim']
        print(f"AutoEncoder模型已从 {load_path} 加载")
