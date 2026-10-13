"""
VAE (Variational Autoencoder) 基线实现

经典的变分自编码器异常检测方法：
1. 在正常样本上训练VAE
2. 重构误差 + KL散度作为异常分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class VAE(nn.Module):
    """变分自编码器"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 128, input_size: int = 256):
        super().__init__()
        
        # 计算经过4次下采样后的特征图尺寸
        feat_size = input_size // 16  # 256->16, 224->14
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Latent space - 动态计算维度
        self.feat_size = feat_size
        self.fc_mu = nn.Linear(256 * feat_size * feat_size, latent_dim)
        self.fc_logvar = nn.Linear(256 * feat_size * feat_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * feat_size * feat_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, self.feat_size, self.feat_size)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class VAEModel:
    """VAE异常检测模型"""
    
    def __init__(
        self,
        latent_dim: int = 128,
        lr: float = 1e-3,
        beta: float = 1.0,  # KL散度权重
        input_size: int = 256,  # 与run_baseline.py保持一致
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.model = VAE(latent_dim=latent_dim, input_size=input_size).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=1e-5
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 与其他基线保持一致
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
        ])
        
    def vae_loss(self, recon_x, x, mu, logvar):
        """VAE损失函数 = 重构损失 + β * KL散度"""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss
    
    def train(self, train_loader, epochs: int = 50):
        """训练VAE"""
        print(f"VAE: 开始训练 ({epochs} epochs)...")
        
        self.model.train()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            num_batches = 0
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(self.device)
                
                # Forward
                recon, mu, logvar = self.model(images)
                loss = self.vae_loss(recon, images, mu, logvar)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / (num_batches * len(images))
            print(f"  Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f}")
        
        print("VAE: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算异常分数"""
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # 重构
            recon, mu, logvar = self.model(images)
            
            # 异常分数 = 重构误差
            recon_error = torch.mean((images - recon) ** 2, dim=[1, 2, 3])
            
            # 也可以加入KL散度
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # 综合异常分数
            anomaly_scores = recon_error + 0.1 * kl_divergence
            
            # 生成逐像素异常图
            anomaly_map = torch.mean((images - recon) ** 2, dim=1, keepdim=True)
        
        return anomaly_map.cpu(), anomaly_scores.cpu()
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """评估模型"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_scores = []
        all_labels = []
        
        print("VAE: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"VAE Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'latent_dim': self.latent_dim,
        }, save_path)
        print(f"VAE模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.latent_dim = checkpoint['latent_dim']
        print(f"VAE模型已从 {load_path} 加载")
