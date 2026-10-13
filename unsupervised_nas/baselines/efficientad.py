"""
EfficientAD 基线实现 (2024)

参考: "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies"
https://arxiv.org/abs/2303.14535 (WACV 2024)

核心思想:
1. 使用预训练Teacher提取特征
2. 训练Student模仿Teacher
3. 添加Autoencoder进一步约束
4. 组合多种异常信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple


class EfficientADStudent(nn.Module):
    """EfficientAD Student网络"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class EfficientADModel:
    """
    EfficientAD异常检测模型
    
    结合了:
    1. Teacher-Student特征匹配
    2. Autoencoder重构
    3. 多尺度异常检测
    """
    
    def __init__(
        self,
        teacher_backbone: str = 'resnet18',
        lr: float = 1e-3,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.teacher_backbone = teacher_backbone
        
        # Teacher网络（冻结）
        self.teacher = self._build_teacher()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Student网络
        self.student = EfficientADStudent().to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), 
            lr=lr, 
            weight_decay=1e-5
        )
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def _build_teacher(self) -> nn.Module:
        """构建Teacher网络"""
        if 'resnet18' in self.teacher_backbone:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif 'resnet34' in self.teacher_backbone:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported teacher backbone: {self.teacher_backbone}")
        
        # 移除分类头
        modules = list(model.children())[:-2]
        model = nn.Sequential(*modules)
        model.eval()
        
        return model.to(self.device)
    
    def train(self, train_loader, epochs: int = 50):
        """训练Student网络"""
        print(f"EfficientAD: 开始训练 ({epochs} epochs)...")
        
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            num_batches = 0
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(self.device)
                
                # Teacher特征（最后一层）
                with torch.no_grad():
                    teacher_feat = self.teacher(images)
                
                # Student输出
                reconstructed, student_feat = self.student(images)
                
                # 损失1: 重构损失
                recon_loss = F.mse_loss(reconstructed, images)
                
                # 损失2: 特征匹配损失
                # 调整student特征尺寸以匹配teacher
                if student_feat.shape[2:] != teacher_feat.shape[2:]:
                    student_feat_resized = F.interpolate(
                        student_feat,
                        size=teacher_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    student_feat_resized = student_feat
                
                # 如果通道数不同，调整通道
                if student_feat_resized.shape[1] != teacher_feat.shape[1]:
                    adapter = nn.Conv2d(student_feat_resized.shape[1], teacher_feat.shape[1], kernel_size=1).to(self.device)
                    student_feat_resized = adapter(student_feat_resized)
                
                feat_loss = F.mse_loss(student_feat_resized, teacher_feat)
                
                # 总损失
                loss = recon_loss + feat_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f}")
        
        print("EfficientAD: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算异常分数"""
        self.teacher.eval()
        self.student.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # Teacher特征
            teacher_feat = self.teacher(images)
            
            # Student输出
            reconstructed, student_feat = self.student(images)
            
            # 异常信号1: 重构误差
            recon_error = torch.mean((images - reconstructed) ** 2, dim=1, keepdim=True)
            
            # 异常信号2: 特征差异
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                student_feat_resized = F.interpolate(
                    student_feat,
                    size=teacher_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                student_feat_resized = student_feat
            
            # 如果通道数不同，调整通道
            if student_feat_resized.shape[1] != teacher_feat.shape[1]:
                adapter = nn.Conv2d(student_feat_resized.shape[1], teacher_feat.shape[1], kernel_size=1).to(self.device)
                student_feat_resized = adapter(student_feat_resized)
            
            feat_diff = torch.mean((teacher_feat - student_feat_resized) ** 2, dim=1, keepdim=True)
            
            # 上采样到原始尺寸
            recon_error_up = F.interpolate(
                recon_error,
                size=(images.shape[2], images.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            
            feat_diff_up = F.interpolate(
                feat_diff,
                size=(images.shape[2], images.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            
            # 组合异常图
            anomaly_map = recon_error_up + feat_diff_up
            
            # 全局异常分数
            anomaly_scores = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)
        
        return anomaly_map.cpu(), anomaly_scores.cpu()
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """评估模型性能"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_scores = []
        all_labels = []
        
        print("EfficientAD: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"EfficientAD Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        """保存模型"""
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'teacher_backbone': self.teacher_backbone,
        }, save_path)
        print(f"EfficientAD模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher_backbone = checkpoint['teacher_backbone']
        print(f"EfficientAD模型已从 {load_path} 加载")
