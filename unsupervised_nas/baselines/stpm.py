"""
STPM (Student-Teacher Feature Pyramid Matching) 基线实现

参考: "Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection"
https://arxiv.org/abs/2103.04257 (ECCV 2022)

核心思想:
1. 使用预训练模型作为Teacher（冻结）
2. 训练一个Student网络模仿Teacher的特征
3. 只在正常样本上训练
4. 测试时，异常样本会导致Student和Teacher特征不匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple


class StudentNetwork(nn.Module):
    """Student网络 - 轻量级CNN"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.features(x)


class STPMModel:
    """
    STPM异常检测模型
    
    核心思想:
    1. Teacher: 预训练ResNet18（冻结）
    2. Student: 轻量级CNN（可训练）
    3. 在多个尺度上匹配Teacher和Student的特征
    4. 特征差异作为异常分数
    """
    
    def __init__(
        self,
        teacher_backbone: str = 'resnet18',
        lr: float = 1e-3,
        device: str = 'cuda'
    ):
        """
        Args:
            teacher_backbone: Teacher骨干网络
            lr: 学习率
            device: 计算设备
        """
        self.device = torch.device(device)
        self.teacher_backbone = teacher_backbone
        
        # 构建Teacher网络（冻结）
        self.teacher = self._build_teacher()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 构建Student网络
        self.student = StudentNetwork().to(self.device)
        
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
        
        # 移除分类头，保留特征提取部分
        modules = list(model.children())[:-2]  # 去掉avgpool和fc
        model = nn.Sequential(*modules)
        model.eval()
        
        return model.to(self.device)
    
    def extract_features(self, model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多层特征
        
        Returns:
            features: 不同尺度的特征列表
        """
        features = []
        
        # 手动前向传播以获取中间层特征
        if isinstance(model, nn.Sequential):
            for module in model:
                x = module(x)
                # 在MaxPool后保存特征
                if isinstance(module, nn.MaxPool2d):
                    features.append(x.clone())
        
        # 如果没有捕获到特征，返回最后一层
        if not features:
            features = [x]
        
        return features
    
    def train(self, train_loader, epochs: int = 50):
        """
        训练Student网络
        
        Args:
            train_loader: 正常样本数据加载器
            epochs: 训练轮数
        """
        print(f"STPM: 开始训练Student网络 ({epochs} epochs)...")
        
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            num_batches = 0
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(self.device)
                
                # 提取Teacher特征（多尺度）
                with torch.no_grad():
                    teacher_feats = self.extract_features(self.teacher, images)
                
                # 提取Student特征
                student_output = self.student(images)
                student_feats = [student_output]
                
                # 计算特征匹配损失（MSE）
                loss = 0
                for t_feat, s_feat in zip(teacher_feats[-1:], student_feats):
                    # 调整Student特征尺寸以匹配Teacher
                    if s_feat.shape[2:] != t_feat.shape[2:]:
                        s_feat_resized = F.interpolate(
                            s_feat, 
                            size=t_feat.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                    else:
                        s_feat_resized = s_feat
                    
                    # 如果通道数不同，使用1x1卷积调整
                    if s_feat_resized.shape[1] != t_feat.shape[1]:
                        # 创建临时的1x1卷积来调整通道数
                        adapter = nn.Conv2d(s_feat_resized.shape[1], t_feat.shape[1], kernel_size=1).to(self.device)
                        s_feat_resized = adapter(s_feat_resized)
                    
                    loss += F.mse_loss(s_feat_resized, t_feat)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f}")
        
        print("STPM: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算异常分数
        
        Args:
            images: 输入图像 [B, C, H, W]
        
        Returns:
            anomaly_maps: 逐像素异常图 [B, 1, H, W]
            anomaly_scores: 全局异常分数 [B]
        """
        self.teacher.eval()
        self.student.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # 提取Teacher特征
            teacher_feats = self.extract_features(self.teacher, images)
            
            # 提取Student特征
            student_output = self.student(images)
            student_feats = [student_output]
            
            # 计算特征差异
            anomaly_map = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3]).to(self.device)
            
            for t_feat, s_feat in zip(teacher_feats[-1:], student_feats):
                # 调整尺寸
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat_resized = F.interpolate(
                        s_feat, 
                        size=t_feat.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                else:
                    s_feat_resized = s_feat
                
                # 如果通道数不同，调整通道
                if s_feat_resized.shape[1] != t_feat.shape[1]:
                    adapter = nn.Conv2d(s_feat_resized.shape[1], t_feat.shape[1], kernel_size=1).to(self.device)
                    s_feat_resized = adapter(s_feat_resized)
                
                # 计算逐像素差异
                diff = torch.mean((t_feat - s_feat_resized) ** 2, dim=1, keepdim=True)
                
                # 上采样到原始尺寸
                diff_upsampled = F.interpolate(
                    diff, 
                    size=(images.shape[2], images.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                anomaly_map += diff_upsampled
            
            # 全局异常分数
            anomaly_scores = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)
        
        return anomaly_map.cpu(), anomaly_scores.cpu()
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """评估模型性能"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_scores = []
        all_labels = []
        
        print("STPM: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"STPM Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        """保存模型"""
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'teacher_backbone': self.teacher_backbone,
        }, save_path)
        print(f"STPM模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher_backbone = checkpoint['teacher_backbone']
        print(f"STPM模型已从 {load_path} 加载")
