"""
Reverse Distillation 基线实现 (CVPR 2022)

参考: "Anomaly Detection via Reverse Distillation from One-Class Embedding"
https://arxiv.org/abs/2201.10703

核心思想:
1. Teacher和Student使用相同架构
2. Student从低层到高层反向蒸馏Teacher的特征
3. 特征差异作为异常分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List


class ReverseDistillationModel:
    """
    Reverse Distillation异常检测模型
    
    简化版实现：
    1. Teacher和Student都是ResNet18
    2. 在多个尺度上匹配特征
    3. MSE损失训练Student
    4. 特征差异作为异常分数
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        lr: float = 1e-3,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        
        # Teacher和Student（相同架构）
        self.teacher = self._build_backbone(backbone)
        self.student = self._build_backbone(backbone)
        
        # 冻结Teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        
        # Hook存储特征
        self.teacher_features = {}
        self.student_features = {}
        
        # 注册hook
        self._register_hooks()
        
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
        
        return model.to(self.device)
    
    def _register_hooks(self):
        """注册hook提取中间层特征"""
        layers = ['layer1', 'layer2', 'layer3']
        
        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output.detach()
            return hook
        
        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook
        
        for name, module in self.teacher.named_modules():
            if name in layers:
                module.register_forward_hook(get_teacher_hook(name))
        
        for name, module in self.student.named_modules():
            if name in layers:
                module.register_forward_hook(get_student_hook(name))
    
    def train(self, train_loader, epochs: int = 50):
        """训练Student网络"""
        print(f"Reverse Distillation: 开始训练 ({epochs} epochs)...")
        
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            num_batches = 0
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(self.device)
                
                # 清空特征缓存
                self.teacher_features.clear()
                self.student_features.clear()
                
                # Forward
                with torch.no_grad():
                    _ = self.teacher(images)
                
                _ = self.student(images)
                
                # 计算多层特征匹配损失
                loss = 0
                for layer_name in ['layer1', 'layer2', 'layer3']:
                    t_feat = self.teacher_features[layer_name]
                    s_feat = self.student_features[layer_name]
                    
                    # 调整尺寸
                    if s_feat.shape != t_feat.shape:
                        s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear')
                    
                    loss += F.mse_loss(s_feat, t_feat)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f}")
        
        print("Reverse Distillation: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算异常分数"""
        self.teacher.eval()
        self.student.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # 清空特征缓存
            self.teacher_features.clear()
            self.student_features.clear()
            
            # Forward
            _ = self.teacher(images)
            _ = self.student(images)
            
            # 计算多层特征差异
            anomaly_map = torch.zeros(images.size(0), 1, images.size(2), images.size(3)).to(self.device)
            
            for layer_name in ['layer1', 'layer2', 'layer3']:
                t_feat = self.teacher_features[layer_name]
                s_feat = self.student_features[layer_name]
                
                # 调整尺寸
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear')
                
                # 逐像素差异
                diff = torch.mean((t_feat - s_feat) ** 2, dim=1, keepdim=True)
                
                # 上采样到原始尺寸
                diff_up = F.interpolate(diff, size=(images.size(2), images.size(3)), mode='bilinear')
                
                anomaly_map += diff_up
            
            # 全局异常分数
            anomaly_scores = anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)
        
        return anomaly_map.cpu(), anomaly_scores.cpu()
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """评估模型"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_scores = []
        all_labels = []
        
        print("Reverse Distillation: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"Reverse Distillation Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        torch.save({
            'student_state_dict': self.student.state_dict(),
        }, save_path)
        print(f"Reverse Distillation模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        print(f"Reverse Distillation模型已从 {load_path} 加载")
