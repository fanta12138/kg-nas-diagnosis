"""
PatchCore 基线实现 (CVPR 2022)

参考: "Towards Total Recall in Industrial Anomaly Detection"
https://arxiv.org/abs/2106.08265

核心思想:
1. 使用预训练模型提取多尺度特征
2. 构建正常样本的特征记忆库（Memory Bank）
3. 最近邻距离作为异常分数
4. 可选的coreset采样减少内存
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List
import random


class PatchCoreModel:
    """
    PatchCore异常检测模型
    
    简化版实现：
    1. 使用预训练WideResNet50提取特征
    2. 从多个层提取patch-level特征
    3. 构建记忆库
    4. 最近邻搜索计算异常分数
    """
    
    def __init__(
        self,
        backbone: str = 'wide_resnet50',
        layers: List[str] = None,
        coreset_ratio: float = 0.1,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.backbone_name = backbone
        self.coreset_ratio = coreset_ratio
        
        # 默认使用layer2和layer3
        if layers is None:
            self.layers = ['layer2', 'layer3']
        else:
            self.layers = layers
        
        # 骨干网络（冻结）
        self.backbone = self._build_backbone(backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 记忆库
        self.memory_bank = None
        self.feature_shapes = {}
        
        # Hook用于提取中间层特征
        self.features = {}
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
        if 'wide_resnet50' in backbone_name:
            model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif 'resnet18' in backbone_name:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return model.to(self.device)
    
    def _register_hooks(self):
        """注册hook以提取中间层特征"""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        # 根据骨干网络注册hook
        if 'wide_resnet' in self.backbone_name or 'resnet' in self.backbone_name:
            for name, module in self.backbone.named_modules():
                if name in self.layers:
                    module.register_forward_hook(get_hook(name))
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取多层patch-level特征并融合"""
        self.features.clear()
        
        with torch.no_grad():
            _ = self.backbone(images.to(self.device))
        
        # 收集指定层的特征
        patch_features = []
        target_h, target_w = None, None
        
        for layer_name in self.layers:
            if layer_name in self.features:
                feat = self.features[layer_name]
                B, C, H, W = feat.shape
                
                # 记录目标尺寸（使用第一个特征图的尺寸）
                if target_h is None:
                    target_h, target_w = H, W
                
                # 上采样到目标尺寸
                if H != target_h or W != target_w:
                    feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                # 展平为patch向量 [B, C, H, W] -> [B*H*W, C]
                feat = feat.permute(0, 2, 3, 1).reshape(-1, C)
                patch_features.append(feat)
                
                # 保存形状信息用于重建
                self.feature_shapes[layer_name] = (B, target_h, target_w, C)
        
        # 拼接所有层的特征
        if len(patch_features) > 1:
            combined = torch.cat(patch_features, dim=1)
        else:
            combined = patch_features[0]
        
        return combined
    
    def train(self, train_loader):
        """构建记忆库"""
        print("PatchCore: 提取正常样本特征...")
        
        self.backbone.eval()
        
        all_features = []
        
        for images, _ in tqdm(train_loader, desc="Building memory bank"):
            features = self.extract_features(images)
            all_features.append(features.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        print(f"  总特征数: {all_features.shape[0]}")
        
        # Coreset采样（随机采样简化版）
        if self.coreset_ratio < 1.0:
            n_samples = int(len(all_features) * self.coreset_ratio)
            indices = random.sample(range(len(all_features)), n_samples)
            self.memory_bank = all_features[indices].to(self.device)
            print(f"  Coreset采样后: {self.memory_bank.shape[0]}")
        else:
            self.memory_bank = all_features.to(self.device)
        
        print(f"  记忆库大小: {self.memory_bank.shape}")
        print("PatchCore: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算异常分数"""
        self.backbone.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # 提取测试样本特征
            test_features = self.extract_features(images)
            
            # 计算到记忆库中最近邻的距离
            # 使用余弦相似度或欧氏距离
            anomaly_maps = []
            anomaly_scores = []
            
            batch_size = images.size(0)
            for i in range(batch_size):
                # 获取当前样本的特征
                feat = test_features[i:i+1]  # [1, C]
                
                # 计算与记忆库中所有特征的距离
                # 使用余弦距离
                feat_norm = F.normalize(feat, p=2, dim=1)
                memory_norm = F.normalize(self.memory_bank, p=2, dim=1)
                
                # 余弦相似度
                similarity = torch.mm(feat_norm, memory_norm.T)  # [1, N]
                
                # 最大相似度作为正常度
                max_sim = similarity.max(dim=1)[0]  # [1]
                
                # 异常分数 = 1 - 最大相似度
                score = 1 - max_sim
                
                anomaly_scores.append(score.item())
            
            anomaly_scores = torch.tensor(anomaly_scores)
            
            # 生成伪异常图（简化版，实际应该逐patch计算）
            anomaly_map = anomaly_scores.view(batch_size, 1, 1, 1).expand(
                batch_size, 1, images.size(2), images.size(3)
            )
        
        return anomaly_map.cpu(), anomaly_scores.cpu()
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """评估模型"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_scores = []
        all_labels = []
        
        print("PatchCore: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"PatchCore Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        torch.save({
            'memory_bank': self.memory_bank,
            'feature_shapes': self.feature_shapes,
            'layers': self.layers,
            'coreset_ratio': self.coreset_ratio,
        }, save_path)
        print(f"PatchCore模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.memory_bank = checkpoint['memory_bank']
        self.feature_shapes = checkpoint['feature_shapes']
        self.layers = checkpoint['layers']
        self.coreset_ratio = checkpoint['coreset_ratio']
        print(f"PatchCore模型已从 {load_path} 加载")
