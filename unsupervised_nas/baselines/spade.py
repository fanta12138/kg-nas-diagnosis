"""
SPADE (Spanning and Pruning) 基线实现

参考: "Sub-Image Anomaly Detection with Deep Pyramid Correspondences"
https://arxiv.org/abs/2005.02357
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
import faiss  # Facebook AI Similarity Search


class SPADEModel:
    """
    SPADE异常检测模型
    
    核心思想:
    1. 使用预训练CNN提取特征
    2. 存储所有正常样本的特征作为记忆库
    3. 测试时用k-NN查找最近邻
    4. 距离作为异常分数
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        layer: str = 'layer3',
        k: int = 5,
        device: str = 'cuda'
    ):
        """
        Args:
            backbone_name: 骨干网络名称
            layer: 用于提取特征的层
            k: k-NN的k值
            device: 计算设备
        """
        self.device = torch.device(device)
        self.backbone_name = backbone_name
        self.layer = layer
        self.k = k
        
        # 构建特征提取器
        self.feature_extractor = self._build_feature_extractor()
        
        # 记忆库（正常样本特征）
        self.memory_bank = None
        self.memory_labels = None
        
        # FAISS索引（用于高效k-NN搜索）
        self.index = None
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def _build_feature_extractor(self) -> nn.Module:
        """构建特征提取器"""
        if 'resnet18' in self.backbone_name:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif 'resnet34' in self.backbone_name:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif 'resnet50' in self.backbone_name:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        # 注册hook提取指定层特征
        self.features = {}
        
        def get_features(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        if hasattr(model, self.layer):
            layer_module = getattr(model, self.layer)
            layer_module.register_forward_hook(get_features(self.layer))
        
        model.eval()
        return model.to(self.device)
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取特征
        
        Args:
            images: 输入图像 [B, C, H, W]
        
        Returns:
            features: 特征向量 [B, C]
        """
        self.features = {}
        with torch.no_grad():
            _ = self.feature_extractor(images.to(self.device))
        
        feat = self.features[self.layer]  # [B, C, H, W]
        
        # 全局平均池化
        feat = F.adaptive_avg_pool2d(feat, (1, 1))  # [B, C, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, C]
        
        # L2归一化
        feat = F.normalize(feat, p=2, dim=1)
        
        return feat
    
    def build_memory_bank(self, train_loader):
        """
        构建记忆库
        
        Args:
            train_loader: 正常样本数据加载器
        """
        print("SPADE: 构建记忆库...")
        
        all_features = []
        
        for images, _ in tqdm(train_loader, desc="Extracting normal features"):
            features = self.extract_features(images)
            all_features.append(features.cpu())
        
        # 拼接所有特征
        self.memory_bank = torch.cat(all_features, dim=0).numpy()  # [N, C]
        
        print(f"  记忆库大小: {self.memory_bank.shape}")
        
        # 构建FAISS索引（使用L2距离）
        dimension = self.memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.memory_bank)
        
        print(f"  FAISS索引构建完成")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算异常分数
        
        Args:
            images: 输入图像 [B, C, H, W]
        
        Returns:
            anomaly_maps: 异常图（这里简化为全局分数）
            anomaly_scores: 全局异常分数 [B]
        """
        self.feature_extractor.eval()
        
        # 提取特征
        features = self.extract_features(images)
        features_np = features.cpu().numpy()
        
        # k-NN搜索
        distances, indices = self.index.search(features_np, self.k)
        
        # 异常分数：到k个最近邻的平均距离
        anomaly_scores = torch.tensor(distances.mean(axis=1), dtype=torch.float32)
        
        # 简化的异常图（实际SPADE会在多个尺度上计算）
        B = images.shape[0]
        anomaly_maps = anomaly_scores.view(B, 1, 1, 1).expand(-1, 1, images.shape[2], images.shape[3])
        
        return anomaly_maps, anomaly_scores
    
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
        
        print("SPADE: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # 计算指标
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"SPADE Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        """保存模型"""
        import pickle
        
        save_data = {
            'memory_bank': self.memory_bank,
            'layer': self.layer,
            'backbone_name': self.backbone_name,
            'k': self.k
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"SPADE模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        """加载模型"""
        import pickle
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.memory_bank = save_data['memory_bank']
        self.layer = save_data['layer']
        self.backbone_name = save_data['backbone_name']
        self.k = save_data['k']
        
        # 重建FAISS索引
        dimension = self.memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.memory_bank)
        
        print(f"SPADE模型已从 {load_path} 加载")
