"""
PaDiM (Patch Distribution Modeling) 基线实现

参考: "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization"
https://arxiv.org/abs/2011.08785
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import pickle
from pathlib import Path
from typing import List, Tuple, Dict


class PaDiMModel:
    """
    PaDiM异常检测模型
    
    核心思想:
    1. 使用预训练CNN提取多层特征
    2. 对每个空间位置的特征向量拟合多元高斯分布
    3. 测试时计算马氏距离作为异常分数
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        layers: List[str] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            backbone_name: 骨干网络名称
            layers: 用于提取特征的层名列表
            device: 计算设备
        """
        self.device = torch.device(device)
        self.backbone_name = backbone_name
        
        # 默认使用layer1, layer2, layer3
        if layers is None:
            if 'resnet' in backbone_name:
                self.layers = ['layer1', 'layer2', 'layer3']
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
        else:
            self.layers = layers
        
        # 构建特征提取器
        self.feature_extractor = self._build_feature_extractor()
        
        # 高斯分布参数（均值和协方差）
        self.gaussian_params = {}  # {layer_name: {'mean': tensor, 'cov_inv': tensor}}
        
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
        
        # 注册hook来提取中间层特征（在移除层之前）
        self.features = {}
        self.hooks = []
        
        def get_features(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        # 为每一层注册hook
        for layer_name in self.layers:
            if hasattr(model, layer_name):
                layer_module = getattr(model, layer_name)
                hook = layer_module.register_forward_hook(get_features(layer_name))
                self.hooks.append(hook)
                print(f"  Registered hook for {layer_name}")
        
        model.eval()
        return model.to(self.device)
    
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取多层特征
        
        Args:
            images: 输入图像 [B, C, H, W]
        
        Returns:
            features: {layer_name: feature_tensor}
        """
        self.features = {}
        with torch.no_grad():
            _ = self.feature_extractor(images.to(self.device))
        
        return self.features
    
    def fit(self, train_loader, random_sample_size: int = 10000):
        """
        训练阶段：拟合多元高斯分布
        
        Args:
            train_loader: 正常样本数据加载器
            random_sample_size: 随机采样的最大样本数（节省内存）
        """
        print("PaDiM: 开始拟合高斯分布...")
        
        # 收集所有正常样本的特征
        all_features = {layer: [] for layer in self.layers}
        
        for images, _ in tqdm(train_loader, desc="Extracting features"):
            features = self.extract_features(images)
            
            for layer_name in self.layers:
                feat = features[layer_name]  # [B, C, H, W]
                B, C, H, W = feat.shape
                
                # 重塑为 [B*H*W, C]
                feat = feat.permute(0, 2, 3, 1).reshape(-1, C)
                all_features[layer_name].append(feat.cpu())
        
        # 拼接所有特征
        for layer_name in self.layers:
            all_features[layer_name] = torch.cat(all_features[layer_name], dim=0)
            
            # 随机采样（如果太多）
            if all_features[layer_name].shape[0] > random_sample_size:
                indices = torch.randperm(all_features[layer_name].shape[0])[:random_sample_size]
                all_features[layer_name] = all_features[layer_name][indices]
            
            print(f"  {layer_name}: {all_features[layer_name].shape}")
        
        # 对每个空间位置拟合高斯分布
        print("PaDiM: 计算均值和协方差...")
        self.gaussian_params = {}
        
        for layer_name in self.layers:
            features = all_features[layer_name]  # [N, C]
            C = features.shape[1]
            
            # 计算均值
            mean = features.mean(dim=0)  # [C]
            
            # 计算协方差矩阵（添加正则化项保证可逆）
            features_centered = features - mean  # [N, C]
            cov = torch.mm(features_centered.t(), features_centered) / (features.shape[0] - 1)
            
            # 添加小的正则化项保证数值稳定性
            reg = 0.01 * torch.eye(C)
            cov = cov + reg
            
            # 计算协方差的逆（使用伪逆避免奇异矩阵问题）
            try:
                cov_inv = torch.inverse(cov)
            except RuntimeError:
                print(f"  Warning: Using pseudo-inverse for {layer_name}")
                cov_inv = torch.pinverse(cov)
            
            self.gaussian_params[layer_name] = {
                'mean': mean.to(self.device),
                'cov_inv': cov_inv.to(self.device)
            }
            
            print(f"  {layer_name}: mean shape={mean.shape}, cov_inv shape={cov_inv.shape}")
        
        print("PaDiM: 训练完成！")
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算异常分数
        
        Args:
            images: 输入图像 [B, C, H, W]
        
        Returns:
            anomaly_maps: 异常图 [B, 1, H, W]
            anomaly_scores: 全局异常分数 [B]
        """
        self.feature_extractor.eval()
        
        # 提取特征
        features = self.extract_features(images)
        
        B = images.shape[0]
        anomaly_maps_list = []
        
        for layer_name in self.layers:
            feat = features[layer_name]  # [B, C, H, W]
            B, C, H, W = feat.shape
            
            mean = self.gaussian_params[layer_name]['mean']  # [C]
            cov_inv = self.gaussian_params[layer_name]['cov_inv']  # [C, C]
            
            # 重塑特征
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            
            # 计算马氏距离
            diff = feat_flat - mean.unsqueeze(0)  # [B*H*W, C]
            
            # 马氏距离: d^2 = (x-μ)^T Σ^{-1} (x-μ)
            left = torch.mm(diff, cov_inv)  # [B*H*W, C]
            mahalanobis_dist = torch.sum(left * diff, dim=1)  # [B*H*W]
            
            # 重塑回空间维度
            mahalanobis_dist = mahalanobis_dist.reshape(B, H, W)  # [B, H, W]
            
            # 上采样到原始尺寸
            anomaly_map = F.interpolate(
                mahalanobis_dist.unsqueeze(1),  # [B, 1, H, W]
                size=(images.shape[2], images.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            
            anomaly_maps_list.append(anomaly_map)
        
        # 融合多层异常图（取平均或最大）
        anomaly_maps = torch.stack(anomaly_maps_list, dim=0).mean(dim=0)  # [B, 1, H, W]
        
        # 全局异常分数（取最大值或平均值）
        anomaly_scores = anomaly_maps.view(B, -1).max(dim=1)[0]  # [B]
        
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
        
        print("PaDiM: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # 计算指标
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"PaDiM Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        """保存模型"""
        save_data = {
            'gaussian_params': {},
            'layers': self.layers,
            'backbone_name': self.backbone_name
        }
        
        # 将tensor转换为numpy以便保存
        for layer in self.gaussian_params:
            save_data['gaussian_params'][layer] = {
                'mean': self.gaussian_params[layer]['mean'].cpu().numpy(),
                'cov_inv': self.gaussian_params[layer]['cov_inv'].cpu().numpy()
            }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"PaDiM模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        """加载模型"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.layers = save_data['layers']
        self.backbone_name = save_data['backbone_name']
        
        # 将numpy转换回tensor
        self.gaussian_params = {}
        for layer in save_data['gaussian_params']:
            self.gaussian_params[layer] = {
                'mean': torch.tensor(save_data['gaussian_params'][layer]['mean']).to(self.device),
                'cov_inv': torch.tensor(save_data['gaussian_params'][layer]['cov_inv']).to(self.device)
            }
        
        print(f"PaDiM模型已从 {load_path} 加载")
