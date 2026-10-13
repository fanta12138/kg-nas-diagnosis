"""
SimpleNet 基线实现 (CVPR 2023)

参考: "SimpleNet: A Simple Network for Image Anomaly Detection and Localization"
https://arxiv.org/abs/2305.16389

核心思想:
1. 使用预训练骨干网络提取多层特征
2. 全局特征和局部特征的简单融合
3. 通过特征差异计算异常分数
4. 架构简洁，计算效率高
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List


class SimpleNetModel(nn.Module):
    """
    SimpleNet异常检测模型
    
    简化版实现：
    1. 使用预训练ResNet提取特征
    2. 从layer2和layer3提取特征
    3. 上采样到相同尺寸后拼接
    4. 通过1x1卷积生成异常图
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        layers: List[str] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = torch.device(device)
        
        # 默认使用layer2和layer3
        if layers is None:
            self.layers = ['layer2', 'layer3']
        else:
            self.layers = layers
        
        # 骨干网络（冻结）
        self.backbone = self._build_backbone(backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 特征融合层
        # ResNet18: layer2=128, layer3=256 -> total=384
        # ResNet34: layer2=128, layer3=256 -> total=384
        # WideResNet50: layer2=512, layer3=1024 -> total=1536
        if 'wide_resnet' in backbone:
            input_channels = 1536
        else:
            input_channels = 384
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        ).to(self.device)
        
        # Hook存储特征
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
        if 'resnet18' in backbone_name:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif 'resnet34' in backbone_name:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif 'wide_resnet50' in backbone_name:
            model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return model.to(self.device)
    
    def _register_hooks(self):
        """注册hook提取中间层特征"""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        for name, module in self.backbone.named_modules():
            if name in self.layers:
                module.register_forward_hook(get_hook(name))
    
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取多层特征"""
        self.features.clear()
        
        with torch.no_grad():
            _ = self.backbone(images.to(self.device))
        
        return self.features
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播，生成异常图"""
        # 提取特征
        features = self.extract_features(images)
        
        # 获取指定层的特征
        feat_list = []
        target_size = None
        
        for layer_name in self.layers:
            feat = features[layer_name]
            
            # 记录目标尺寸（使用第一个特征图的尺寸）
            if target_size is None:
                target_size = feat.shape[2:]
            
            # 上采样到目标尺寸
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            feat_list.append(feat)
        
        # 拼接特征
        fused_feat = torch.cat(feat_list, dim=1)  # [B, C_total, H, W]
        
        # 生成异常图
        anomaly_map = self.fusion_conv(fused_feat)  # [B, 1, H, W]
        
        # Sigmoid归一化到[0, 1]
        anomaly_map = torch.sigmoid(anomaly_map)
        
        return anomaly_map
    
    def compute_anomaly_scores(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算异常分数"""
        self.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # 生成异常图
            anomaly_map = self.forward(images)
            
            # 全局异常分数（取最大值或平均值）
            anomaly_scores = anomaly_map.view(anomaly_map.size(0), -1).max(dim=1)[0]
        
        return anomaly_map.cpu(), anomaly_scores.cpu()
    
    def train_model(self, train_loader, epochs: int = 10):
        """
        训练融合层（可选）
        
        SimpleNet原版不需要训练，但我们可以微调融合层以适应特定数据集
        """
        print(f"SimpleNet: 微调融合层 ({epochs} epochs)...")
        
        # 只训练融合层
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.fusion_conv.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(self.fusion_conv.parameters(), lr=1e-3)
        
        self.train()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            num_batches = 0
            
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(self.device)
                
                # 前向传播
                anomaly_map = self.forward(images)
                
                # 对于正常样本，异常图应该接近0
                # 使用MSE损失
                target = torch.zeros_like(anomaly_map)
                loss = F.mse_loss(anomaly_map, target)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f}")
        
        print("SimpleNet: 训练完成！")
        
        # 重新冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """评估模型"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_scores = []
        all_labels = []
        
        print("SimpleNet: 开始评估...")
        
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            _, scores = self.compute_anomaly_scores(images)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        print(f"SimpleNet Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AP: {ap:.4f}")
        
        return {'auroc': auroc, 'ap': ap}
    
    def save(self, save_path: str):
        torch.save({
            'fusion_conv_state_dict': self.fusion_conv.state_dict(),
            'layers': self.layers,
        }, save_path)
        print(f"SimpleNet模型已保存到: {save_path}")
    
    def load(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.fusion_conv.load_state_dict(checkpoint['fusion_conv_state_dict'])
        self.layers = checkpoint['layers']
        print(f"SimpleNet模型已从 {load_path} 加载")
