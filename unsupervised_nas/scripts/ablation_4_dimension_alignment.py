#!/usr/bin/env python3
"""
消融实验4: 动态维度对齐 vs 固定维度
验证动态维度对齐技术的有效性
"""

import os
import sys
import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import AnomalyDetector
from datasets import MVTecWoodDataset, MVTecTestDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation 4: Dynamic Alignment vs Fixed Dim')
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--teacher-dim', type=int, default=768, help='教师模型特征维度 (ConvNeXtV2-Tiny)')
    parser.add_argument('--epochs', type=int, default=50, help='快速训练轮数')
    return parser.parse_args()


class StudentModelFixed(nn.Module):
    """固定维度的学生模型（不使用动态对齐）"""
    
    def __init__(self, backbone_name, feature_dim, teacher_dim):
        super().__init__()
        self.backbone = AnomalyDetector(backbone_name=backbone_name, feature_dim=feature_dim)
        
        # 如果学生特征维度与教师不同，强制使用教师维度（可能导致信息损失）
        if feature_dim != teacher_dim:
            print(f"⚠️  警告: 学生维度({feature_dim}) != 教师维度({teacher_dim})")
            print(f"   固定维度模式: 将截断或填充到{teacher_dim}维")
        
        self.teacher_dim = teacher_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        output = self.backbone(x)
        features = output['features']
        
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        
        # 固定维度策略：强制匹配教师维度
        if features.shape[1] != self.teacher_dim:
            if features.shape[1] < self.teacher_dim:
                # 填充零
                padding = torch.zeros(features.shape[0], self.teacher_dim - features.shape[1], 
                                     device=features.device)
                features = torch.cat([features, padding], dim=1)
            else:
                # 截断
                features = features[:, :self.teacher_dim]
        
        return {'features': features}


class StudentModelDynamic(nn.Module):
    """动态维度对齐的学生模型"""
    
    def __init__(self, backbone_name, feature_dim, teacher_dim):
        super().__init__()
        self.backbone = AnomalyDetector(backbone_name=backbone_name, feature_dim=feature_dim)
        
        # 动态对齐层：线性投影
        # 注意：AnomalyDetector已经通过projector将特征压缩到feature_dim
        # 所以这里应该从feature_dim映射到teacher_dim
        if feature_dim != teacher_dim:
            self.alignment_layer = nn.Linear(feature_dim, teacher_dim)
            print(f"✓ 动态对齐: {feature_dim} -> {teacher_dim}")
        else:
            self.alignment_layer = nn.Identity()
            print(f"✓ 无需对齐: {feature_dim} == {teacher_dim}")
        
        self.teacher_dim = teacher_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        output = self.backbone(x)
        features = output['features']
        
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        
        # 动态对齐
        features_aligned = self.alignment_layer(features)
        
        return {'features': features_aligned}


def compute_normal_prototype(model, train_loader, device):
    """计算正常样本的特征原型"""
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            output = model(images)
            features = output['features']
            features_list.append(features.cpu())
    
    all_features = torch.cat(features_list, dim=0)
    prototype = all_features.mean(dim=0)
    return prototype


def evaluate_anomaly_detection(model, test_loader, normal_prototype, device):
    """评估异常检测性能"""
    model.eval()
    all_scores = []
    all_labels = []
    
    # 确保prototype在正确的设备上
    normal_prototype = normal_prototype.to(device)
    
    with torch.no_grad():
        for images, labels, masks in test_loader:
            images = images.to(device)
            output = model(images)
            features = output['features']
            
            distances = torch.norm(features - normal_prototype.unsqueeze(0), dim=1)
            all_scores.extend(distances.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    return {'auroc': float(auroc), 'ap': float(ap)}


def train_and_evaluate(args, alignment_type, backbones_to_test):
    """测试不同的对齐策略"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_size = 224
    batch_size = 16
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MVTecWoodDataset(
        root=args.data_dir, split='train', defect_type='good', transform=train_transform
    )
    
    test_dataset = MVTecTestDataset(root=args.data_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    results = {}
    
    for backbone_name in backbones_to_test:
        print(f"\n{'='*60}")
        print(f"测试 Backbone: {backbone_name} | 对齐方式: {alignment_type}")
        print(f"{'='*60}\n")
        
        # 构建模型
        if alignment_type == 'fixed':
            model = StudentModelFixed(backbone_name, feature_dim=256, teacher_dim=args.teacher_dim)
        else:  # dynamic
            model = StudentModelDynamic(backbone_name, feature_dim=256, teacher_dim=args.teacher_dim)
        
        model = model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # 快速训练
        best_auroc = 0
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_meter = 0
            n_batches = 0
            
            for images, _ in train_loader:
                images = images.to(device)
                optimizer.zero_grad()
                
                output = model(images)
                features = output['features']
                
                # 简单的L2正则化损失
                loss = torch.norm(features, dim=1).mean()
                
                loss.backward()
                optimizer.step()
                
                loss_meter += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            # 每10个epoch评估
            if epoch % 10 == 0 or epoch == args.epochs:
                normal_prototype = compute_normal_prototype(model, train_loader, device)
                metrics = evaluate_anomaly_detection(model, test_loader, normal_prototype, device)
                
                print(f"Epoch [{epoch}/{args.epochs}] Loss: {loss_meter/n_batches:.4f} "
                      f"AUROC: {metrics['auroc']:.4f}")
                
                if metrics['auroc'] > best_auroc:
                    best_auroc = metrics['auroc']
        
        results[backbone_name] = {'auroc': best_auroc}
        print(f"✅ {backbone_name}: Best AUROC = {best_auroc:.4f}")
    
    return results


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择代表性的backbones进行测试
    backbones_to_test = [
        'resnet18',           # 512维
        'efficientnet_b0',    # 1280维
        'vit_tiny_patch16_224',  # 192维
        'convnextv2_tiny'     # 768维 (与教师相同)
    ]
    
    # 测试两种对齐策略
    print("\n" + "="*60)
    print("消融实验4: 动态维度对齐 vs 固定维度")
    print("="*60)
    
    results_fixed = train_and_evaluate(args, 'fixed', backbones_to_test)
    results_dynamic = train_and_evaluate(args, 'dynamic', backbones_to_test)
    
    # 保存结果
    summary = {
        'fixed_alignment': results_fixed,
        'dynamic_alignment': results_dynamic,
        'teacher_dim': args.teacher_dim,
        'epochs': args.epochs
    }
    
    summary_file = output_dir / 'ablation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印对比结果
    print(f"\n{'='*60}")
    print("消融实验4完成！结果对比:")
    print(f"{'='*60}")
    print(f"{'Backbone':25s} {'Fixed':>10s} {'Dynamic':>10s} {'Improvement':>12s}")
    print("-"*60)
    
    for backbone in backbones_to_test:
        fixed_auroc = results_fixed[backbone]['auroc']
        dynamic_auroc = results_dynamic[backbone]['auroc']
        improvement = dynamic_auroc - fixed_auroc
        print(f"{backbone:25s} {fixed_auroc:10.4f} {dynamic_auroc:10.4f} {improvement:+12.4f}")
    
    print(f"\n结果已保存到: {summary_file}")


if __name__ == '__main__':
    main()
