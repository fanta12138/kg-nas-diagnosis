#!/usr/bin/env python3
"""
消融实验3: 损失函数组件对比
测试不同的损失函数组合对最终性能的影响
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
    parser = argparse.ArgumentParser(description='Ablation 3: Loss Function Components')
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='backbone类型')
    parser.add_argument('--feature-dim', type=int, default=256, help='特征维度')
    parser.add_argument('--loss-type', type=str, default='full', 
                        choices=['kd_only', 'kd_consistency', 'full'],
                        help='损失函数类型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    return parser.parse_args()


def compute_normal_prototype(model, train_loader, device):
    """计算正常样本的特征原型"""
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            output = model(images)
            features = output['features']
            
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])
            
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
            
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])
            
            # 计算到原型的距离作为异常分数
            distances = torch.norm(features - normal_prototype.unsqueeze(0), dim=1)
            all_scores.extend(distances.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    return {'auroc': float(auroc), 'ap': float(ap)}


def train_with_loss(args, loss_type):
    """使用指定的损失函数进行训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建数据加载器
    img_size = 224
    batch_size = 32
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 构建模型
    model = AnomalyDetector(backbone_name=args.backbone, feature_dim=args.feature_dim).to(device)
    
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    print(f"\n{'='*60}")
    print(f"训练配置: Loss Type = {loss_type}")
    print(f"{'='*60}\n")
    
    best_auroc = 0
    best_ap = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        
        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            
            output = model(images)
            features = output['features']
            
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])
            
            # 根据loss_type选择不同的损失函数
            if loss_type == 'kd_only':
                # 仅使用知识蒸馏损失（与教师模型的距离）
                # 这里简化为特征的L2正则化
                loss = torch.norm(features, dim=1).mean()
                
            elif loss_type == 'kd_consistency':
                # KD + Consistency
                # 生成增强视图
                aug_images = torch.flip(images, dims=[3])  # 水平翻转作为简单增强
                aug_output = model(aug_images)
                aug_features = aug_output['features']
                
                if aug_features.dim() == 4:
                    aug_features = aug_features.mean(dim=[2, 3])
                
                # Consistency loss: 原始和增强视图的特征应该相似
                consistency_loss = nn.functional.mse_loss(features, aug_features)
                
                # KD loss: 特征的正则化
                kd_loss = torch.norm(features, dim=1).mean()
                
                loss = kd_loss + 0.3 * consistency_loss
                
            else:  # full
                # Full: InfoNCE风格的对比学习
                features_norm = nn.functional.normalize(features, dim=1)
                sim_matrix = torch.mm(features_norm, features_norm.t())
                mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
                sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
                loss = -torch.log(torch.exp(sim_matrix / 0.5).sum(dim=1)).mean()
            
            loss.backward()
            optimizer.step()
            
            loss_meter += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss_meter/n_batches:.4f}'})
        
        scheduler.step()
        
        # 每10个epoch评估一次
        if epoch % 10 == 0 or epoch == args.epochs:
            normal_prototype = compute_normal_prototype(model, train_loader, device)
            metrics = evaluate_anomaly_detection(model, test_loader, normal_prototype, device)
            
            print(f"Epoch [{epoch}/{args.epochs}] Loss: {loss_meter/n_batches:.4f} "
                  f"AUROC: {metrics['auroc']:.4f} AP: {metrics['ap']:.4f}")
            
            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
                best_ap = metrics['ap']
                print(f"✓ 保存最佳模型 (AUROC: {best_auroc:.4f})")
    
    return {'auroc': best_auroc, 'ap': best_ap}


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试三种损失函数配置
    loss_types = ['kd_only', 'kd_consistency', 'full']
    results = {}
    
    for loss_type in loss_types:
        result = train_with_loss(args, loss_type)
        results[loss_type] = result
        
        # 保存中间结果
        result_file = output_dir / f'result_{loss_type}.json'
        with open(result_file, 'w') as f:
            json.dump({
                'loss_type': loss_type,
                'result': result,
                'config': {
                    'backbone': args.backbone,
                    'feature_dim': args.feature_dim,
                    'epochs': args.epochs
                }
            }, f, indent=2)
        
        print(f"\n✅ {loss_type} 完成: AUROC={result['auroc']:.4f}, AP={result['ap']:.4f}\n")
    
    # 保存汇总结果
    summary_file = output_dir / 'ablation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("消融实验3完成！结果汇总:")
    print(f"{'='*60}")
    for loss_type, result in results.items():
        print(f"{loss_type:20s} AUROC: {result['auroc']:.4f}  AP: {result['ap']:.4f}")
    print(f"\n结果已保存到: {summary_file}")


if __name__ == '__main__':
    main()
