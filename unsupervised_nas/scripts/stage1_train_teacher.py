#!/usr/bin/env python3
"""
阶段1: 使用对比学习训练教师模型

这是两阶段无监督NAS的第一步：
在无标签的正常样本上训练一个强大的特征提取器
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import yaml
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ContrastiveModel, SimCLRLoss
from datasets.mvtec_loader import MVTecWoodDataset
from utils.logger import Logger, AverageMeter


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='阶段1: 对比学习预训练')
    parser.add_argument('--config', type=str, default='configs/stage1_contrastive.yaml',
                        help='配置文件路径')
    parser.add_argument('--data-dir', type=str, default='../datasets/MVTec-AD/wood',
                        help='数据集目录')
    parser.add_argument('--output-dir', type=str, default='experiments/stage1',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数（覆盖配置）')
    parser.add_argument('--batch-size', type=int, default=None, help='批大小（覆盖配置）')
    parser.add_argument('--lr', type=float, default=None, help='学习率（覆盖配置）')
    parser.add_argument('--backbone', type=str, default=None, help='骨干网络（覆盖配置）')
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_dataloaders(data_dir: str, batch_size: int, img_size: int, num_workers: int = 4):
    """构建数据加载器（只使用正常样本）"""
    
    # 强数据增强用于对比学习
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集（只使用train/good）
    train_dataset = MVTecWoodDataset(
        root=data_dir,
        split='train',
        defect_type='good',  # 只用正常样本
        transform=train_transform,
        return_pair=True  # 返回两个增强视图
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    
    return train_loader


def train_one_epoch(epoch, model, teacher_model, train_loader, criterion, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (view1, view2) in enumerate(pbar):
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        z_i = model(view1)
        z_j = model(view2)
        
        # 计算对比损失
        loss = criterion(z_i, z_j)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新指标
        loss_meter.update(loss.item(), view1.size(0))
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return {'loss': loss_meter.avg}


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖命令行参数
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    if args.backbone:
        config['model']['backbone'] = args.backbone
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志
    logger = Logger(output_dir / 'logs')
    
    # 保存配置
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 构建数据加载器
    train_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        img_size=config['data']['img_size'],
        num_workers=config['data']['num_workers']
    )
    
    # 构建模型
    model = ContrastiveModel(
        backbone_name=config['model']['backbone'],
        projection_dim=config['model']['projection_dim'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    print(f"模型: {config['model']['backbone']}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 构建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 构建损失函数
    criterion = SimCLRLoss(temperature=config['loss']['temperature'])
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # 训练循环
    print("\n开始训练...")
    best_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练
        train_metrics = train_one_epoch(
            epoch, model, None, train_loader, criterion, optimizer, device, config
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录日志
        logger.log({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'lr': current_lr
        })
        
        # 打印进度
        print(f"Epoch [{epoch}/{config['training']['epochs']}] "
              f"Loss: {train_metrics['loss']:.4f} "
              f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }
            
            torch.save(checkpoint, output_dir / 'best_teacher.pth')
            print(f"✓ 保存最佳模型 (loss: {best_loss:.4f})")
        
        # 定期保存检查点
        if epoch % config['training']['save_freq'] == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    print(f"\n训练完成！最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {output_dir / 'best_teacher.pth'}")
    
    # 保存训练历史
    logger.save_history(output_dir / 'training_history.json')


if __name__ == '__main__':
    main()
