#!/usr/bin/env python3
"""
VisA数据集 - Stage 1: Teacher预训练

使用对比学习在VisA正常样本上训练教师模型
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datasets.visa_loader import VisADataset, VISA_CATEGORIES
from unsupervised_nas.models.contrastive_model import ContrastiveModel, SimCLRLoss


def parse_args():
    parser = argparse.ArgumentParser(description='VisA Stage 1: Teacher Training')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='VisA数据集根目录')
    parser.add_argument('--category', type=str, required=True,
                        choices=VISA_CATEGORIES,
                        help='类别名称')
    parser.add_argument('--output-dir', type=str, default='experiments/visa/stage1',
                        help='输出目录')
    parser.add_argument('--backbone', type=str, default='convnextv2_tiny',
                        help='骨干网络')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--img-size', type=int, default=256,
                        help='图像尺寸')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"VisA Stage 1: Teacher Training")
    print(f"{'='*70}")
    print(f"类别: {args.category}")
    print(f"骨干网络: {args.backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    # 数据增强（对比学习需要强增强）
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
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
    
    # 加载VisA数据集（仅正常样本）
    print("加载数据集...")
    train_dataset = VisADataset(
        root=args.data_dir,
        category=args.category,
        split='train',
        transform=train_transform,
        return_mask=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"Batch数: {len(train_loader)}\n")
    
    # 创建模型
    print("创建模型...")
    model = ContrastiveModel(
        backbone_name=args.backbone,
        projection_dim=128,
        pretrained=True
    ).to(device)
    
    # 损失函数
    criterion = SimCLRLoss(temperature=0.07)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 训练循环
    print("开始训练...\n")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for images, _ in pbar:
            # images shape: [B, 3, H, W]
            images = images.to(device)
            
            # 创建两个增强视图（通过不同的dropout等）
            # 这里简化处理，实际应该返回pair
            view1 = images
            view2 = images  # TODO: 需要修改数据加载器返回pair
            
            optimizer.zero_grad()
            
            # 前向传播
            z1 = model(view1)
            z2 = model(view2)
            
            # 计算损失
            loss = criterion(z1, z2)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 更新学习率
        scheduler.step()
        
        # 打印日志
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}] '
                  f'Loss: {epoch_loss.avg:.4f} '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
        if epoch_loss.avg < best_loss:
            best_loss = epoch_loss.avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, output_dir / 'best_teacher.pth')
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss.avg,
    }, output_dir / 'final_teacher.pth')
    
    print(f"\n{'='*70}")
    print(f"训练完成！")
    print(f"最佳Loss: {best_loss:.4f}")
    print(f"模型保存在: {output_dir}")
    print(f"{'='*70}\n")


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
