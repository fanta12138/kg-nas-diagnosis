#!/usr/bin/env python3
"""
BTAD数据集 - Stage 1: Teacher预训练

使用对比学习在BTAD正常样本上训练教师模型
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

from datasets.btad_loader import BTADDataset, get_btad_transforms
from unsupervised_nas.models.contrastive_model import ContrastiveModel, SimCLRLoss


def parse_args():
    parser = argparse.ArgumentParser(description='BTAD Stage 1: Teacher Training')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='BTAD数据集根目录')
    parser.add_argument('--category', type=str, required=True,
                        choices=['01', '02', '03'],
                        help='类别名称')
    parser.add_argument('--output-dir', type=str, default='experiments/btad_stage1',
                        help='输出目录')
    parser.add_argument('--backbone', type=str, default='convnextv2_tiny',
                        help='骨干网络')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小（优化：从64降至32以减少显存压力）')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--img-size', type=int, default=256,
                        help='图像尺寸')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    return parser.parse_args()


class PairDataset(torch.utils.data.Dataset):
    """返回两个增强视图的数据集（用于对比学习）"""
    
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label, path = self.dataset[idx]
        
        # 应用两种不同的增强
        view1 = self.transform1(img)
        view2 = self.transform2(img)
        
        return view1, view2, label


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_path = Path(args.output_dir) / args.category
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"BTAD Stage 1: Teacher预训练")
    print(f"{'='*80}")
    print(f"类别: {args.category}")
    print(f"骨干网络: {args.backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"图像尺寸: {args.img_size}")
    print(f"输出目录: {output_path}")
    print(f"{'='*80}\n")
    
    # 加载数据集（只使用训练集的正常样本）
    print("加载数据集...")
    train_transform1, _ = get_btad_transforms(args.img_size)
    train_transform2, _ = get_btad_transforms(args.img_size)
    
    # 为对比学习创建更强的增强
    strong_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    base_dataset = BTADDataset(
        root_dir=args.data_dir,
        category=args.category,
        split='train',
        transform=None  # 不在这里应用transform
    )
    
    # 创建配对数据集
    pair_dataset = PairDataset(base_dataset, strong_transform, strong_transform)
    
    train_loader = DataLoader(
        pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # 从8降至4，减少内存压力
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,  # 从4降至2，减少预取
        persistent_workers=True  # 保持worker进程，避免重复创建
    )
    
    print(f"训练样本数: {len(pair_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    
    # 创建模型
    print("\n创建对比学习模型...")
    model = ContrastiveModel(
        backbone_name=args.backbone,
        projection_dim=128,
        pretrained=True
    ).to(device)
    
    # 损失函数
    criterion = SimCLRLoss(temperature=0.07).to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    print(f"\n开始训练 ({args.epochs} epochs)...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for view1, view2, labels in progress_bar:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            z1 = model(view1)
            z2 = model(view2)
            
            # 计算损失
            loss = criterion(z1, z2)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # 更新学习率
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        
        # 打印进度
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch [{epoch}/{args.epochs}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'backbone': args.backbone,
            }, output_path / 'best_teacher.pth')
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'backbone': args.backbone,
    }, output_path / 'final_teacher.pth')
    
    print(f"\n{'='*80}")
    print(f"训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"最终损失: {avg_loss:.4f}")
    print(f"结果保存在: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
