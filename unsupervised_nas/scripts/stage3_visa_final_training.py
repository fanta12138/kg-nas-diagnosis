#!/usr/bin/env python3
"""
VisA数据集 - Stage 3: 使用最佳架构进行最终训练
基于KG-NAS找到的最优backbone和feature_dim进行100 epochs训练
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent  # unsupervised_nas -> autopaper
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[0]}")

from unsupervised_nas.models.contrastive_model import ContrastiveModel
from datasets.visa_loader import VisADataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VisA Stage 3: Final Training')
    parser.add_argument('--dataset', type=str, default='visa', help='数据集名称')
    parser.add_argument('--data_root', type=str, required=True, help='VisA数据集根目录')
    parser.add_argument('--category', type=str, required=True, help='类别名称')
    parser.add_argument('--teacher_model', type=str, required=True, help='教师模型路径')
    parser.add_argument('--backbone', type=str, required=True, help='Backbone类型')
    parser.add_argument('--feature_dim', type=int, required=True, help='特征维度')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')  # 增加到128，充分利用显存
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数')  # 有效batch = batch_size * accumulation_steps
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    
    return parser.parse_args()


def create_dataloaders(data_root, category, img_size, batch_size):
    """创建数据加载器"""
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = VisADataset(
        root=data_root,
        category=category,
        split='train',
        transform=train_transform
    )
    
    test_dataset = VisADataset(
        root=data_root,
        category=category,
        split='test',
        transform=test_transform
    )
    
    print(f"[VisA-{category}] train: 加载了 {len(train_dataset)} 张图像")
    print(f"[VisA-{category}] test: 加载了 {len(test_dataset)} 张图像")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # 增加到8，最大化数据加载速度
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # 保持worker进程，减少创建开销
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,  # 增加到8
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader


def train_epoch(model, teacher, train_loader, optimizer, device, epoch, scaler=None, accumulation_steps=1):
    """训练一个epoch"""
    
    model.train()
    teacher.eval()
    
    total_loss = 0
    count = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    optimizer.zero_grad()  # 在epoch开始时清零
    
    for batch_idx, (images, _) in enumerate(progress_bar):
        images = images.to(device)
        
        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                student_features = model(images)
                
                with torch.no_grad():
                    teacher_features = teacher(images)
                
                # 计算知识蒸馏损失
                kd_loss = nn.MSELoss()(student_features, teacher_features) / accumulation_steps
            
            # 反向传播（混合精度 + 梯度累积）
            scaler.scale(kd_loss).backward()
            
            # 每accumulation_steps步更新一次
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 前向传播
            student_features = model(images)
            
            with torch.no_grad():
                teacher_features = teacher(images)
            
            # 计算知识蒸馏损失
            kd_loss = nn.MSELoss()(student_features, teacher_features) / accumulation_steps
            
            # 反向传播
            kd_loss.backward()
            
            # 每accumulation_steps步更新一次
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 记录损失
        batch_size = images.size(0)
        total_loss += kd_loss.item() * batch_size * accumulation_steps
        count += batch_size
        
        avg_loss = total_loss / count
        progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    # 处理剩余的梯度
    if accumulation_steps > 1 and (len(train_loader) % accumulation_steps) != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    return total_loss / count


def evaluate(model, teacher, test_loader, device):
    """评估模型"""
    
    model.eval()
    teacher.eval()
    
    total_kd_loss = 0
    count = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            
            student_features = model(images)
            teacher_features = teacher(images)
            
            kd_loss = nn.MSELoss()(student_features, teacher_features)
            
            total_kd_loss += kd_loss.item() * images.size(0)
            count += images.size(0)
    
    avg_kd_loss = total_kd_loss / count
    
    return avg_kd_loss


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载教师模型
    print("\n加载教师模型...")
    teacher_checkpoint = torch.load(args.teacher_model, map_location=device)
    
    # 检查教师模型的projection_dim
    teacher_proj_dim = teacher_checkpoint['model_state_dict']['projector.projection.3.weight'].shape[0]
    print(f"教师模型 projection_dim: {teacher_proj_dim}")
    print(f"学生模型 feature_dim: {args.feature_dim}")
    
    # 如果维度不匹配，创建教师模型时使用学生的feature_dim
    if teacher_proj_dim != args.feature_dim:
        print(f"[WARN]  维度不匹配！将教师模型投影层调整为 {args.feature_dim}d")
        # 创建教师模型时使用学生的feature_dim
        teacher = ContrastiveModel(
            backbone_name='convnextv2_tiny',
            projection_dim=args.feature_dim,  # 使用学生的维度
            pretrained=False
        ).to(device)
        
        # 加载权重（跳过projection层的shape不匹配）
        teacher_state_dict = teacher.state_dict()
        for key, value in teacher_checkpoint['model_state_dict'].items():
            if key in teacher_state_dict and teacher_state_dict[key].shape == value.shape:
                teacher_state_dict[key] = value
            else:
                print(f"  跳过不匹配的层: {key} (teacher: {value.shape}, student: {teacher_state_dict[key].shape})")
        
        teacher.load_state_dict(teacher_state_dict, strict=False)
    else:
        # 维度匹配，直接加载
        teacher = ContrastiveModel(
            backbone_name='convnextv2_tiny',
            projection_dim=args.feature_dim,
            pretrained=False
        ).to(device)
        teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    
    teacher.eval()
    print("[OK] 教师模型加载成功\n")
    
    # 创建学生模型（使用NAS找到的最佳架构）
    print(f"创建学生模型: {args.backbone} + {args.feature_dim}d")
    student = ContrastiveModel(
        backbone_name=args.backbone,
        projection_dim=args.feature_dim,
        pretrained=True  # 使用ImageNet预训练
    ).to(device)
    print(f"[OK] 学生模型创建成功\n")
    
    # 创建数据加载器
    print("加载数据集...")
    train_loader, test_loader = create_dataloaders(
        args.data_root,
        args.category,
        args.img_size,
        args.batch_size
    )
    print()
    
    # 优化器
    optimizer = AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 混合精度训练（提高GPU利用率）
    scaler = torch.cuda.amp.GradScaler()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 训练循环
    print(f"\n开始训练 ({args.epochs} epochs)...")
    print("="*70)
    
    best_loss = float('inf')
    training_log = []
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(student, teacher, train_loader, optimizer, device, epoch, scaler, args.accumulation_steps)
        
        # 评估
        test_loss = evaluate(student, teacher, test_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'lr': scheduler.get_last_lr()[0]
        }
        training_log.append(log_entry)
        
        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'architecture': {
                    'backbone': args.backbone,
                    'feature_dim': args.feature_dim
                }
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  🎉 New best! Test Loss: {test_loss:.4f}")
        
        # 每10个epoch打印一次
        if epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Best: {best_loss:.4f} | "
                  f"Time: {elapsed/60:.1f}min")
    
    # 保存训练日志
    import json
    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    total_time = time.time() - start_time
    
    # 保存最终结果
    final_result = {
        'category': args.category,
        'architecture': {
            'backbone': args.backbone,
            'feature_dim': args.feature_dim
        },
        'best_test_loss': best_loss,
        'total_epochs': args.epochs,
        'training_time_hours': total_time / 3600
    }
    
    with open(os.path.join(args.output_dir, 'final_result.json'), 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print("\n" + "="*70)
    print("训练完成！")
    print(f"最佳测试损失: {best_loss:.4f}")
    print(f"总耗时: {total_time/3600:.2f}小时")
    print(f"结果保存在: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
