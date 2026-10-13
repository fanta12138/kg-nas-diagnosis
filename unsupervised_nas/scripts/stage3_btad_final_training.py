#!/usr/bin/env python3
"""
BTAD数据集 - Stage 3: 使用最佳架构进行最终训练
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
from unsupervised_nas.datasets.btad_loader import BTADDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BTAD Stage 3: Final Training')
    parser.add_argument('--data-dir', type=str, required=True, help='BTAD数据集根目录')
    parser.add_argument('--category', type=str, required=True, help='类别名称 (01, 02, 03)')
    parser.add_argument('--nas-results', type=str, required=True, help='NAS搜索结果JSON文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小（优化：从64降至32以减少显存压力）')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--img-size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    return parser.parse_args()


def create_dataloaders(data_dir, category, img_size, batch_size):
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
    train_dataset = BTADDataset(
        root_dir=data_dir,
        category=category,
        split='train',
        transform=train_transform
    )
    
    test_dataset = BTADDataset(
        root_dir=data_dir,
        category=category,
        split='test',
        transform=test_transform
    )
    
    print(f"[BTAD-{category}] Train samples: {len(train_dataset)}")
    print(f"[BTAD-{category}] Test samples: {len(test_dataset)}")
    
    # 创建DataLoader（优化配置：减少worker数量以降低内存压力）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 从8降至4，减少内存占用
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,  # 从4降至2，减少预取缓冲
        persistent_workers=True  # 保持worker进程，避免重复创建
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # 从8降至4
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader


def load_teacher_model(category, data_dir, device):
    """加载教师模型"""
    # 教师模型路径（支持嵌套和扁平两种路径）
    teacher_paths = [
        Path("experiments/btad_stage1") / category / category / "final_teacher.pth",
        Path("experiments/btad_stage1") / category / "final_teacher.pth"
    ]
    
    teacher_path = None
    for p in teacher_paths:
        if p.exists():
            teacher_path = p
            break
    
    if teacher_path is None:
        raise FileNotFoundError(f"教师模型不存在！尝试的路径:\n{teacher_paths}")
    
    print(f"加载教师模型: {teacher_path}")
    checkpoint = torch.load(teacher_path, map_location=device)
    
    # 检查projection_dim
    proj_dim_key = 'projector.projection.3.weight'
    if proj_dim_key not in checkpoint['model_state_dict']:
        raise KeyError(f"教师模型缺少关键层: {proj_dim_key}")
    
    teacher_proj_dim = checkpoint['model_state_dict'][proj_dim_key].shape[0]
    print(f"教师模型 projection_dim: {teacher_proj_dim}")
    
    # 创建教师模型
    teacher = ContrastiveModel(
        backbone_name='convnextv2_tiny',
        projection_dim=teacher_proj_dim,
        pretrained=False
    ).to(device)
    
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()
    
    print("[OK] 教师模型加载成功\n")
    return teacher, teacher_proj_dim


def train_epoch(model, teacher, train_loader, optimizer, device, epoch, scaler=None, alignment_layer=None):
    """训练一个epoch"""
    
    model.train()
    teacher.eval()
    if alignment_layer is not None:
        alignment_layer.train()
    
    total_loss = 0
    count = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, _ in progress_bar:
        images = images.to(device)
        
        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                student_features = model(images)
                
                with torch.no_grad():
                    teacher_features = teacher(images)
                    # 如果维度不匹配，使用对齐层
                    if alignment_layer is not None:
                        teacher_features = alignment_layer(teacher_features)
                
                # 计算知识蒸馏损失
                kd_loss = nn.MSELoss()(student_features, teacher_features)
            
            # 反向传播（混合精度）
            scaler.scale(kd_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # 前向传播
            student_features = model(images)
            
            with torch.no_grad():
                teacher_features = teacher(images)
                # 如果维度不匹配，使用对齐层
                if alignment_layer is not None:
                    teacher_features = alignment_layer(teacher_features)
            
            # 计算知识蒸馏损失
            kd_loss = nn.MSELoss()(student_features, teacher_features)
            
            # 反向传播
            kd_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 记录损失
        batch_size = images.size(0)
        total_loss += kd_loss.item() * batch_size
        count += batch_size
        
        avg_loss = total_loss / count
        progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    return total_loss / count


def evaluate(model, teacher, test_loader, device, alignment_layer=None):
    """评估模型"""
    
    model.eval()
    teacher.eval()
    if alignment_layer is not None:
        alignment_layer.eval()
    
    total_kd_loss = 0
    count = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            
            student_features = model(images)
            teacher_features = teacher(images)
            
            # 如果维度不匹配，使用对齐层
            if alignment_layer is not None:
                teacher_features = alignment_layer(teacher_features)
            
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
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载NAS结果
    print(f"\n加载NAS结果: {args.nas_results}")
    with open(args.nas_results, 'r') as f:
        nas_results = json.load(f)
    
    best_architecture = nas_results['best_architecture']
    backbone = best_architecture['backbone']
    feature_dim = best_architecture['feature_dim']
    
    print(f"最佳架构: {backbone} + {feature_dim}d")
    if 'best_score' in nas_results:
        print(f"Best Score: {nas_results['best_score']:.4f}")
    print()
    
    # 加载教师模型
    teacher, teacher_proj_dim = load_teacher_model(args.category, args.data_dir, device)
    
    # 如果维度不匹配，调整教师模型
    if teacher_proj_dim != feature_dim:
        print(f"[WARN] 维度不匹配！教师: {teacher_proj_dim}d, 学生: {feature_dim}d")
        print(f"[INFO] 将创建维度对齐层\n")
        
        # 创建维度对齐层（线性投影）
        alignment_layer = nn.Linear(teacher_proj_dim, feature_dim).to(device)
    else:
        alignment_layer = None
    
    # 创建学生模型（使用NAS找到的最佳架构）
    print(f"创建学生模型: {backbone} + {feature_dim}d")
    student = ContrastiveModel(
        backbone_name=backbone,
        projection_dim=feature_dim,
        pretrained=True  # 使用ImageNet预训练
    ).to(device)
    print(f"[OK] 学生模型创建成功\n")
    
    # 创建数据加载器
    print("加载数据集...")
    train_loader, test_loader = create_dataloaders(
        args.data_dir,
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
        train_loss = train_epoch(student, teacher, train_loader, optimizer, device, epoch, scaler, alignment_layer)
        
        # 评估
        test_loss = evaluate(student, teacher, test_loader, device, alignment_layer)
        
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
                    'backbone': backbone,
                    'feature_dim': feature_dim
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
    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    total_time = time.time() - start_time
    
    # 保存最终结果
    final_result = {
        'category': args.category,
        'architecture': {
            'backbone': backbone,
            'feature_dim': feature_dim
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
