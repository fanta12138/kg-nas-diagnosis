#!/usr/bin/env python3
"""
BTAD数据集 - Stage 2: NAS架构搜索

使用教师模型的知识指导神经架构搜索
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from unsupervised_nas.models.contrastive_model import ContrastiveModel
from unsupervised_nas.methods.nas_engine import NASSearchEngine
from unsupervised_nas.datasets.btad_loader import BTADDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BTAD Stage 2: NAS架构搜索')
    parser.add_argument('--teacher-model', type=str, required=True,
                        help='教师模型路径')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='BTAD数据集根目录')
    parser.add_argument('--category', type=str, required=True,
                        choices=['01', '02', '03'],
                        help='BTAD类别')
    parser.add_argument('--output-dir', type=str, default='experiments/btad_stage2',
                        help='输出目录')
    parser.add_argument('--num-trials', type=int, default=100,
                        help='搜索次数')
    parser.add_argument('--epochs-per-trial', type=int, default=3,
                        help='每次trial训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--img-size', type=int, default=224,
                        help='图像尺寸（ViT/DeiT需要224）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    return parser.parse_args()


def build_dataloaders(data_dir: str, category: str, batch_size: int, img_size: int, num_workers: int = 4):
    """构建BTAD数据加载器"""
    
    # 验证集变换
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 训练集（只使用正常样本）
    train_dataset = BTADDataset(
        root_dir=data_dir,
        category=category,
        split='train',
        transform=val_transform
    )
    
    # 验证集（测试集的正常样本）
    val_dataset = BTADDataset(
        root_dir=data_dir,
        category=category,
        split='test',
        transform=val_transform
    )
    
    # 过滤出正常样本用于验证
    normal_indices = [i for i, label in enumerate(val_dataset.labels) if label == 0]
    if len(normal_indices) > 0:
        val_dataset.image_paths = [val_dataset.image_paths[i] for i in normal_indices]
        val_dataset.labels = [val_dataset.labels[i] for i in normal_indices]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    return train_loader, val_loader


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载教师模型
    print("\n加载教师模型...")
    teacher_checkpoint = torch.load(args.teacher_model, map_location=device, weights_only=False)
    
    # 提取backbone名称
    if 'backbone' in teacher_checkpoint:
        backbone_name = teacher_checkpoint['backbone']
    else:
        backbone_name = 'convnextv2_tiny'  # 默认
    
    teacher_model = ContrastiveModel(
        backbone_name=backbone_name,
        projection_dim=128,
        pretrained=False
    ).to(device)
    
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()  # 设置为评估模式
    print(f"教师模型已加载: {args.teacher_model}")
    print(f"Backbone: {backbone_name}")
    
    # 构建数据加载器
    print("\n构建数据加载器...")
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        category=args.category,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=4
    )
    
    # 定义搜索空间（使用BackboneFactory支持的名称）
    search_space = {
        'backbones': [
            'resnet18', 'resnet34', 'resnet50',
            'efficientnet_b0', 'mobilenetv3_small_100',
            'convnext_tiny', 'convnextv2_tiny',
            'swin_tiny_patch4_window7_224', 'deit_tiny_patch16_224', 'vit_tiny_patch16_224'
        ],
        'feature_dims': [128, 256, 512]
    }
    
    # 创建NAS搜索引擎
    print("\n初始化NAS搜索引擎...")
    search_engine = NASSearchEngine(
        search_space=search_space,
        teacher_model=teacher_model,
        device=device,
        alpha=0.5,   # KD loss权重
        beta=0.3,    # Consistency loss权重
        gamma=0.2    # InfoNCE loss权重
    )
    
    # 执行搜索
    print(f"\n开始架构搜索 ({args.num_trials} trials)...")
    print(f"每个trial训练 {args.epochs_per_trial} epochs")
    
    best_architecture = search_engine.search(
        train_loader=train_loader,
        val_loader=val_loader,
        num_trials=args.num_trials,
        quick_epochs=args.epochs_per_trial
    )
    
    # 保存结果
    print("\n保存搜索结果...")
    
    # 保存JSON格式结果
    results_data = {
        'best_architecture': best_architecture,
        'num_trials': args.num_trials,
        'epochs_per_trial': args.epochs_per_trial,
        'category': args.category,
        'teacher_backbone': backbone_name
    }
    
    with open(output_dir / 'nas_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # 保存YAML格式配置（兼容stage3）
    best_config = {
        'best_architecture': best_architecture,
        'search_config': {
            'num_trials': args.num_trials,
            'epochs_per_trial': args.epochs_per_trial,
            'batch_size': args.batch_size,
            'img_size': args.img_size
        }
    }
    
    import yaml
    with open(output_dir / 'best_architecture.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"\n{'='*80}")
    print(f"BTAD-{args.category} Stage 2完成！")
    print(f"{'='*80}")
    print(f"最佳架构: {best_architecture}")
    print(f"结果保存到: {output_dir}")
    print(f"\n下一步:")
    print(f"  使用最佳架构进行最终训练")
    print(f"  python unsupervised_nas/scripts/stage3_final_train.py \\")
    print(f"    --category {args.category} \\")
    print(f"    --nas-results {output_dir / 'nas_results.json'}")


if __name__ == '__main__':
    main()
