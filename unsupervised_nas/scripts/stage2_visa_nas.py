#!/usr/bin/env python3
"""
VisA数据集 - Stage 2: NAS架构搜索

使用教师模型的知识指导神经架构搜索
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 必须在导入其他模块之前设置路径
from datasets.visa_loader import VisADataset, VISA_CATEGORIES
from unsupervised_nas.models.contrastive_model import ContrastiveModel
from unsupervised_nas.methods.nas_engine import NASSearchEngine


def parse_args():
    parser = argparse.ArgumentParser(description='VisA Stage 2: NAS Search')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='VisA数据集根目录')
    parser.add_argument('--category', type=str, required=True,
                        choices=VISA_CATEGORIES,
                        help='类别名称')
    parser.add_argument('--teacher-model', type=str, required=True,
                        help='教师模型路径')
    parser.add_argument('--output-dir', type=str, default='experiments/visa/stage2',
                        help='输出目录')
    parser.add_argument('--trials', type=int, default=50,
                        help='NAS搜索次数')
    parser.add_argument('--search-epochs', type=int, default=3,
                        help='每个trial的训练epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--img-size', type=int, default=224,
                        help='图像尺寸（使用224以兼容Swin/DeiT等模型）')
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
    print(f"VisA Stage 2: NAS Architecture Search")
    print(f"{'='*70}")
    print(f"类别: {args.category}")
    print(f"教师模型: {args.teacher_model}")
    print(f"Trials: {args.trials}")
    print(f"Search epochs: {args.search_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    # 数据变换
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载VisA数据集（仅正常样本用于NAS评估）
    print("加载数据集...")
    train_dataset = VisADataset(
        root=args.data_dir,
        category=args.category,
        split='train',
        transform=val_transform,
        return_mask=False
    )
    
    test_dataset = VisADataset(
        root=args.data_dir,
        category=args.category,
        split='test',
        transform=val_transform,
        return_mask=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 避免多进程路径问题
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 避免多进程路径问题
        pin_memory=True
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # 加载教师模型
    print("加载教师模型...")
    teacher_checkpoint = torch.load(args.teacher_model, map_location=device)
    
    # 从checkpoint推断backbone名称（简化处理，默认convnextv2_tiny）
    teacher_backbone = 'convnextv2_tiny'
    
    teacher_model = ContrastiveModel(
        backbone_name=teacher_backbone,
        projection_dim=128,
        pretrained=False
    ).to(device)
    
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()
    
    print(f"✅ 教师模型加载成功\n")
    
    # 定义搜索空间（基于稳定性诊断结果）
    # 已移除不稳定的模型：MobileNetV3 (CV=155%) 和 EfficientNet_b0 (CV=113%)
    # 保留7个高度稳定的backbones (CV < 5%, 100%成功率)
    search_space = {
        'backbones': [
            # CNN系列 (2015-2016)
            'resnet18', 'resnet34', 'resnet50',          # ResNet - CV: 1.9-4.2%
            
            # ConvNeXt系列 (2022) - 最稳定
            'convnext_tiny', 'convnext_small',           # ConvNeXt - CV: 0.9-1.5%
            
            # Transformer系列 (2021)
            'deit_tiny_patch16_224',                     # DeiT - CV: 1.1%
            'swin_tiny_patch4_window7_224'               # Swin - CV: 1.3%
        ],
        'feature_dims': [128, 256, 512]
    }
    
    # 创建NAS搜索引擎
    print("初始化NAS搜索引擎...")
    nas_engine = NASSearchEngine(
        search_space=search_space,
        teacher_model=teacher_model,
        device=device
    )
    
    # 执行NAS搜索
    print(f"开始NAS搜索 ({args.trials} trials)...\n")
    
    best_score = -float('inf')
    best_architecture = None
    all_results = []
    
    for trial in range(args.trials):
        print(f"\n--- Trial {trial+1}/{args.trials} ---")
        
        # 采样架构
        architecture = nas_engine.sample_architecture()
        
        # 快速训练并评估
        score = nas_engine.evaluate_architecture(
            arch_config=architecture,
            train_loader=train_loader,
            val_loader=test_loader,
            quick_epochs=args.search_epochs
        )
        
        print(f"Trial {trial+1}: Score = {score:.4f}")
        print(f"  Architecture: {architecture}")
        
        # 记录结果
        result = {
            'trial': trial + 1,
            'score': score,
            'architecture': architecture
        }
        all_results.append(result)
        
        # 更新最佳
        if score > best_score:
            best_score = score
            best_architecture = architecture
            print(f"  🎉 New best! Score: {best_score:.4f}")
            
            # 保存最佳架构
            best_arch_path = output_dir / 'best_architecture.json'
            with open(best_arch_path, 'w') as f:
                json.dump({
                    'architecture': best_architecture,
                    'score': best_score,
                    'trial': trial + 1
                }, f, indent=2)
    
    # 保存所有结果
    results_path = output_dir / 'all_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'category': args.category,
            'total_trials': args.trials,
            'best_score': best_score,
            'best_architecture': best_architecture,
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"NAS搜索完成！")
    print(f"最佳Score: {best_score:.4f}")
    print(f"最佳架构: {best_architecture}")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
