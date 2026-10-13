#!/usr/bin/env python3
"""
阶段2: NAS架构搜索

使用教师模型的知识指导神经架构搜索
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ContrastiveModel
from methods.nas_engine import NASSearchEngine
from datasets import MVTecWoodDataset
from utils import Logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='阶段2: NAS架构搜索')
    parser.add_argument('--config', type=str, default='configs/stage2_nas_search.yaml',
                        help='配置文件路径')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='教师模型路径（覆盖配置）')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据集目录（覆盖配置）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（覆盖配置）')
    parser.add_argument('--num-trials', type=int, default=None, help='搜索次数（覆盖配置）')
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_dataloaders(data_dir: str, batch_size: int, img_size: int, num_workers: int = 4):
    """构建数据加载器"""
    
    # 验证集变换（不需要强增强）
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 训练集（用于快速训练学生模型）
    train_dataset = MVTecWoodDataset(
        root=data_dir,
        split='train',
        defect_type='good',
        transform=val_transform,
        return_pair=False
    )
    
    # 验证集
    val_dataset = MVTecWoodDataset(
        root=data_dir,
        split='test',
        defect_type='good',
        transform=val_transform,
        return_pair=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖命令行参数
    if args.num_trials:
        config['nas']['num_trials'] = args.num_trials
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['output']['dir'] = args.output_dir
    if args.teacher_model:
        teacher_model_path = args.teacher_model
    else:
        teacher_model_path = config.get('teacher', {}).get('model_path', 'experiments/stage1/best_teacher.pth')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志
    logger = Logger(output_dir / 'logs')
    
    # 保存配置
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 加载教师模型
    print("\n加载教师模型...")
    teacher_checkpoint = torch.load(teacher_model_path, map_location=device)
    teacher_config = teacher_checkpoint['config']
    
    teacher_model = ContrastiveModel(
        backbone_name=teacher_config['model']['backbone'],
        projection_dim=teacher_config['model']['projection_dim'],
        pretrained=False
    ).to(device)
    
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    print(f"教师模型已加载: {args.teacher_model}")
    
    # 构建数据加载器
    print("\n构建数据加载器...")
    train_loader, val_loader = build_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['nas']['batch_size'],
        img_size=config['data']['img_size'],
        num_workers=config['data']['num_workers']
    )
    
    # 创建NAS搜索引擎
    print("\n初始化NAS搜索引擎...")
    search_engine = NASSearchEngine(
        search_space=config['nas']['search_space'],
        teacher_model=teacher_model,
        device=device,
        alpha=config['nas'].get('alpha', 0.5),
        beta=config['nas'].get('beta', 0.3),
        gamma=config['nas'].get('gamma', 0.2)
    )
    
    # 执行搜索
    print("\n开始架构搜索...")
    best_architecture = search_engine.search(
        train_loader=train_loader,
        val_loader=val_loader,
        num_trials=config['nas']['num_trials'],
        quick_epochs=config['nas']['quick_train_epochs']
    )
    
    # 保存结果
    print("\n保存搜索结果...")
    search_engine.save_results(output_dir / 'search_results.json')
    
    # 保存最佳架构配置
    best_config = {
        'best_architecture': best_architecture,
        'search_config': config
    }
    
    with open(output_dir / 'best_architecture.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"阶段2完成！")
    print(f"{'='*60}")
    print(f"最佳架构已保存到: {output_dir / 'best_architecture.yaml'}")
    print(f"完整结果已保存到: {output_dir / 'search_results.json'}")
    print(f"\n下一步:")
    print(f"  使用最佳架构进行最终训练")
    print(f"  python scripts/stage3_final_training.py --arch-config {output_dir / 'best_architecture.yaml'}")


if __name__ == '__main__':
    main()
