#!/usr/bin/env python3
"""
统一实验运行器 - 支持所有基线方法
用法: python run_baseline.py --method padim --category wood
"""

import sys
import argparse
import torch
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import MVTecWoodDataset, MVTecTestDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def get_model_class(method_name):
    """根据方法名获取模型类"""
    method_map = {
        'padim': ('baselines.padim', 'PaDiMModel'),
        'cfa': ('baselines.cfa', 'CFAModel'),
        'patchcore': ('baselines.patchcore', 'PatchCoreModel'),
        'simplenet': ('baselines.simplenet', 'SimpleNetModel'),
        'spade': ('baselines.spade', 'SPADEModel'),
        'autoencoder': ('baselines.autoencoder', 'AutoEncoderModel'),
        'vae': ('baselines.vae', 'VAEModel'),  # 修复: VAEBaseline -> VAEModel
        'fastflow': ('baselines.fastflow', 'FastFlowModel'),
        'stpm': ('baselines.stpm', 'STPMModel'),
        'efficientad': ('baselines.efficientad', 'EfficientADModel'),
        'reverse_distillation': ('baselines.reverse_distillation', 'ReverseDistillationModel'),
    }
    
    if method_name not in method_map:
        raise ValueError(f"未知的方法: {method_name}. 可用方法: {list(method_map.keys())}")
    
    module_name, class_name = method_map[method_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description='统一基线实验运行器')
    parser.add_argument('--method', type=str, required=True, 
                        choices=['padim', 'cfa', 'patchcore', 'simplenet', 'spade', 
                                'autoencoder', 'vae', 'fastflow', 'stpm', 'efficientad', 'reverse_distillation'])
    parser.add_argument('--category', type=str, default='wood')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)  # RTX 3090优化：从32增加到128
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    # 设置默认路径
    if args.data_dir is None:
        args.data_dir = f'../datasets/mvtec_anomaly_detection/{args.category}'
    
    if args.output_dir is None:
        args.output_dir = f'experiments/baselines_full/{args.category}/{args.method}'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"🚀 运行 {args.method.upper()} 基线 on {args.category}")
    print(f"{'='*70}")
    print(f"设备: {device}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{'='*70}\n")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("加载数据集...")
    train_dataset = MVTecWoodDataset(root=args.data_dir, split='train', defect_type='good', transform=transform)
    test_dataset = MVTecTestDataset(root=args.data_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本\n")
    
    # 创建模型
    print(f"初始化 {args.method} 模型...")
    ModelClass = get_model_class(args.method)
    
    # 根据不同方法传递不同参数
    if args.method in ['padim', 'spade']:
        model = ModelClass(backbone_name='resnet18')
    elif args.method in ['cfa', 'patchcore']:
        model = ModelClass(backbone='resnet18')
    elif args.method in ['autoencoder', 'vae']:
        model = ModelClass(latent_dim=128, device=str(device))
    elif args.method == 'simplenet':
        model = ModelClass(backbone='resnet18', device=str(device))
    else:
        model = ModelClass(device=str(device))
    
    # 训练
    print(f"\n开始训练...")
    if args.method == 'padim':
        model.fit(train_loader)
    elif args.method in ['cfa', 'patchcore']:
        model.train(train_loader)
    elif args.method == 'simplenet':
        model.train_model(train_loader, epochs=args.epochs)
    elif args.method == 'spade':
        model.build_memory_bank(train_loader)
    else:
        model.train(train_loader, epochs=args.epochs)
    
    # 保存模型
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.method == 'padim':
        model.save(output_path / f'{args.method}_model.pkl')
    else:
        model.save(output_path / f'{args.method}_model.pth')
    
    print(f"模型已保存到: {output_path}\n")
    
    # 评估
    print("开始评估...")
    metrics = model.evaluate(test_loader)
    
    print(f"\n{'='*70}")
    print(f"✅ {args.method.upper()} 实验完成！")
    print(f"{'='*70}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AP: {metrics['ap']:.4f}")
    print(f"{'='*70}\n")
    
    # 保存结果
    with open(output_path / 'results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"结果已保存到: {output_path / 'results.json'}")


if __name__ == '__main__':
    main()
