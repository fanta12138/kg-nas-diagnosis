#!/usr/bin/env python3
"""
模型评估脚本
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import AnomalyDetector
from datasets import MVTecTestDataset


def parse_args():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--data-dir', type=str, default='../datasets/MVTec-AD/wood',
                        help='数据集目录')
    parser.add_argument('--output-dir', type=str, default='experiments/final',
                        help='输出目录')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    normal_prototype = checkpoint['normal_prototype']
    
    model = AnomalyDetector(
        backbone_name=config['model']['backbone'],
        feature_dim=config['model']['feature_dim'],
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型架构: {config['model']['backbone']}")
    print(f"特征维度: {config['model']['feature_dim']}")
    
    # 数据加载
    test_transform = transforms.Compose([
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = MVTecTestDataset(root=args.data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 评估
    print("\n开始评估...")
    all_scores = []
    all_labels = []
    all_defect_types = []
    
    with torch.no_grad():
        for images, labels, defect_types in test_loader:
            images = images.to(device)
            features = model.get_features(images)
            scores = torch.norm(features - normal_prototype.to(device), dim=1)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_defect_types.extend(defect_types)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 计算整体指标
    auroc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    # 找到最佳F1阈值
    thresholds = np.percentile(all_scores, np.arange(0, 100, 1))
    best_f1 = 0
    best_threshold = 0
    
    for thresh in thresholds:
        predictions = (all_scores >= thresh).astype(int)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"\n{'='*60}")
    print(f"评估结果:")
    print(f"{'='*60}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AP: {ap:.4f}")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"{'='*60}")
    
    # 按缺陷类型统计
    print(f"\n按缺陷类型统计:")
    print(f"{'-'*60}")
    unique_types = sorted(set(all_defect_types))
    for defect_type in unique_types:
        mask = np.array([dt == defect_type for dt in all_defect_types])
        if mask.sum() > 0:
            type_scores = all_scores[mask]
            type_labels = all_labels[mask]
            type_auroc = roc_auc_score(type_labels, type_scores) if len(set(type_labels)) > 1 else 0.0
            print(f"  {defect_type:20s}: AUROC={type_auroc:.4f}, Count={mask.sum()}")
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'auroc': float(auroc),
        'ap': float(ap),
        'best_f1': float(best_f1),
        'threshold': float(best_threshold),
        'model_path': str(args.model_path),
        'per_defect_type': {}
    }
    
    for defect_type in unique_types:
        mask = np.array([dt == defect_type for dt in all_defect_types])
        if mask.sum() > 0:
            type_scores = all_scores[mask]
            type_labels = all_labels[mask]
            type_auroc = roc_auc_score(type_labels, type_scores) if len(set(type_labels)) > 1 else 0.0
            results['per_defect_type'][defect_type] = {
                'auroc': float(type_auroc),
                'count': int(mask.sum())
            }
    
    with open(output_dir / 'final_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_dir / 'final_evaluation.json'}")


if __name__ == '__main__':
    main()
