#!/usr/bin/env python3
"""
阶段3: 使用最佳架构进行最终训练和异常检测
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Dict
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 修复 Windows 管道输出编码问题（GBK 无法编码 ✓ 等 Unicode 字符）
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

from models import AnomalyDetector
from datasets import MVTecWoodDataset, MVTecTestDataset
from utils.logger import Logger, AverageMeter

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard不可用，将只保存JSON日志")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='阶段3: 最终训练')
    parser.add_argument('--config', type=str, default='configs/final_training.yaml',
                        help='配置文件路径')
    parser.add_argument('--arch-config', type=str, default='experiments/stage2/best_architecture.yaml',
                        help='NAS搜索得到的最佳架构配置')
    parser.add_argument('--data-dir', type=str, default='../datasets/MVTec-AD/wood',
                        help='数据集目录')
    parser.add_argument('--output-dir', type=str, default='experiments/final',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数（覆盖配置）')
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_dataloaders(data_dir: str, batch_size: int, img_size: int, num_workers: int = 4):
    """构建数据加载器"""
    
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试集变换
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 训练集（只用正常样本）
    train_dataset = MVTecWoodDataset(
        root=data_dir,
        split='train',
        defect_type='good',
        transform=train_transform
    )
    
    # 测试集（包含正常和异常）
    test_dataset = MVTecTestDataset(
        root=data_dir,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_loader, test_loader


def compute_normal_prototype(model, train_loader, device):
    """计算正常样本的原型特征"""
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc="Extracting normal features"):
            images = images.to(device)
            features = model.get_features(images)
            all_features.append(features.cpu())
    
    if len(all_features) == 0:
        raise ValueError("No features extracted")
    
    all_features = torch.cat(all_features, dim=0)
    prototype = all_features.mean(dim=0)
    
    return prototype


def evaluate_anomaly_detection(model, test_loader, normal_prototype, device):
    """评估异常检测性能"""
    model.eval()
    
    all_scores = []
    all_labels = []
    per_defect_metrics = {}
    
    with torch.no_grad():
        for images, labels, defect_types in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # 计算异常分数（与正常原型的距离）
            features = model.get_features(images)
            scores = torch.norm(features - normal_prototype.to(device), dim=1)
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # 按缺陷类型分组
            for score, label, defect_type in zip(scores.cpu().numpy(), labels.numpy(), defect_types):
                if defect_type not in per_defect_metrics:
                    per_defect_metrics[defect_type] = {'scores': [], 'labels': []}
                per_defect_metrics[defect_type]['scores'].append(score)
                per_defect_metrics[defect_type]['labels'].append(label)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 计算整体指标
    if len(np.unique(all_labels)) < 2:
        print("警告: 测试集只有一类标签，无法计算AUROC")
        auroc = 0.0
        ap = 0.0
    else:
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
    
    # 计算每个缺陷类型的指标
    per_defect_auroc = {}
    for defect_type, data in per_defect_metrics.items():
        scores_arr = np.array(data['scores'])
        labels_arr = np.array(data['labels'])
        
        if len(np.unique(labels_arr)) >= 2:
            defect_auroc = roc_auc_score(labels_arr, scores_arr)
        else:
            defect_auroc = 0.0
        
        per_defect_auroc[defect_type] = {
            'auroc': float(defect_auroc),
            'count': int(len(labels_arr))
        }
    
    return {
        'auroc': float(auroc),
        'ap': float(ap),
        'scores': all_scores,
        'labels': all_labels,
        'per_defect_type': per_defect_auroc
    }


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)

    # 覆盖训练轮数
    if args.epochs:
        config['training']['epochs'] = args.epochs

    # 设置随机种子
    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    print(f"随机种子: {random_seed}")
    
    # 加载NAS搜索结果
    if Path(args.arch_config).exists():
        with open(args.arch_config, 'r', encoding='utf-8') as f:
            arch_data = yaml.safe_load(f)
        
        # 覆盖模型配置
        config['model']['backbone'] = arch_data['best_architecture']['backbone']
        config['model']['feature_dim'] = arch_data['best_architecture']['feature_dim']
        print(f"使用NAS搜索的最佳架构:")
        print(f"  Backbone: {config['model']['backbone']}")
        print(f"  Feature Dim: {config['model']['feature_dim']}")
    else:
        print(f"警告: 未找到架构配置 {args.arch_config}，使用默认配置")
    
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
    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        img_size=config['data']['img_size'],
        num_workers=config['data']['num_workers']
    )
    
    # 构建模型
    model = AnomalyDetector(
        backbone_name=config['model']['backbone'],
        feature_dim=config['model']['feature_dim'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    print(f"\n模型: {config['model']['backbone']}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 构建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # 初始化TensorBoard（如果可用）
    writer = None
    if TENSORBOARD_AVAILABLE:
        tb_dir = Path(args.output_dir) / 'tensorboard'
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard日志目录: {tb_dir}")
        print(f"运行命令查看: tensorboard --logdir={tb_dir}")
    
    # 训练循环
    print("\n开始训练...")
    best_auroc = 0
    training_history = []  # 详细的训练历史
    
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for images, _ in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(images)
            features = output['features']
            
            # 使用对比学习风格的自监督损失
            # 策略：对同一batch的样本，最大化特征多样性（避免坍缩）
            if features.dim() == 2:  # 1D特征
                # 标准化特征
                features_norm = torch.nn.functional.normalize(features, dim=1)
                
                # 计算相似度矩阵
                sim_matrix = torch.mm(features_norm, features_norm.t())
                
                # 排除对角线（自身相似度）
                mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
                sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
                
                # 最大化负样本对的差异（InfoNCE风格）
                loss = -torch.log(torch.exp(sim_matrix / 0.5).sum(dim=1)).mean()
            else:  # 2D特征图
                reconstructed = output['reconstructed']
                loss = nn.functional.mse_loss(reconstructed, features)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # 更新学习率
        scheduler.step()
        
        # 记录训练指标
        train_metrics = {
            'epoch': epoch,
            'train_loss': loss_meter.avg,
            'lr': optimizer.param_groups[0]['lr']
        }
        logger.log(train_metrics)
        
        # TensorBoard记录
        if writer is not None:
            writer.add_scalar('Loss/train', loss_meter.avg, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 定期评估
        if epoch % 10 == 0 or epoch == config['training']['epochs']:
            # 计算正常原型
            normal_prototype = compute_normal_prototype(model, train_loader, device)
            
            # 评估
            metrics = evaluate_anomaly_detection(model, test_loader, normal_prototype, device)
            
            auroc = metrics['auroc']
            ap = metrics['ap']
            
            print(f"Epoch [{epoch}/{config['training']['epochs']}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"AUROC: {auroc:.4f} "
                  f"AP: {ap:.4f}")
            
            # 记录验证指标
            val_metrics = {
                'epoch': epoch,
                'val_auroc': auroc,
                'val_ap': ap,
                'train_loss': loss_meter.avg
            }
            logger.log(val_metrics)
            training_history.append(val_metrics)
            
            # TensorBoard记录验证指标
            if writer is not None:
                writer.add_scalar('Metrics/AUROC', auroc, epoch)
                writer.add_scalar('Metrics/AP', ap, epoch)
            
            # 保存最佳模型
            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auroc': best_auroc,
                    'ap': metrics['ap'],
                    'config': config,
                    'normal_prototype': normal_prototype
                }
                
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"✓ 保存最佳模型 (AUROC: {best_auroc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"{'='*60}")
    print(f"最佳AUROC: {best_auroc:.4f}")
    print(f"模型保存在: {output_dir / 'best_model.pth'}")
    
    # 保存训练历史
    logger.save_history(output_dir / 'training_history.json')
    
    # 保存详细的训练历史（包含验证指标）
    history_path = output_dir / 'detailed_training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2)
    print(f"详细训练历史已保存到: {history_path}")
    
    # 关闭TensorBoard
    if writer is not None:
        writer.close()
        print("TensorBoard日志器已关闭")
    
    # 最终评估
    print("\n最终评估...")
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
    
    # 关键修复：使用最佳模型重新计算normal prototype
    print("使用最佳模型重新计算Normal Prototype...")
    model.load_state_dict(checkpoint['model_state_dict'])
    normal_prototype = compute_normal_prototype(model, train_loader, device)
    
    final_metrics = evaluate_anomaly_detection(model, test_loader, normal_prototype, device)
    
    print(f"\n最终结果:")
    print(f"  AUROC: {final_metrics['auroc']:.4f}")
    print(f"  AP: {final_metrics['ap']:.4f}")
    
    # 保存评估结果（包含每个缺陷类型的详细指标）
    eval_results = {
        'auroc': final_metrics['auroc'],
        'ap': final_metrics['ap'],
        'best_f1': float(final_metrics.get('best_f1', 0.0)),
        'threshold': float(final_metrics.get('threshold', 0.0)),
        'model_path': str(output_dir / 'best_model.pth'),
        'per_defect_type': final_metrics.get('per_defect_type', {})
    }
    
    with open(output_dir / 'final_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n评估结果已保存到: {output_dir / 'final_evaluation.json'}")


if __name__ == '__main__':
    main()
