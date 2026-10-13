#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTAD (BeanTech Anomaly Detection Dataset) 数据加载器

数据集信息:
- 3个工业产品类别: 01, 02, 03
- Product 01: 400训练图(1600x1600), 70测试图(49异常+21正常)
- Product 02: 399训练图(600x600), 230测试图(200异常+30正常)
- Product 03: 1000训练图(800x600), 441测试图(41异常+400正常)
"""
# --- repository bootstrap (added for the public release) ---
import os as _os, sys as _sys
_KGD_P = _os.path.abspath(__file__)
while _KGD_P != _os.path.dirname(_KGD_P) and not _os.path.exists(
        _os.path.join(_KGD_P, 'kgd_paths.py')):
    _KGD_P = _os.path.dirname(_KGD_P)
if _KGD_P not in _sys.path:
    _sys.path.insert(0, _KGD_P)
from kgd_paths import (REPO_ROOT as _KGD_ROOT, DATASETS_ROOT as _KGD_DATA,
                       PAPER_ROOT as _KGD_PAPER)
# --- end bootstrap ------------------------------------------------------------


import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BTADDataset(Dataset):
    """BTAD数据集加载器"""
    
    def __init__(self, root_dir, category='01', split='train', transform=None, target_transform=None):
        """
        Args:
            root_dir: BTAD数据集根目录
            category: 产品类别 ('01', '02', '03')
            split: 数据分割 ('train' 或 'test')
            transform: 图像变换
            target_transform: 标签变换
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # BTAD目录结构
        self.category_dir = self.root_dir / category
        self.split_dir = self.category_dir / split
        
        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []
        
        if split == 'train':
            # 训练集只有正常样本
            ok_dir = self.split_dir / 'ok'
            if ok_dir.exists():
                for img_path in sorted(ok_dir.glob('*.png')) + sorted(ok_dir.glob('*.bmp')):
                    self.image_paths.append(img_path)
                    self.labels.append(0)  # 0表示正常
        else:
            # 测试集包含正常和异常样本
            # 正常样本
            ok_dir = self.split_dir / 'ok'
            if ok_dir.exists():
                for img_path in sorted(ok_dir.glob('*.png')) + sorted(ok_dir.glob('*.bmp')):
                    self.image_paths.append(img_path)
                    self.labels.append(0)  # 0表示正常
            
            # 异常样本
            ko_dir = self.split_dir / 'ko'
            if ko_dir.exists():
                for img_path in sorted(ko_dir.glob('*.png')) + sorted(ko_dir.glob('*.bmp')):
                    self.image_paths.append(img_path)
                    self.labels.append(1)  # 1表示异常
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        # 应用标签变换
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    
    def get_stats(self):
        """获取数据集统计信息"""
        total = len(self)
        normal = sum(1 for l in self.labels if l == 0)
        anomaly = sum(1 for l in self.labels if l == 1)
        
        return {
            'total': total,
            'normal': normal,
            'anomaly': anomaly,
            'anomaly_ratio': anomaly / total if total > 0 else 0
        }


def get_btad_transforms(image_size=256):
    """
    获取BTAD数据集的图像变换
    
    Args:
        image_size: 输出图像尺寸
    
    Returns:
        train_transform, test_transform
    """
    # 训练集变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 测试集变换（无增强）
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def load_btad_category(root_dir, category='01', image_size=256, batch_size=32):
    """
    加载BTAD单个类别的数据
    
    Args:
        root_dir: BTAD数据集根目录
        category: 产品类别 ('01', '02', '03')
        image_size: 图像尺寸
        batch_size: batch大小
    
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    from torch.utils.data import DataLoader
    
    train_transform, test_transform = get_btad_transforms(image_size)
    
    # 创建数据集
    train_dataset = BTADDataset(
        root_dir=root_dir,
        category=category,
        split='train',
        transform=train_transform
    )
    
    test_dataset = BTADDataset(
        root_dir=root_dir,
        category=category,
        split='test',
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # 避免最后一个batch过小
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"BTAD-{category} 数据集统计")
    print(f"{'='*60}")
    
    train_stats = train_dataset.get_stats()
    test_stats = test_dataset.get_stats()
    
    print(f"训练集:")
    print(f"  总样本数: {train_stats['total']}")
    print(f"  正常样本: {train_stats['normal']}")
    print(f"  异常样本: {train_stats['anomaly']}")
    print(f"  异常比例: {train_stats['anomaly_ratio']:.2%}")
    
    print(f"\n测试集:")
    print(f"  总样本数: {test_stats['total']}")
    print(f"  正常样本: {test_stats['normal']}")
    print(f"  异常样本: {test_stats['anomaly']}")
    print(f"  异常比例: {test_stats['anomaly_ratio']:.2%}")
    print(f"{'='*60}\n")
    
    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == '__main__':
    # 测试数据加载器
    btad_root = _os.path.join(_KGD_DATA, 'datasets-3', 'BTech_Dataset_transformed')
    
    for category in ['01', '02', '03']:
        print(f"\n{'#'*60}")
        print(f"# 测试BTAD类别: {category}")
        print(f"{'#'*60}")
        
        train_loader, test_loader, train_dataset, test_dataset = load_btad_category(
            root_dir=btad_root,
            category=category,
            image_size=256,
            batch_size=32
        )
        
        # 验证数据加载
        print(f"\n验证数据加载...")
        for batch_idx, (images, labels, paths) in enumerate(train_loader):
            print(f"Batch {batch_idx}: images shape={images.shape}, labels={labels.unique()}")
            if batch_idx >= 2:
                break
        
        print("\n✅ 数据加载器测试通过！")
