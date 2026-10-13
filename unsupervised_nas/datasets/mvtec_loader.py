"""MVTec-AD Wood数据集加载器"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MVTecWoodDataset(Dataset):
    """MVTec-AD Wood数据集 - 用于对比学习和训练"""
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        defect_type: str = 'good',
        transform: Optional[Callable] = None,
        return_pair: bool = False
    ):
        """
        Args:
            root: 数据集根目录
            split: 'train' 或 'test'
            defect_type: 缺陷类型，'good'表示正常样本
            transform: 图像变换
            return_pair: 是否返回两个增强视图（用于对比学习）
        """
        self.root = Path(root)
        self.split = split
        self.defect_type = defect_type
        self.transform = transform
        self.return_pair = return_pair
        
        # 加载图像路径
        self.image_paths = self._load_images()
        
    def _load_images(self) -> List[Path]:
        """加载图像路径"""
        img_dir = self.root / self.split / self.defect_type
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Directory not found: {img_dir}")
        
        extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        images = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in extensions
        ])
        
        if len(images) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        return images
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple:
        """获取样本"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.return_pair:
            # 返回两个增强视图（用于对比学习）
            if self.transform:
                view1 = self.transform(image)
                view2 = self.transform(image)
            else:
                view1 = view2 = torch.tensor(np.array(image))
            return view1, view2
        else:
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.tensor(np.array(image))
            # 标签：good=0, 其他=1
            label = 0 if self.defect_type == 'good' else 1
            return image, label


class MVTecTestDataset(Dataset):
    """MVTec-AD测试集（包含正常和异常样本）"""
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            root: 数据集根目录
            transform: 图像变换
        """
        self.root = Path(root)
        self.transform = transform
        
        # 加载所有测试样本
        self.samples = self._load_all_test_samples()
        
    def _load_all_test_samples(self) -> List[Dict]:
        """加载所有测试样本"""
        test_dir = self.root / 'test'
        samples = []
        
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        for defect_dir in sorted(test_dir.iterdir()):
            if defect_dir.is_dir():
                defect_type = defect_dir.name
                for img_path in sorted(defect_dir.glob('*.png')):
                    # 标准定义: good=0, anomaly=1
                    is_anomaly = 0 if defect_type == 'good' else 1
                    samples.append({
                        'path': img_path,
                        'defect_type': defect_type,
                        'is_anomaly': is_anomaly  # 改名更清晰
                    })
        
        if len(samples) == 0:
            raise ValueError("No test samples found")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image))
        
        return image, sample['is_anomaly'], sample['defect_type']
    
    def get_defect_types(self) -> List[str]:
        """获取所有缺陷类型"""
        return list(set(s['defect_type'] for s in self.samples))
