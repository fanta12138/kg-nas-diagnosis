"""
NAS搜索引擎
实现知识引导的神经架构搜索
"""

import torch
import torch.nn as nn
import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# 添加项目根目录到路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import AnomalyDetector, ContrastiveModel
from utils import AverageMeter


class NASSearchEngine:
    """
    神经架构搜索引擎
    
    使用知识蒸馏和一致性正则化评估架构
    """
    
    def __init__(
        self,
        search_space: Dict,
        teacher_model: ContrastiveModel,
        device: torch.device,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2
    ):
        """
        Args:
            search_space: 搜索空间配置
            teacher_model: 预训练的教师模型
            device: 计算设备
            alpha: KD损失权重
            beta: 一致性损失权重
            gamma: 重构损失权重
        """
        self.search_space = search_space
        self.teacher = teacher_model
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        self.results = []
    
    def sample_architecture(self) -> Dict:
        """随机采样一个架构配置"""
        arch = {
            'backbone': random.choice(self.search_space['backbones']),
            'feature_dim': random.choice(self.search_space['feature_dims']),
        }
        return arch
    
    def evaluate_architecture(
        self,
        arch_config: Dict,
        train_loader,
        val_loader,
        quick_epochs: int = 5
    ) -> float:
        """
        评估一个架构
        
        Args:
            arch_config: 架构配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            quick_epochs: 快速训练轮数
        
        Returns:
            score: 综合评分（越低越好）
        """
        # 构建学生模型（使用ContrastiveModel）
        from models import ContrastiveModel
        
        # 使用arch_config中指定的feature_dim
        student = ContrastiveModel(
            backbone_name=arch_config['backbone'],
            projection_dim=arch_config['feature_dim'],  # 使用配置的feature_dim
            pretrained=True
        ).to(self.device)
        
        # 获取教师模型的输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            teacher_output_dim = self.teacher(dummy_input).shape[1]
        
        # 如果学生投影维度与教师不同，添加对齐层
        student_output_dim = arch_config['feature_dim']
        if student_output_dim != teacher_output_dim:
            print(f"  添加维度对齐层: {student_output_dim} -> {teacher_output_dim}")
            align_layer = nn.Linear(student_output_dim, teacher_output_dim).to(self.device)
        else:
            align_layer = None
        
        # 快速训练
        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
        if align_layer is not None:
            optimizer.add_param_group({'params': align_layer.parameters()})
        
        for epoch in range(quick_epochs):
            # 使用eval模式避免BatchNorm在small batch时的问题
            student.eval()
            if align_layer is not None:
                align_layer.eval()
            
            for images, _ in train_loader:
                images = images.to(self.device)
                
                # 如果batch size为1，跳过（避免BatchNorm问题）
                if images.size(0) <= 1:
                    continue
                
                optimizer.zero_grad()
                
                # 前向传播
                features = student.get_features(images)
                z = student.projector(features)
                
                # 知识蒸馏损失（在投影空间比较）
                with torch.no_grad():
                    teacher_z = self.teacher(images)
                
                # 使用对齐层（如果需要）
                if align_layer is not None:
                    z_aligned = align_layer(z)
                    kd_loss = nn.functional.mse_loss(z_aligned, teacher_z)
                else:
                    kd_loss = nn.functional.mse_loss(z, teacher_z)
                
                loss = kd_loss
                loss.backward()
                optimizer.step()
        
        # 在验证集上评估
        student.eval()
        if align_layer is not None:
            align_layer.eval()
        
        total_score = 0
        count = 0
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(self.device)
                
                # 知识蒸馏损失
                student_z = student(images)
                teacher_z = self.teacher(images)
                
                # 使用对齐层（如果需要）
                if align_layer is not None:
                    student_z_aligned = align_layer(student_z)
                    kd_loss = nn.functional.mse_loss(student_z_aligned, teacher_z)
                else:
                    kd_loss = nn.functional.mse_loss(student_z, teacher_z)
                
                # 一致性损失（两个增强视图）
                aug1 = self._augment(images)
                aug2 = self._augment(images)
                z1 = student(aug1)
                z2 = student(aug2)
                
                # 一致性损失也需要对齐
                if align_layer is not None:
                    z1_aligned = align_layer(z1)
                    z2_aligned = align_layer(z2)
                    consistency_loss = nn.functional.mse_loss(z1_aligned, z2_aligned)
                else:
                    consistency_loss = nn.functional.mse_loss(z1, z2)
                
                # 综合评分（添加数值稳定性检查）
                score = kd_loss + 0.3 * consistency_loss
                
                # 检查数值有效性（不仅检查finite，还要检查合理性）
                score_val = score.item()
                if torch.isfinite(score) and score_val < 1000:  # 设置合理阈值
                    total_score += score_val
                    count += 1
                else:
                    print(f"    ⚠️  Warning: Invalid score detected (score={score_val:.2e}, kd={kd_loss.item():.4f}), skipping")
        
        avg_score = total_score / max(count, 1)
        
        # 最终检查
        if not torch.isfinite(torch.tensor(avg_score)):
            print(f"    ⚠️  Warning: Final score is non-finite ({avg_score}), setting to large penalty")
            avg_score = 1e6  # 给予大惩罚值
        
        return avg_score
    
    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        """简单的数据增强（翻转）"""
        if random.random() > 0.5:
            images = torch.flip(images, dims=[3])  # 水平翻转
        if random.random() > 0.5:
            images = torch.flip(images, dims=[2])  # 垂直翻转
        return images
    
    def search(
        self,
        train_loader,
        val_loader,
        num_trials: int = 30,
        quick_epochs: int = 5
    ) -> Dict:
        """
        执行架构搜索
        
        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            num_trials: 搜索次数
            quick_epochs: 每个架构的快速训练轮数
        
        Returns:
            best_architecture: 最佳架构配置
        """
        print(f"\n开始NAS搜索，trials={num_trials}")
        print(f"搜索空间:")
        print(f"  Backbones: {self.search_space['backbones']}")
        print(f"  Feature dims: {self.search_space['feature_dims']}")
        print()
        
        pbar = tqdm(range(num_trials), desc="NAS Search")
        
        for i in pbar:
            # 采样架构
            arch = self.sample_architecture()
            
            # 评估
            try:
                score = self.evaluate_architecture(
                    arch, train_loader, val_loader, quick_epochs
                )
            except Exception as e:
                print(f"Trial {i+1} failed: {e}")
                continue
            
            self.results.append({
                'trial': i + 1,
                'architecture': arch,
                'score': score
            })
            
            # 更新进度条
            pbar.set_postfix({'score': f'{score:.4f}'})
            
            # 打印当前最佳
            if len(self.results) > 0:
                best = min(self.results, key=lambda x: x['score'])
                pbar.set_description(f"Best: {best['score']:.4f}")
        
        if len(self.results) == 0:
            raise RuntimeError("No successful trials")
        
        # 找到最佳架构
        best = min(self.results, key=lambda x: x['score'])
        
        print(f"\n{'='*60}")
        print(f"搜索完成！")
        print(f"{'='*60}")
        print(f"最佳架构:")
        print(f"  Backbone: {best['architecture']['backbone']}")
        print(f"  Feature Dim: {best['architecture']['feature_dim']}")
        print(f"  Score: {best['score']:.4f}")
        print(f"{'='*60}\n")
        
        return best['architecture']
    
    def get_all_results(self) -> List[Dict]:
        """获取所有搜索结果"""
        return self.results
    
    def save_results(self, save_path: str):
        """保存搜索结果"""
        import json
        from pathlib import Path
        
        results_data = {
            'best_architecture': min(self.results, key=lambda x: x['score'])['architecture'],
            'all_results': self.results,
            'num_trials': len(self.results)
        }
        
        save_path = Path(save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"搜索结果已保存到: {save_path}")
