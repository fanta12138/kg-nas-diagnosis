"""日志和工具函数"""

import json
import time
from pathlib import Path
from typing import Dict, List


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
    
    def log(self, metrics: Dict):
        """记录指标"""
        metrics['timestamp'] = time.time()
        self.history.append(metrics)
    
    def save_history(self, save_path: Path):
        """保存训练历史"""
        save_path = Path(save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存到: {save_path}")


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
