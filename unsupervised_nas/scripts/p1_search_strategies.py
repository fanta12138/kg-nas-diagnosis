#!/usr/bin/env python3
"""
P1-D 搜索策略对比实验

在 3 个代表性类别（wood, hazelnut, capsule）上对比三种搜索策略：
  - Random (baseline，已有 100 trials 结果)
  - Evolutionary (μ+λ 进化搜索，30 trials)
  - Bayesian (简化 TPE，30 trials)

每个策略使用相同的评估协议（quick_epochs=3，同一教师模型），
输出到 experiments/p1_search_strategies/{category}/{strategy}/results.json

用法: python scripts/p1_search_strategies.py [--categories wood hazelnut capsule] [--trials 30]
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

# 强制离线模式：从本地 HF 缓存加载 timm 预训练权重
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# 修复 Windows 管道输出编码问题（GBK 无法编码 ✓⚠ 等 Unicode 字符）
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ContrastiveModel
from methods.nas_engine import NASSearchEngine
from datasets import MVTecWoodDataset

BASE_DIR = Path(__file__).parent.parent
DATA_BASE = BASE_DIR.parent / 'datasets' / 'mvtec_anomaly_detection'
TEACHER_PATH = BASE_DIR / 'experiments' / 'stage1' / 'best_teacher.pth'
SEARCH_SPACE = {
    'backbones': ['convnextv2_tiny', 'convnextv2_base', 'mobileone_s0', 'convnext_tiny',
                  'convnext_small', 'vit_tiny_patch16_224', 'deit_tiny_patch16_224',
                  'swin_tiny_patch4_window7_224', 'efficientnet_b0', 'efficientnet_b1',
                  'mobilenetv3_small_100', 'resnet18', 'resnet34', 'resnet50'],
    'feature_dims': [128, 256, 512],
}


def build_loaders(data_dir, batch_size=16, img_size=224, num_workers=0):
    # num_workers=0: Windows spawn 开销巨大，主进程加载更快
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = MVTecWoodDataset(root=data_dir, split='train', defect_type='good',
                                     transform=val_transform, return_pair=False)
    val_dataset = MVTecWoodDataset(root=data_dir, split='test', defect_type='good',
                                   transform=val_transform, return_pair=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def load_teacher(device):
    # 真实实验的教师模型：resnet18 + projection 128（见 experiments/stage1/config.yaml）
    # 注：论文正文写 ConvNeXtV2-Tiny，与实验事实不符，P2 需修正
    teacher = ContrastiveModel(backbone_name='resnet18', projection_dim=128, pretrained=False).to(device)
    ckpt = torch.load(TEACHER_PATH, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    # 处理可能的键前缀差异
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    return teacher


class EvolutionarySearch(NASSearchEngine):
    """(μ+λ) 进化搜索"""

    def search(self, train_loader, val_loader, num_trials=30, quick_epochs=3):
        backbones = self.search_space['backbones']
        dims = self.search_space['feature_dims']
        pop_size = 5
        evaluated = {}

        def arch_key(a):
            return f"{a['backbone']}_{a['feature_dim']}"

        def eval_arch(a):
            k = arch_key(a)
            if k in evaluated:
                return evaluated[k]
            score = self.evaluate_architecture(a, train_loader, val_loader, quick_epochs)
            evaluated[k] = score
            self.results.append({'trial': len(self.results) + 1, 'architecture': a, 'score': score})
            return score

        # 初始化种群（5 个随机架构）
        pop = [self.sample_architecture() for _ in range(pop_size)]
        pop_scores = [eval_arch(a) for a in pop]

        # 变异
        def mutate(a):
            child = dict(a)
            if random.random() < 0.6:
                child['backbone'] = random.choice(backbones)
            if random.random() < 0.6:
                child['feature_dim'] = random.choice(dims)
            return child

        trial_count = len(self.results)
        print(f"  进化搜索初始化种群完成 ({trial_count} evaluations)")

        # (μ+λ): 每代生成 offspring，保留全局精英
        while len(self.results) < num_trials:
            # 精英选择（分数越低越好）
            sorted_pop = sorted(zip(pop, pop_scores), key=lambda x: x[1])
            elite = sorted_pop[0][0]
            parent = random.choice(sorted_pop[:3])[0]
            child = mutate(parent)
            score = eval_arch(child)
            pop.append(child)
            pop_scores.append(score)
            # 保持种群大小
            if len(pop) > pop_size:
                sorted_all = sorted(zip(pop, pop_scores), key=lambda x: x[1])
                pop = [x[0] for x in sorted_all[:pop_size]]
                pop_scores = [x[1] for x in sorted_all[:pop_size]]
            print(f"  [Evo {len(self.results)}/{num_trials}] {arch_key(child)}: {score:.4f}")

        best = min(self.results, key=lambda x: x['score'])
        print(f"  进化搜索最佳: {best['architecture']} score={best['score']:.4f}")
        return best['architecture']


class BayesianSearch(NASSearchEngine):
    """简化 TPE (Tree-structured Parzen Estimator) 搜索

    将已评估的配置按分数分为 top-25% (good) 和其余 (bad)，
    新候选按 good/bad 的频率比优先采样。
    """

    def search(self, train_loader, val_loader, num_trials=30, quick_epochs=3):
        backbones = self.search_space['backbones']
        dims = self.search_space['feature_dims']
        evaluated = {}

        def arch_key(a):
            return f"{a['backbone']}_{a['feature_dim']}"

        def eval_arch(a):
            k = arch_key(a)
            if k in evaluated:
                return evaluated[k]
            score = self.evaluate_architecture(a, train_loader, val_loader, quick_epochs)
            evaluated[k] = score
            self.results.append({'trial': len(self.results) + 1, 'architecture': a, 'score': score})
            return score

        # 前 8 个随机初始化
        init = [self.sample_architecture() for _ in range(8)]
        for a in init:
            eval_arch(a)
        print(f"  TPE 初始化完成 ({len(self.results)} evaluations)")

        while len(self.results) < num_trials:
            results = self.results
            scores = np.array([r['score'] for r in results])
            gamma = max(3, int(len(results) * 0.25))  # top-25% 为 good
            threshold = np.partition(scores, gamma - 1)[gamma - 1]
            good = [r['architecture'] for r in results if r['score'] <= threshold]
            bad = [r['architecture'] for r in results if r['score'] > threshold]

            # 基于 good 分布采样（每维独立频率，拉普拉斯平滑）
            def sample_with_prior():
                def pick(choices, prior_pool):
                    from collections import Counter
                    # 注意：prior_pool 元素可能是 str(backbone) 或 int(feature_dim)
                    cnt = Counter(prior_pool)
                    # l(x) = (count_good + 1) / (total_good + num_choices)
                    weights = []
                    for c in choices:
                        w = (cnt.get(c, 0) + 1) / (len(prior_pool) + len(choices))
                        weights.append(w)
                    return random.choices(choices, weights=weights)[0]

                gb = [g['backbone'] for g in good]
                gd = [g['feature_dim'] for g in good]
                bb = pick(backbones, gb)
                dd = pick(dims, gd)
                return {'backbone': bb, 'feature_dim': dd}

            # 候选生成：TPE 采样或随机探索（50/50）
            candidate = None
            attempts = 0
            while candidate is None or arch_key(candidate) in evaluated:
                if random.random() < 0.7:
                    candidate = sample_with_prior()
                else:
                    candidate = self.sample_architecture()
                attempts += 1
                if attempts > 20:
                    break
            if candidate is None or arch_key(candidate) in evaluated:
                # 穷举未评估配置兜底
                for bb in backbones:
                    for dd in dims:
                        if f"{bb}_{dd}" not in evaluated:
                            candidate = {'backbone': bb, 'feature_dim': dd}
                            break
                    if candidate:
                        break
            if candidate is None:
                break

            score = eval_arch(candidate)
            print(f"  [TPE {len(self.results)}/{num_trials}] {arch_key(candidate)}: {score:.4f}")

        best = min(self.results, key=lambda x: x['score'])
        print(f"  TPE 最佳: {best['architecture']} score={best['score']:.4f}")
        return best['architecture']


def run_strategy(category, strategy, num_trials, device):
    out_dir = BASE_DIR / 'experiments' / 'p1_search_strategies' / category / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    # 断点续跑：已有完整结果则跳过
    if (out_dir / 'results.json').exists():
        with open(out_dir / 'results.json') as f:
            prev = json.load(f)
        print(f"[skip] {category}/{strategy}: 已有结果 ({prev.get('num_trials')} trials)")
        return prev.get('all_results', [])
    print(f"\n{'=' * 60}\n[{category}] {strategy} 搜索 ({num_trials} trials)\n{'=' * 60}")

    train_loader, val_loader = build_loaders(DATA_BASE / category)
    teacher = load_teacher(device)

    if strategy == 'evolutionary':
        engine = EvolutionarySearch(SEARCH_SPACE, teacher, device)
    elif strategy == 'bayesian':
        engine = BayesianSearch(SEARCH_SPACE, teacher, device)
    else:
        engine = NASSearchEngine(SEARCH_SPACE, teacher, device)

    best = engine.search(train_loader, val_loader, num_trials=num_trials, quick_epochs=3)

    engine.save_results(str(out_dir / 'results.json'))
    with open(out_dir / 'best.json', 'w') as f:
        json.dump({'best_architecture': best,
                   'num_trials': len(engine.results),
                   'best_score': min(r['score'] for r in engine.results)}, f, indent=2)
    return engine.get_all_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', default=['wood', 'hazelnut', 'capsule'])
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--strategies', nargs='+', default=['random', 'evolutionary', 'bayesian'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"教师模型: {TEACHER_PATH}")
    print(f"类别: {args.categories}, 策略: {args.strategies}, trials: {args.trials}")

    summary = {}
    for cat in args.categories:
        for strat in args.strategies:
            results = run_strategy(cat, strat, args.trials, device)
            best_score = min(r['score'] for r in results)
            summary[f"{cat}/{strat}"] = best_score
            print(f"[{cat}/{strat}] best={best_score:.4f}")

    with open(BASE_DIR / 'experiments' / 'p1_search_strategies' / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n全部完成:", json.dumps(summary, indent=2))
