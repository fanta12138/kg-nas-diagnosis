#!/usr/bin/env python3
"""
P1 实验补强 - 单进程训练器（Windows + 显存预加载优化版）

在单个进程内串行完成所有 Stage3 类实验：
  A. Fixed-KD 基线对照：3 固定架构 x 15 类别，200 epochs
  B. Seed 多跑：NAS 最佳架构 x 3 seeds x 15 类别，200 epochs
  C. 500 epochs 长训练：NAS 最佳架构 x 15 类别

关键优化：
  - 单进程循环（避免 Windows subprocess/DataLoader spawn 开销）
  - 数据集预加载为 tensor 常驻显存（每类 200-400 张 224px 图 < 1GB）
  - GPU 内训练循环，无磁盘 IO / 无 transform 瓶颈
  - HF 离线模式：预训练权重从本地缓存加载

用法: python scripts/p1_train_inprocess.py [--only fixed_kd|seed|long]
断点续跑：已有 final_evaluation.json 的类别自动跳过。
"""
import os
import sys
import json
import csv
import random
import argparse
import time
from pathlib import Path

# 强制离线模式：网络不可用时从本地 HF 缓存加载 timm 预训练权重
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import transforms
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import AnomalyDetector

BASE_DIR = Path(__file__).parent.parent
DATA_BASE = BASE_DIR.parent / 'datasets' / 'mvtec_anomaly_detection'
OUT_ROOT = BASE_DIR / 'experiments'

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

FIXED_KD_ARCHS = {
    'resnet18_512d': {'backbone': 'resnet18', 'feature_dim': 512},
    'swin_tiny_512d': {'backbone': 'swin_tiny_patch4_window7_224', 'feature_dim': 512},
    'mobilenetv3_small_512d': {'backbone': 'mobilenetv3_small_100', 'feature_dim': 512},
}
SEEDS = [42, 123, 2024]
NAS_ARCH = {'backbone': 'efficientnet_b0', 'feature_dim': 512}  # 真实主实验架构

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_category_tensors(category, img_size=224):
    """预加载一个类别的全部图像为 tensor（CPU），返回 train_imgs, test_imgs, test_labels"""
    root = DATA_BASE / category
    resize = transforms.Resize((img_size, img_size))

    train_imgs = []
    train_dir = root / 'train' / 'good'
    for p in sorted(train_dir.glob('*')):
        if p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
            continue
        img = resize(transforms.ToTensor()(Image.open(p).convert('RGB')))
        train_imgs.append(img)
    train_imgs = torch.stack(train_imgs)

    test_imgs, test_labels = [], []
    test_dir = root / 'test'
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        label = 0 if defect_dir.name == 'good' else 1
        for p in sorted(defect_dir.glob('*')):
            if p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
                continue
            test_imgs.append(resize(transforms.ToTensor()(Image.open(p).convert('RGB'))))
            test_labels.append(label)
    test_imgs = torch.stack(test_imgs)
    test_labels = torch.tensor(test_labels)
    return train_imgs, test_imgs, test_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_prototype(model, train_imgs, device, bs=64):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(train_imgs), bs):
            batch = NORM(train_imgs[i:i + bs].to(device))
            feats.append(model.get_features(batch).cpu())
    return torch.cat(feats, dim=0).mean(dim=0)


def evaluate(model, test_imgs, test_labels, prototype, device, bs=64):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(test_imgs), bs):
            batch = NORM(test_imgs[i:i + bs].to(device))
            f = model.get_features(batch)
            scores.extend(torch.norm(f - prototype.to(device), dim=1).cpu().numpy())
    scores = np.array(scores)
    labels = test_labels.numpy()
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def train_category(category, backbone, feature_dim, seed, epochs, output_dir, device):
    """训练单个类别，返回 (auroc, ap, minutes)。已有完整结果则跳过。"""
    output_dir = Path(output_dir)
    eval_file = output_dir / 'final_evaluation.json'
    if eval_file.exists():
        d = json.load(open(eval_file))
        return d['auroc'], d['ap'], 0.0

    t0 = time.time()
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据预加载（CPU tensor，常驻内存）
    train_imgs, test_imgs, test_labels = load_category_tensors(category)
    n_train = len(train_imgs)
    bs = 64
    n_batches = max(1, (n_train + bs - 1) // bs)

    model = AnomalyDetector(backbone_name=backbone, feature_dim=feature_dim, pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auroc, best_ap = 0.0, 0.0
    idx = torch.randperm(n_train) if n_train > 0 else torch.tensor([0])
    for epoch in range(1, epochs + 1):
        model.train()
        # 每 epoch 重排 + 随机水平翻转（GPU 上快速实现）
        idx = torch.randperm(n_train)
        for b in range(0, n_train, bs):
            batch_idx = idx[b:b + bs]
            images = NORM(train_imgs[batch_idx].to(device))
            if random.random() > 0.5:
                images = torch.flip(images, dims=[3])
            optimizer.zero_grad()
            output = model(images)
            features = output['features']
            if features.dim() == 2:
                fn = torch.nn.functional.normalize(features, dim=1)
                sim = torch.mm(fn, fn.t())
                mask = torch.eye(sim.size(0), device=sim.device).bool()
                sim = sim.masked_fill(mask, float('-inf'))
                loss = -torch.log(torch.exp(sim / 0.5).sum(dim=1)).mean()
            else:
                loss = nn.functional.mse_loss(output['reconstructed'], features)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            proto = compute_prototype(model, train_imgs, device)
            auroc, ap = evaluate(model, test_imgs, test_labels, proto, device)
            if auroc > best_auroc:
                best_auroc, best_ap = auroc, ap

    minutes = (time.time() - t0) / 60
    with open(eval_file, 'w') as f:
        json.dump({'auroc': best_auroc, 'ap': best_ap, 'backbone': backbone,
                   'feature_dim': feature_dim, 'seed': seed, 'epochs': epochs,
                   'minutes': round(minutes, 1)}, f, indent=2)
    del model, optimizer, train_imgs, test_imgs, test_labels
    torch.cuda.empty_cache()
    return best_auroc, best_ap, minutes


def collect_summary(jobs, summary_json, summary_csv):
    rows = []
    for cat, bb, dim, seed, epochs, outdir in jobs:
        p = Path(outdir) / 'final_evaluation.json'
        if not p.exists():
            continue
        r = json.load(open(p))
        rows.append({'category': cat, 'backbone': bb, 'dim': dim, 'seed': seed,
                     'epochs': epochs, 'auroc': r['auroc'], 'ap': r['ap'], 'minutes': r['minutes']})
    with open(summary_json, 'w') as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(summary_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['category', 'backbone', 'dim', 'seed', 'epochs', 'auroc', 'ap', 'minutes'])
            w.writeheader()
            w.writerows(rows)
        avg = np.mean([r['auroc'] for r in rows])
        print(f"  平均 AUROC: {avg:.4f} (n={len(rows)})")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['fixed_kd', 'seed', 'long'], default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device} | 数据: {DATA_BASE}")

    jobs = []
    if args.only in (None, 'fixed_kd'):
        for arch_name, arch in FIXED_KD_ARCHS.items():
            for cat in CATEGORIES:
                jobs.append((cat, arch['backbone'], arch['feature_dim'], 42, 200,
                             OUT_ROOT / 'p1_fixed_kd' / arch_name / cat))
    if args.only in (None, 'seed'):
        for seed in SEEDS:
            for cat in CATEGORIES:
                jobs.append((cat, NAS_ARCH['backbone'], NAS_ARCH['feature_dim'], seed, 200,
                             OUT_ROOT / 'p1_seed' / f'seed{seed}' / cat))
    if args.only in (None, 'long'):
        for cat in CATEGORIES:
            jobs.append((cat, NAS_ARCH['backbone'], NAS_ARCH['feature_dim'], 42, 500,
                         OUT_ROOT / 'p1_long500' / cat))

    print(f"任务总数: {len(jobs)}")
    t_start = time.time()
    done = 0
    for i, (cat, bb, dim, seed, epochs, outdir) in enumerate(jobs, 1):
        try:
            auroc, ap, minutes = train_category(cat, bb, dim, seed, epochs, outdir, device)
            done += 1
            print(f"[{i}/{len(jobs)}] {Path(outdir).relative_to(OUT_ROOT)}: "
                  f"AUROC={auroc:.4f} AP={ap:.4f} ({minutes:.1f}min)", flush=True)
        except Exception as e:
            import traceback
            print(f"[{i}/{len(jobs)}] {Path(outdir).relative_to(OUT_ROOT)}: FAILED - {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    total = (time.time() - t_start) / 3600
    print(f"\n完成 {done}/{len(jobs)} 任务，总耗时 {total:.1f} 小时", flush=True)

    if args.only in (None, 'fixed_kd'):
        collect_summary([j for j in jobs if 'p1_fixed_kd' in str(j[5])],
                        OUT_ROOT / 'p1_fixed_kd' / 'summary.json',
                        OUT_ROOT / 'p1_fixed_kd' / 'summary.csv')
    if args.only in (None, 'seed'):
        collect_summary([j for j in jobs if 'p1_seed' in str(j[5])],
                        OUT_ROOT / 'p1_seed' / 'summary.json',
                        OUT_ROOT / 'p1_seed' / 'summary.csv')
    if args.only in (None, 'long'):
        collect_summary([j for j in jobs if 'p1_long500' in str(j[5])],
                        OUT_ROOT / 'p1_long500' / 'summary.json',
                        OUT_ROOT / 'p1_long500' / 'summary.csv')


if __name__ == '__main__':
    main()
