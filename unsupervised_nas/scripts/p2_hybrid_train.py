#!/usr/bin/env python3
"""
P2 双范式融合检测（Hybrid RD-Prototype）

训练与 RD 相同（学生重建教师多尺度特征），推理时融合两种互补分数：
  - RD 分数：教师-学生多尺度特征差异（对纹理/结构缺陷强）
  - 原型分数：bottleneck 特征与正常原型的距离（对螺纹类缺陷强，如 screw）

融合: score = (rd - μ_rd)/σ_rd + w × (proto - μ_p)/σ_p
μ/σ 用训练集统计，w 为融合权重（脚本会扫描多个 w 并报告各 AUROC）。

用法: python scripts/p2_hybrid_train.py [--epochs 200] [--category wood] [--weights 0.5 1.0 2.0]
"""
import os
import sys
import json
import random
import argparse
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

import torch
from torch.optim import AdamW
from torchvision import transforms
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.rd_detector import RDDetector

BASE_DIR = Path(__file__).parent.parent
DATA_BASE = BASE_DIR.parent / 'datasets' / 'mvtec_anomaly_detection'
OUT_ROOT = BASE_DIR / 'experiments' / 'p2_hybrid'

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
NAS_ARCH = 'efficientnet_b0'

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_category_tensors(category, img_size=224):
    root = DATA_BASE / category
    resize = transforms.Resize((img_size, img_size))
    train_imgs = []
    for p in sorted((root / 'train' / 'good').glob('*')):
        if p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
            continue
        train_imgs.append(resize(transforms.ToTensor()(Image.open(p).convert('RGB'))))
    test_imgs, test_labels = [], []
    for defect_dir in sorted((root / 'test').iterdir()):
        if not defect_dir.is_dir():
            continue
        label = 0 if defect_dir.name == 'good' else 1
        for p in sorted(defect_dir.glob('*')):
            if p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
                continue
            test_imgs.append(resize(transforms.ToTensor()(Image.open(p).convert('RGB'))))
            test_labels.append(label)
    return torch.stack(train_imgs), torch.stack(test_imgs), torch.tensor(test_labels)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_scores(model, imgs, device, bs=64):
    """返回 rd_scores 与 proto_scores（未标准化）"""
    rd, proto = [], []
    for i in range(0, len(imgs), bs):
        x = NORM(imgs[i:i + bs].to(device))
        rd.extend(model.anomaly_score(x).cpu().numpy())
        z = model.encoder(x)
        z_pool = z.reshape(z.shape[0], z.shape[1], -1).mean(dim=2)
        proto.append(z_pool.cpu())
    proto = torch.cat(proto, dim=0)
    return np.array(rd), proto


def train_category(category, epochs, seed, output_dir, device, weights=(1.0,)):
    output_dir = Path(output_dir)
    eval_file = output_dir / 'final_evaluation.json'
    if eval_file.exists():
        return json.load(open(eval_file))

    t0 = time.time()
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_imgs, test_imgs, test_labels = load_category_tensors(category)
    n_train = len(train_imgs)
    bs = 64

    model = RDDetector(student_backbone=NAS_ARCH).to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n_train)
        for b in range(0, n_train, bs):
            images = NORM(train_imgs[idx[b:b + bs]].to(device))
            if random.random() > 0.5:
                images = torch.flip(images, dims=[3])
            optimizer.zero_grad()
            loss = model.training_loss(images)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # ---- 训练后统计：原型中心 + 训练集分数分布 ----
    model.eval()
    train_rd, train_proto = compute_scores(model, train_imgs, device)
    prototype = train_proto.mean(dim=0).to(device)

    def get_scores(imgs):
        rd, proto = compute_scores(model, imgs, device)
        p = torch.norm(proto - prototype.cpu(), dim=1).numpy()
        return rd, p

    tr_rd, tr_p = get_scores(train_imgs)
    mu_rd, sd_rd = tr_rd.mean(), tr_rd.std() + 1e-8
    mu_p, sd_p = tr_p.mean(), tr_p.std() + 1e-8
    te_rd, te_p = get_scores(test_imgs)
    labels = test_labels.numpy()

    results = {}
    for w in weights:
        fused = (te_rd - mu_rd) / sd_rd + w * (te_p - mu_p) / sd_p
        if len(np.unique(labels)) < 2:
            results[f'w{w}'] = {'auroc': 0.0, 'ap': 0.0}
        else:
            results[f'w{w}'] = {
                'auroc': float(roc_auc_score(labels, fused)),
                'ap': float(average_precision_score(labels, fused)),
            }
    # 单范式对照
    results['rd_only'] = {'auroc': float(roc_auc_score(labels, te_rd)),
                          'ap': float(average_precision_score(labels, te_rd))}
    results['proto_only'] = {'auroc': float(roc_auc_score(labels, te_p)),
                             'ap': float(average_precision_score(labels, te_p))}

    minutes = (time.time() - t0) / 60
    results['minutes'] = round(minutes, 1)
    with open(eval_file, 'w') as f:
        json.dump(results, f, indent=2)
    del model, train_imgs, test_imgs
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--category', nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weights', nargs='+', type=float, default=[0.5, 1.0, 2.0])
    args = parser.parse_args()

    cats = args.category or CATEGORIES
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device} | 类别: {cats} | epochs: {args.epochs} | weights: {args.weights}")

    all_rows = {}
    for i, c in enumerate(cats, 1):
        outdir = OUT_ROOT / f'ep{args.epochs}' / c
        try:
            r = train_category(c, args.epochs, args.seed, outdir, device, args.weights)
            all_rows[c] = r
            print(f"[{i}/{len(cats)}] {c}: rd_only={r['rd_only']['auroc']:.4f} "
                  f"best_w={max(r, key=lambda k: r[k]['auroc'] if isinstance(r[k], dict) else -1)}", flush=True)
        except Exception as e:
            import traceback
            print(f"[{i}/{len(cats)}] {c}: FAILED - {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    # 汇总各权重下的平均
    print("\n=== 汇总 ===")
    for key in ['rd_only', 'proto_only'] + [f'w{w}' for w in args.weights]:
        vals = [all_rows[c][key]['auroc'] for c in all_rows if key in all_rows[c]]
        if vals:
            print(f"{key:12s}: 平均 AUROC = {np.mean(vals):.4f}")

    with open(OUT_ROOT / f'summary_ep{args.epochs}.json', 'w') as f:
        json.dump(all_rows, f, indent=2)


if __name__ == '__main__':
    main()
