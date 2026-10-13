#!/usr/bin/env python3
"""
P2: RD-KG-NAS 训练（Reverse Distillation 检测范式）

教师（冻结 resnet18 多尺度特征）→ 学生（NAS encoder + bottleneck + decoder）重建
推理用教师-学生差异做异常分数。

用法: python scripts/p2_rd_train.py [--only nas|fixed_kd] [--epochs 200] [--category wood]
断点续跑：已有 final_evaluation.json 自动跳过。
"""
import os
import sys
import json
import csv
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
OUT_ROOT = BASE_DIR / 'experiments' / 'p2_rd'

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

# NAS 发现的学生架构 + Fixed-KD 对照（与 P1 一致）
NAS_ARCH = {'backbone': 'efficientnet_b0', 'label': 'nas_effb0'}
FIXED_ARCHS = {
    'resnet18': {'backbone': 'resnet18', 'label': 'fixed_resnet18'},
    'swin_tiny': {'backbone': 'swin_tiny_patch4_window7_224', 'label': 'fixed_swin'},
    'mbv3_small': {'backbone': 'mobilenetv3_small_100', 'label': 'fixed_mbv3'},
}

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


def train_category(category, student_backbone, label, epochs, seed, output_dir, device, teacher='resnet18'):
    output_dir = Path(output_dir)
    eval_file = output_dir / 'final_evaluation.json'
    if eval_file.exists():
        d = json.load(open(eval_file))
        return d['auroc'], d['ap'], 0.0

    t0 = time.time()
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_imgs, test_imgs, test_labels = load_category_tensors(category)
    n_train = len(train_imgs)
    bs = 64

    model = RDDetector(student_backbone=student_backbone, teacher_arch=teacher).to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=1e-3, weight_decay=1e-4)
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

        if epoch % 20 == 0 or epoch == epochs:
            auroc, ap = evaluate(model, test_imgs, test_labels, device, bs=64)
            print(f"  [{label}/{category}] epoch {epoch}: loss={loss.item():.4f} "
                  f"AUROC={auroc:.4f} AP={ap:.4f}", flush=True)

    auroc, ap = evaluate(model, test_imgs, test_labels, device, bs=64)
    minutes = (time.time() - t0) / 60
    with open(eval_file, 'w') as f:
        json.dump({'auroc': auroc, 'ap': ap, 'student_backbone': student_backbone,
                   'teacher': teacher, 'epochs': epochs, 'seed': seed,
                   'minutes': round(minutes, 1)}, f, indent=2)
    del model, train_imgs, test_imgs, test_labels
    torch.cuda.empty_cache()
    return auroc, ap, minutes


@torch.no_grad()
def evaluate(model, test_imgs, test_labels, device, bs=64):
    model.eval()
    scores = []
    for i in range(0, len(test_imgs), bs):
        images = NORM(test_imgs[i:i + bs].to(device))
        scores.extend(model.anomaly_score(images).cpu().numpy())
    labels = test_labels.numpy()
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0
    return (float(roc_auc_score(labels, np.array(scores))),
            float(average_precision_score(labels, np.array(scores))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['nas', 'fixed_kd'], default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--category', nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--teacher', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2'])
    args = parser.parse_args()

    cats = args.category or CATEGORIES
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device} | 类别: {cats} | epochs: {args.epochs} | 教师: {args.teacher}")

    teacher_tag = args.teacher
    jobs = []
    if args.only in (None, 'nas'):
        for c in cats:
            jobs.append((c, NAS_ARCH['backbone'], NAS_ARCH['label'], args.epochs,
                         OUT_ROOT / f'teacher_{teacher_tag}' / 'nas' / f'ep{args.epochs}' / c))
    if args.only in (None, 'fixed_kd'):
        for name, a in FIXED_ARCHS.items():
            for c in cats:
                jobs.append((c, a['backbone'], a['label'], args.epochs,
                             OUT_ROOT / f'teacher_{teacher_tag}' / 'fixed' / name / f'ep{args.epochs}' / c))

    print(f"任务数: {len(jobs)}")
    results = []
    for i, (cat, bb, label, epochs, outdir) in enumerate(jobs, 1):
        try:
            auroc, ap, minutes = train_category(cat, bb, label, epochs, args.seed, outdir, device, teacher=args.teacher)
            results.append({'category': cat, 'backbone': bb, 'label': label,
                            'auroc': auroc, 'ap': ap, 'minutes': minutes})
            print(f"[{i}/{len(jobs)}] {label}/{cat}: AUROC={auroc:.4f} AP={ap:.4f} ({minutes:.1f}min)", flush=True)
        except Exception as e:
            import traceback
            print(f"[{i}/{len(jobs)}] {label}/{cat}: FAILED - {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    if results:
        avg = np.mean([r['auroc'] for r in results])
        print(f"\n平均 AUROC: {avg:.4f} (n={len(results)})", flush=True)
        with open(OUT_ROOT / f'teacher_{teacher_tag}' / f'summary_ep{args.epochs}.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
