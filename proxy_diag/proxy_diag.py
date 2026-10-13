# -*- coding: utf-8 -*-
"""Proxy reliability diagnostic for architecture search in unsupervised AD.

For every (category, backbone) pair we measure:
  * the KD proxy score that the search uses to rank architectures -- repeated
    R times so its within-configuration variance is observable;
  * the TRUE AUROC of that configuration after the full 500-epoch protocol.

From these we compute, per category, the Spearman rank correlation between the
proxy and the truth, the within-configuration coefficient of variation, and a
variance decomposition of the proxy score into category / architecture /
residual components.

Usage: python proxy_diag.py [--categories ...] [--backbones ...] [--repeats 3]
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

import argparse, io, json, os, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
sys.path.insert(0, NAS)
os.chdir(NAS)

import numpy as np
import torch
import importlib.util
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location('p2', os.path.join(NAS, 'scripts', 'p2_rd_train.py'))
_p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p2)
load_category_tensors = _p2.load_category_tensors
NORM = _p2.NORM
set_seed = _p2.set_seed

from methods.rd_detector import RDDetector
from torch.optim import AdamW

OUT = _os.path.join(_KGD_ROOT, 'proxy_diag')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'proxy_diag.json')

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']
BACKBONES = ['mobilenetv3_small_100', 'efficientnet_b0', 'mobileone_s0',
             'vit_tiny_patch16_224', 'deit_tiny_patch16_224', 'efficientnet_b1',
             'resnet18', 'resnet34', 'resnet50', 'swin_tiny_patch4_window7_224',
             'convnext_tiny', 'convnextv2_tiny', 'convnext_small',
             'convnextv2_base']


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


@torch.no_grad()
def kd_proxy_score(fit_imgs, backbone, device, quick_epochs=3, seed=None):
    """Replicate the search's evaluation signal: short training under the KD
    objective, score = mean distillation loss on held-out normal data."""
    if seed is not None:
        set_seed(seed)
    n = len(fit_imgs)
    n_val = max(1, int(0.2 * n))
    perm = torch.randperm(n)
    tr, va = fit_imgs[perm[n_val:]], fit_imgs[perm[:n_val]]
    model = RDDetector(student_backbone=backbone, teacher_arch='resnet18').to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    bs = 16
    for _ in range(quick_epochs):
        model.train()
        idx = torch.randperm(len(tr))
        for b in range(0, len(tr), bs):
            x = NORM(tr[idx[b:b + bs]].to(device))
            opt.zero_grad()
            loss = model.training_loss(x)
            loss.backward()
            opt.step()
    model.eval()
    tot, cnt = 0.0, 0
    for b in range(0, len(va), bs):
        x = NORM(va[b:b + bs].to(device))
        with torch.no_grad():
            tot += float(model.training_loss(x)) * len(x)
            cnt += len(x)
    del model
    torch.cuda.empty_cache()
    return tot / max(cnt, 1)


def train_full(fit_imgs, backbone, device, epochs=500, seed=42, bs=64):
    set_seed(seed)
    model = RDDetector(student_backbone=backbone, teacher_arch='resnet18').to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(fit_imgs)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for b in range(0, n, bs):
            x = NORM(fit_imgs[idx[b:b + bs]].to(device))
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[3])
            opt.zero_grad()
            loss = model.training_loss(x)
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def auroc_of(model, test_imgs, test_labels, device, bs=64):
    model.eval()
    sc = []
    for i in range(0, len(test_imgs), bs):
        sc.extend(model.anomaly_score(NORM(test_imgs[i:i + bs].to(device))).cpu().numpy())
    lab = np.asarray(test_labels)
    if len(np.unique(lab)) < 2:
        return float('nan')
    return float(roc_auc_score(lab, np.asarray(sc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=CATS)
    ap.add_argument('--backbones', nargs='+', default=BACKBONES)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=500)
    a = ap.parse_args()

    device = torch.device('cuda')
    res = json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}

    for cat in a.categories:
        train_imgs, test_imgs, test_labels = load_category_tensors(cat)
        for bb in a.backbones:
            key = '%s|%s' % (cat, bb)
            if key in res:
                continue
            t0 = time.time()
            # ---- proxy scores (repeated, so variance is observable) ----
            proxies = []
            for r in range(a.repeats):
                try:
                    proxies.append(kd_proxy_score(train_imgs, bb, device, seed=1000 + r))
                except Exception as e:
                    log('  proxy FAIL %s %s: %s' % (cat, bb, str(e)[:60]))
                    proxies.append(float('nan'))
            # ---- true performance ----
            try:
                model = train_full(train_imgs, bb, device, epochs=a.epochs)
                au = auroc_of(model, test_imgs, test_labels, device)
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                log('  train FAIL %s %s: %s' % (cat, bb, str(e)[:60]))
                au = float('nan')
            res[key] = {'category': cat, 'backbone': bb,
                        'proxy': proxies,
                        'proxy_mean': float(np.nanmean(proxies)),
                        'proxy_std': float(np.nanstd(proxies)),
                        'auroc': au,
                        'minutes': round((time.time() - t0) / 60, 2)}
            json.dump(res, open(RESFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
            log('%-11s %-30s proxy=%.4f+-%.4f  AUROC=%.4f  (%.1f min)'
                % (cat, bb, res[key]['proxy_mean'], res[key]['proxy_std'], au, res[key]['minutes']))

    log('done. results in %s' % RESFILE)


if __name__ == '__main__':
    main()
