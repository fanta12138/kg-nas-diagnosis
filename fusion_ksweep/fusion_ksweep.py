# -*- coding: utf-8 -*-
"""How sensitive is the fusion gain to the top-k fraction?

The headline comparison uses top-k pooling with k = 1% of the map pixels. That
single choice is what reviewers will question first: is the gain a robust
property of the pooling rule, or an artefact of one hyper-parameter? This script
retrains the standard student per category (the retraining is the necessary part;
querying many poolings on one model is not) and sweeps

    k in {0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%}

for uniform weights, plus the best weighted policy from fusion_policy/, and
reports mean and worst-category AUROC as a function of k. It also reports the
per-image top-k (k = 1 pixel) and full-map mean as the limiting cases.

Usage: python fusion_ksweep.py [--epochs 500] [--seed 42]
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
sys.path.insert(0, NAS)
os.chdir(NAS)

import numpy as np
import torch
import torch.nn.functional as F
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

OUT = _os.path.join(_KGD_ROOT, 'fusion_ksweep')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'fusion_ksweep.json')
SUMFILE = os.path.join(OUT, 'fusion_ksweep_summary.json')

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']
KS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
W_UNIFORM = (1 / 3, 1 / 3, 1 / 3)
W_BEST = (0.3, 0.6, 0.1)          # best weighted policy found by fusion_policy/


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def train_student(imgs, backbone, epochs, device, seed=42, bs=64):
    set_seed(seed)
    model = RDDetector(student_backbone=backbone, teacher_arch='resnet18').to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(imgs)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for b in range(0, n, bs):
            x = NORM(imgs[idx[b:b + bs]].to(device))
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[3])
            opt.zero_grad()
            loss = model.training_loss(x)
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def scale_maps(model, imgs, device, bs=16, size=224):
    model.eval()
    out = [[], [], []]
    for i in range(0, len(imgs), bs):
        x = NORM(imgs[i:i + bs].to(device))
        t = model.teacher(x)
        s = model.forward_student(x)
        for k in range(3):
            d = torch.norm(s[k] - t[k], dim=1, keepdim=True)
            out[k].append(F.interpolate(d, size=(size, size), mode='bilinear',
                                        align_corners=False).squeeze(1).cpu())
    return [torch.cat(o, dim=0).numpy() for o in out]


def reduce_map(maps, w, mode, k=None):
    m = w[0] * maps[0] + w[1] * maps[1] + w[2] * maps[2]
    flat = m.reshape(m.shape[0], -1)
    if mode == 'mean':
        return flat.mean(axis=1)
    if mode == 'max':
        return flat.max(axis=1)
    n = max(1, int(round(flat.shape[1] * k)))
    return np.sort(flat, axis=1)[:, -n:].mean(axis=1)


def auc(y, s):
    return float(roc_auc_score(np.asarray(y), s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--backbone', default='efficientnet_b0')
    ap.add_argument('--categories', nargs='+', default=CATS)
    a = ap.parse_args()

    device = torch.device('cuda')
    res = json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}
    for i, cat in enumerate(a.categories, 1):
        if cat in res and res[cat].get('done'):
            continue
        t0 = time.time()
        train_imgs, test_imgs, test_labels = load_category_tensors(cat)
        model = train_student(train_imgs, a.backbone, a.epochs, device, seed=a.seed)
        maps = scale_maps(model, test_imgs, device)
        y = np.asarray(test_labels)
        row = {}
        for wname, w in (('uniform', W_UNIFORM), ('w0.3_0.6_0.1', W_BEST)):
            row['%s|mean' % wname] = auc(y, reduce_map(maps, w, 'mean'))
            row['%s|max' % wname] = auc(y, reduce_map(maps, w, 'max'))
            row['%s|top1px' % wname] = auc(y, reduce_map(maps, w, 'topk', 1.0 / (224 * 224)))
            for k in KS:
                row['%s|topk_%g' % (wname, k)] = auc(y, reduce_map(maps, w, 'topk', k))
        res[cat] = {'done': True, 'backbone': a.backbone, 'seed': a.seed,
                    'n_test': len(test_imgs), 'scores': row,
                    'minutes': round((time.time() - t0) / 60, 1)}
        json.dump(res, open(RESFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        log('  [%2d/%d] %-11s mean=%.4f topk1%%=%.4f  (%.1f min)'
            % (i, len(a.categories), cat, row['uniform|mean'],
               row['uniform|topk_0.01'], res[cat]['minutes']))
        del model
        torch.cuda.empty_cache()

    cells = [v for v in res.values() if v.get('done')]
    if not cells:
        return
    keys = sorted(cells[0]['scores'])
    byk = {}
    for kname in keys:
        vals = np.array([c['scores'][kname] for c in cells])
        byk[kname] = {'mean': float(vals.mean()), 'worst': float(vals.min()),
                      'n': int(len(vals))}
    out = {'n_categories': len(cells), 'seed': a.seed,
           'ks': KS, 'policies': byk,
           'uniform_mean_by_k': {str(k): byk['uniform|topk_%g' % k]['mean'] for k in KS},
           'uniform_worst_by_k': {str(k): byk['uniform|topk_%g' % k]['worst'] for k in KS}}
    json.dump(out, open(SUMFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    log('')
    log('=' * 72)
    log('%-22s %12s %12s' % ('policy', 'mean AUROC', 'worst cat'))
    log('-' * 72)
    for kname in ['uniform|mean', 'uniform|max', 'uniform|top1px'] + \
                 ['uniform|topk_%g' % k for k in KS]:
        if kname in byk:
            log('%-22s %11.4f%% %11.4f%%' % (kname, 100 * byk[kname]['mean'],
                                             100 * byk[kname]['worst']))
    log('-' * 72)
    for kname in ['w0.3_0.6_0.1|topk_0.01', 'w0.3_0.6_0.1|topk_0.02']:
        if kname in byk:
            log('%-22s %11.4f%% %11.4f%%' % (kname, 100 * byk[kname]['mean'],
                                             100 * byk[kname]['worst']))
    log('=' * 72)
    log('saved -> %s' % SUMFILE)


if __name__ == '__main__':
    main()
