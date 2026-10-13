# -*- coding: utf-8 -*-
"""Replicate the scale-fusion result under additional training seeds.

fusion_policy/ measured the headline effect -- mean pooling -> top-k pooling on a
*fixed* detector, +7.4 points of mean AUROC -- on a single trained student per
category (seed 42). A claim of that size needs replication, so this script
retrains the same EfficientNet-B0 student with further seeds and re-evaluates the
small set of policies that carry the effect.

Querying several policies on one model is free; the expensive part, and the
point of this script, is the retraining.

Usage: python fusion_multiseed.py --seeds 43 44        # 30 trainings, ~2 h
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

OUT = _os.path.join(_KGD_ROOT, 'fusion_multiseed')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'fusion_multiseed.json')
SUMFILE = os.path.join(OUT, 'fusion_multiseed_summary.json')

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']

# The policies that matter: the published default, the pooling variants that
# carry the effect, and the best weighted policy found by fusion_policy/.
POLICIES = [
    ('sum_mean', (1 / 3, 1 / 3, 1 / 3), 'mean'),
    ('sum_topk', (1 / 3, 1 / 3, 1 / 3), 'topk'),
    ('sum_max', (1 / 3, 1 / 3, 1 / 3), 'max'),
    ('w0.3_0.6_0.1_topk', (0.3, 0.6, 0.1), 'topk'),
    ('w0.2_0.6_0.2_topk', (0.2, 0.6, 0.2), 'topk'),
]


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


def apply_policy(maps, w, pm, k=0.01):
    m = w[0] * maps[0] + w[1] * maps[1] + w[2] * maps[2]
    flat = m.reshape(m.shape[0], -1)
    if pm == 'mean':
        return flat.mean(axis=1)
    if pm == 'max':
        return flat.max(axis=1)
    n = max(1, int(flat.shape[1] * k))
    return np.sort(flat, axis=1)[:, -n:].mean(axis=1)


def auc(y, s):
    return float(roc_auc_score(np.asarray(y), s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[43, 44])
    ap.add_argument('--categories', nargs='+', default=CATS)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--backbone', default='efficientnet_b0')
    a = ap.parse_args()

    device = torch.device('cuda')
    res = json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}
    total = len(a.seeds) * len(a.categories)
    i = 0
    for seed in a.seeds:
        for cat in a.categories:
            i += 1
            key = '%s|%d' % (cat, seed)
            if key in res and res[key].get('done'):
                continue
            t0 = time.time()
            train_imgs, test_imgs, test_labels = load_category_tensors(cat)
            model = train_student(train_imgs, a.backbone, a.epochs, device, seed=seed)
            maps = scale_maps(model, test_imgs, device)
            y = np.asarray(test_labels)
            pol = {}
            for name, w, pm in POLICIES:
                pol[name] = auc(y, apply_policy(maps, w, pm))
            res[key] = {'done': True, 'category': cat, 'seed': seed,
                        'backbone': a.backbone, 'policies': pol,
                        'minutes': round((time.time() - t0) / 60, 1)}
            json.dump(res, open(RESFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
            log('  [%3d/%d] s%d %-11s mean=%.4f topk=%.4f (%.1f min)'
                % (i, total, seed, cat, pol['sum_mean'], pol['sum_topk'],
                   res[key]['minutes']))
            del model
            torch.cuda.empty_cache()

    # ---- summary across every (category, seed) cell available ---------------
    cells = [v for v in res.values() if v.get('done')]
    if not cells:
        return
    names = [p[0] for p in POLICIES]
    agg = {n: float(np.mean([c['policies'][n] for c in cells])) for n in names}
    per_cell_gain = np.array([c['policies']['sum_topk'] - c['policies']['sum_mean']
                              for c in cells])
    by_seed = {}
    for s in sorted({c['seed'] for c in cells}):
        sub = [c for c in cells if c['seed'] == s]
        by_seed[s] = {
            'n_cats': len(sub),
            'sum_mean': float(np.mean([c['policies']['sum_mean'] for c in sub])),
            'sum_topk': float(np.mean([c['policies']['sum_topk'] for c in sub])),
            'gain_pp': float(100 * np.mean([c['policies']['sum_topk'] - c['policies']['sum_mean']
                                            for c in sub])),
        }
    out = {'n_cells': len(cells),
           'seeds': sorted({c['seed'] for c in cells}),
           'mean_auroc_by_policy': agg,
           'topk_minus_mean_pp': float(100 * per_cell_gain.mean()),
           'topk_minus_mean_sd_pp': float(100 * per_cell_gain.std(ddof=0)),
           'topk_minus_mean_min_pp': float(100 * per_cell_gain.min()),
           'topk_minus_mean_max_pp': float(100 * per_cell_gain.max()),
           'by_seed': by_seed,
           'per_cell': {('%s|%d' % (c['category'], c['seed'])): c['policies'] for c in cells}}
    json.dump(out, open(SUMFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    log('')
    log('=' * 68)
    log('cells (category x seed) = %d   seeds = %s' % (len(cells), out['seeds']))
    for n in names:
        log('  %-22s mean AUROC = %.4f%%' % (n, 100 * agg[n]))
    log('  top-k minus mean: %+.2f pp (sd %.2f, range %+.2f .. %+.2f)'
        % (out['topk_minus_mean_pp'], out['topk_minus_mean_sd_pp'],
           out['topk_minus_mean_min_pp'], out['topk_minus_mean_max_pp']))
    log('=' * 68)
    log('saved -> %s' % SUMFILE)


if __name__ == '__main__':
    main()
