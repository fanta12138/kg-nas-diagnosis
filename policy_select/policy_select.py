# -*- coding: utf-8 -*-
"""Label-free scoring-policy selection.

The fusion study shows that on a FIXED trained student the image-scoring policy
(per-scale weighting x spatial pooling) moves MVTec-AD AUROC from 82.8% (the
manuscript's uniform-sum / global-mean default) to a 93.7% per-category oracle
-- i.e. +10.9 points, obtainable with NO retraining. The open question is
whether the right policy can be chosen WITHOUT anomaly labels.

This script trains one student per category, scores both the held-out NORMALS
and the labelled test set under every policy, and saves everything so that
selection rules can be compared offline.

Usage: python policy_select.py [--categories ...] [--epochs 500]
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

OUT = _os.path.join(_KGD_ROOT, 'policy_select')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'policy_select.json')

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']


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
        t1, t2, t3 = model.teacher(x)
        s1, s2, s3 = model.forward_student(x)
        for k, (s, t) in enumerate([(s1, t1), (s2, t2), (s3, t3)]):
            d = torch.norm(s - t, dim=1, keepdim=True)
            out[k].append(F.interpolate(d, size=(size, size), mode='bilinear',
                                        align_corners=False).squeeze(1).cpu())
    return [torch.cat(o, dim=0).numpy() for o in out]


def policies():
    """(name, weights, pooling) triples."""
    P = []
    for k in range(3):
        w = [0.0, 0.0, 0.0]
        w[k] = 1.0
        for pm in ('mean', 'topk', 'max'):
            P.append(('s%d_%s' % (k + 1, pm), tuple(w), pm))
    for pm in ('mean', 'topk', 'max'):
        P.append(('sum_%s' % pm, (1 / 3, 1 / 3, 1 / 3), pm))
    for w1 in np.arange(0, 1.001, 0.1):
        for w2 in np.arange(0, 1.001 - w1 + 1e-9, 0.1):
            w3 = round(1 - w1 - w2, 10)
            if w3 < -1e-9:
                continue
            P.append(('w%.1f_%.1f_%.1f_topk' % (w1, w2, w3),
                      (round(w1, 1), round(w2, 1), round(w3, 1)), 'topk'))
    return P


POOL = policies()


def apply_policy(maps, w, pm, k=0.01):
    m = w[0] * maps[0] + w[1] * maps[1] + w[2] * maps[2]
    flat = m.reshape(m.shape[0], -1)
    if pm == 'mean':
        return flat.mean(axis=1)
    if pm == 'max':
        return flat.max(axis=1)
    n = max(1, int(flat.shape[1] * k))
    return np.sort(flat, axis=1)[:, -n:].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=CATS)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--backbone', default='efficientnet_b0')
    ap.add_argument('--holdout', type=float, default=0.2)
    a = ap.parse_args()

    device = torch.device('cuda')
    res = json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}

    for cat in a.categories:
        if cat in res and res[cat].get('done'):
            continue
        t0 = time.time()
        log('=== %s ===' % cat)
        train_imgs, test_imgs, test_labels = load_category_tensors(cat)
        n_fit = int(len(train_imgs) * (1 - a.holdout))
        fit, holdout = train_imgs[:n_fit], train_imgs[n_fit:]
        model = train_student(fit, a.backbone, a.epochs, device)

        mh = scale_maps(model, holdout, device)          # held-out NORMALS
        mt = scale_maps(model, test_imgs, device)
        y = np.asarray(test_labels)
        log('  trained; holdout=%d test=%d' % (len(holdout), len(test_imgs)))

        rows = {}
        for name, w, pm in POOL:
            hs = apply_policy(mh, w, pm)
            ts = apply_policy(mt, w, pm)
            au = float(roc_auc_score(y, ts)) if len(np.unique(y)) > 1 else float('nan')
            rows[name] = {
                'weights': list(w), 'pool': pm, 'auroc': au,
                'h_mean': float(hs.mean()), 'h_std': float(hs.std()),
                'h_p90': float(np.percentile(hs, 90)),
                'h_p95': float(np.percentile(hs, 95)),
                'h_max': float(hs.max()),
                'h_cv': float(hs.std() / max(abs(hs.mean()), 1e-12)),
            }
        best = max(rows, key=lambda k: rows[k]['auroc'])
        res[cat] = {'done': True, 'n_fit': len(fit), 'n_holdout': len(holdout),
                    'n_test': len(test_imgs), 'policies': rows,
                    'current_default': rows['sum_mean']['auroc'],
                    'oracle_policy': best, 'oracle_auroc': rows[best]['auroc'],
                    'minutes': round((time.time() - t0) / 60, 1)}
        json.dump(res, open(RESFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        log('  default=%.4f oracle=%s=%.4f (%.1f min)'
            % (rows['sum_mean']['auroc'], best, rows[best]['auroc'], res[cat]['minutes']))
        del model
        torch.cuda.empty_cache()

    # -------- offline evaluation of label-free selection rules --------
    done = {k: v for k, v in res.items() if v.get('done')}
    if not done:
        return
    cats = sorted(done)

    def pick(c, rule):
        pol = done[c]['policies']
        if rule == 'oracle':
            return max(pol, key=lambda k: pol[k]['auroc'])
        if rule == 'default':
            return 'sum_mean'
        if rule == 'h_p95':
            return min(pol, key=lambda k: pol[k]['h_p95'])
        if rule == 'h_p90':
            return min(pol, key=lambda k: pol[k]['h_p90'])
        if rule == 'h_max':
            return min(pol, key=lambda k: pol[k]['h_max'])
        if rule == 'h_cv':
            return min(pol, key=lambda k: pol[k]['h_cv'])
        if rule == 'h_mean':
            return min(pol, key=lambda k: pol[k]['h_mean'])
        raise ValueError(rule)

    RULES = ['default', 'oracle', 'h_p95', 'h_p90', 'h_max', 'h_cv', 'h_mean']
    log('')
    log('=' * 72)
    log('%-14s %12s %10s' % ('selection rule', 'mean AUROC', 'vs default'))
    log('-' * 72)
    base = np.mean([done[c]['current_default'] for c in cats])
    out = {}
    for r in RULES:
        vals = [done[c]['policies'][pick(c, r)]['auroc'] for c in cats]
        hit = np.mean([pick(c, r) == done[c]['oracle_policy'] for c in cats])
        out[r] = {'mean_auroc': float(np.mean(vals)), 'oracle_hit': float(hit)}
        log('%-14s %11.4f%% %9.2f pp   (命中 oracle %d/%d)'
            % (r, 100 * np.mean(vals), 100 * (np.mean(vals) - base), int(hit * len(cats)), len(cats)))
    json.dump({'per_category': done, 'rules': out, 'default_mean': float(base)},
              open(os.path.join(OUT, 'policy_select_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
