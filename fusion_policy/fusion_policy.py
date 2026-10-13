# -*- coding: utf-8 -*-
"""Scale-fusion policy search for reverse-distillation anomaly detection.

The RD detector scores an image by AVERAGING the multi-scale discrepancy map
over all pixels (rd_detector.anomaly_score). Both the per-scale weighting and
the spatial pooling are free design choices: once a student is trained, every
fusion policy can be evaluated on the SAME model, so this axis can be searched
exhaustively without repeating the training run. That is a deployment
convenience -- the policy can be re-tuned on an existing detector -- and it is
what makes an exhaustive sweep affordable. It is NOT an argument that training
the backbone is wasteful: see the capacity study, which had to retrain detectors
across the whole parameter range to show that the backbone axis is inert.

This script trains one student per MVTec-AD category (standard protocol) and
then evaluates a grid of fusion policies on the shared model:
  * single-scale scores (layer1 / layer2 / layer3)
  * uniform sum (the current default)
  * convex weight grid over the three scales
  * per-image scale normalisation before combining
  * spatial pooling variants: global mean / top-k mean / max

If the spread across policies is large, scale fusion is a high-leverage axis
that can be searched exhaustively on an already-trained detector.

Usage: python fusion_policy.py [--categories ...] [--epochs 500]
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

import argparse, io, itertools, json, os, sys, time

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

OUT = _os.path.join(_KGD_ROOT, 'fusion_policy')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'fusion_policy.json')

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def train_student(fit_imgs, backbone, epochs, device, seed=42, bs=64):
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
def scale_maps(model, imgs, device, bs=16, size=224):
    """Per-scale discrepancy maps, each upsampled to (size, size)."""
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
    return [torch.cat(o, dim=0).numpy() for o in out]     # each [N, size, size]


def pool(maps, mode, k=0.01):
    """maps: [N,H,W] -> [N] image-level scores."""
    flat = maps.reshape(maps.shape[0], -1)
    if mode == 'mean':
        return flat.mean(axis=1)
    if mode == 'max':
        return flat.max(axis=1)
    if mode == 'topk':
        n = max(1, int(flat.shape[1] * k))
        return np.sort(flat, axis=1)[:, -n:].mean(axis=1)
    raise ValueError(mode)


def norm01(m):
    lo = m.reshape(m.shape[0], -1).min(axis=1)[:, None, None]
    hi = m.reshape(m.shape[0], -1).max(axis=1)[:, None, None]
    return (m - lo) / np.maximum(hi - lo, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=CATS)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--backbone', default='efficientnet_b0')
    a = ap.parse_args()

    device = torch.device('cuda')
    res = json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}

    for cat in a.categories:
        if cat in res and res[cat].get('done'):
            continue
        t0 = time.time()
        log('=== %s ===' % cat)
        train_imgs, test_imgs, test_labels = load_category_tensors(cat)
        model = train_student(train_imgs, a.backbone, a.epochs, device)
        m = scale_maps(model, test_imgs, device)
        y = np.asarray(test_labels)
        log('  trained (%d test imgs), maps ready' % len(test_imgs))

        rows = {}
        # --- single scale ---
        for k in range(3):
            for pm in ('mean', 'topk', 'max'):
                rows['s%d_%s' % (k + 1, pm)] = auroc(y, pool(m[k], pm))
        # --- uniform sum, several poolings ---
        for pm in ('mean', 'topk', 'max'):
            rows['sum_%s' % pm] = auroc(y, pool(m[0] + m[1] + m[2], pm))
        # --- convex weight grid (step .1) ---
        best_w, best_auc = None, -1
        for w1 in np.arange(0, 1.01, 0.1):
            for w2 in np.arange(0, 1.01 - w1 + 1e-9, 0.1):
                w3 = round(1 - w1 - w2, 10)
                if w3 < -1e-9:
                    continue
                s = pool(w1 * m[0] + w2 * m[1] + w3 * m[2], 'topk')
                au = auroc(y, s)
                tag = 'w%.1f_%.1f_%.1f_topk' % (w1, w2, w3)
                rows[tag] = au
                if au > best_auc:
                    best_auc, best_w = au, tag
        # --- per-image normalised scales ---
        n0, n1, n2 = norm01(m[0]), norm01(m[1]), norm01(m[2])
        for pm in ('mean', 'topk'):
            rows['norm_sum_%s' % pm] = auroc(y, pool(n0 + n1 + n2, pm))
            rows['norm_max_%s' % pm] = auroc(y, pool(np.maximum(np.maximum(n0, n1), n2), pm))
        # --- best single policy overall ---
        best_tag = max(rows, key=lambda k: rows[k])
        res[cat] = {'done': True, 'backbone': a.backbone, 'epochs': a.epochs,
                    'n_test': len(test_imgs), 'policies': rows,
                    'current_default': rows['sum_mean'],
                    'best_policy': best_tag, 'best_auroc': rows[best_tag],
                    'best_weight_policy': best_w, 'best_weight_auroc': best_auc,
                    'minutes': round((time.time() - t0) / 60, 1)}
        json.dump(res, open(RESFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        log('  sum_mean(current)=%.4f | best=%s=%.4f | best_w=%s=%.4f (%.1f min)'
            % (rows['sum_mean'], best_tag, rows[best_tag], best_w, best_auc,
               res[cat]['minutes']))
        del model
        torch.cuda.empty_cache()

    # ---------------- aggregate ----------------
    done = {k: v for k, v in res.items() if v.get('done')}
    if not done:
        return
    cats = sorted(done)
    log('')
    log('=' * 78)
    log('%-26s %10s %10s %10s' % ('policy', 'mean AUROC', 'min', 'max'))
    log('-' * 78)
    allp = sorted({p for c in cats for p in done[c]['policies']})
    cur = np.mean([done[c]['policies']['sum_mean'] for c in cats])
    for p in allp:
        vals = [done[c]['policies'].get(p, np.nan) for c in cats]
        m_ = np.nanmean(vals)
        if m_ > cur - 0.001 or p in ('sum_mean', 'sum_topk', 's1_topk', 's2_topk', 's3_topk'):
            log('%-26s %9.4f%% %9.4f%% %9.4f%%%s'
                % (p, 100 * m_, 100 * np.nanmin(vals), 100 * np.nanmax(vals),
                   '   <== current default' if p == 'sum_mean' else ''))
    oracle = np.mean([done[c]['best_auroc'] for c in cats])
    log('-' * 78)
    log('%-26s %9.4f%%' % ('CURRENT default (sum/mean)', 100 * cur))
    log('%-26s %9.4f%%  (%+.2f pp)' % ('per-category oracle', 100 * oracle, 100 * (oracle - cur)))
    log('%-26s %9.4f%%  (%+.2f pp)'
        % ('best single policy', 100 * max(np.mean([done[c]['policies'][p] for c in cats])
                                           for p in allp), 0))
    json.dump({'per_category': done, 'default_mean': cur, 'oracle_mean': oracle},
              open(os.path.join(OUT, 'fusion_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)


def auroc(y, s):
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


if __name__ == '__main__':
    main()
