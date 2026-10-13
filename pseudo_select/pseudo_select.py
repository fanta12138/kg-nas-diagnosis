# -*- coding: utf-8 -*-
"""Label-free selection of the anomaly-score fusion policy -- v2.

Problem
-------
The reverse-distillation detector scores an image as
    mean over the whole map of (m1 + m2 + m3)
where m_k is the teacher-student discrepancy at scale k.  Both the *pooling*
(mean / top-k / max) and the *scale weights* are free design choices.  On a
FIXED trained student they move MVTec-AD AUROC from 82.4% to a 93.4%
per-category oracle -- +11 points with zero retraining.  But choosing the policy
requires knowing which images are anomalous.

Why not distribution statistics
-------------------------------
policy_select/ showed that selecting on normal-data statistics FAILS (-1.3 to
-0.1 pp): top-k pooling inflates the normal-score tail, so "tightest normal
distribution" is anti-correlated with detection.  Tightness is the wrong proxy.

This script
-----------
Uses a *discriminative* proxy instead: perturb held-out NORMALS with synthetic
anomalies (the DRAEM / SimpleNet trick) and pick the policy that separates clean
from pseudo-anomalous holdout images best.  Three variants of synthetic defect
(smoothed blob, thin scratch, rectangular cut), averaged over several seeds, so
the selection does not hinge on one random perturbation.  We also try an
ensemble of the top-K policies by pseudo-AUROC.

Everything is label-free with respect to real anomalies: only train/good images
are ever used, split into a fit part (trains the student) and a holdout part
(clean reference for the pseudo-anomaly test).

Usage: python pseudo_select.py [--categories ...] [--epochs 500] [--seeds 5]
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
from scipy.stats import spearmanr

_spec = importlib.util.spec_from_file_location('p2', os.path.join(NAS, 'scripts', 'p2_rd_train.py'))
_p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p2)
load_category_tensors = _p2.load_category_tensors
NORM = _p2.NORM
set_seed = _p2.set_seed

from methods.rd_detector import RDDetector
from torch.optim import AdamW

OUT = _os.path.join(_KGD_ROOT, 'pseudo_select')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'pseudo_select.json')
SUMFILE = os.path.join(OUT, 'pseudo_summary.json')

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']
POOL_K = 0.01          # top-1% of pixels


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


# ------------------------------------------------------------------ training
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


# ------------------------------------------------------- synthetic anomalies
def _blob(h, w, g):
    coarse = torch.rand(1, 1, 8, 8, generator=g)
    return F.interpolate(coarse, size=(h, w), mode='bilinear', align_corners=False)[0, 0]


def _scratch(h, w, g):
    """Thin oriented line: coarse noise on a 1-px-wide rotated band."""
    ang = float(torch.rand(1, generator=g).item()) * np.pi
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    cy, cx = h / 2, w / 2
    d = np.abs(np.cos(ang) * (yy - cy).float() + np.sin(ang) * (xx - cx).float())
    width = 1.0 + 2.0 * float(torch.rand(1, generator=g).item())
    band = torch.exp(-(d ** 2) / (2 * width ** 2))
    prof = torch.rand(max(h, w), generator=g)
    return band * prof[:h].unsqueeze(1)


def _cut(h, w, g):
    """Rectangular occlusion-like region."""
    m = torch.zeros(h, w)
    rh = int(h * (0.1 + 0.25 * float(torch.rand(1, generator=g).item())))
    rw = int(w * (0.1 + 0.25 * float(torch.rand(1, generator=g).item())))
    y0 = int(torch.randint(0, max(1, h - rh), (1,), generator=g).item())
    x0 = int(torch.randint(0, max(1, w - rw), (1,), generator=g).item())
    m[y0:y0 + rh, x0:x0 + rw] = 1.0
    return m


KINDS = ('blob', 'scratch', 'cut')


def make_pseudo(imgs, kind='blob', strength=0.35, seed=0, jitter=0.2):
    """Add a synthetic anomaly to each image.  Uses only the images' own pixels
    plus random noise -- never touches real anomalous data.

    `jitter` randomises the perturbation strength per image, so the synthetic
    anomalies span easy and hard cases rather than one fixed magnitude.
    """
    g = torch.Generator().manual_seed(seed)
    n, c, h, w = imgs.shape
    out = imgs.clone()
    for i in range(n):
        if kind == 'blob':
            pat = _blob(h, w, g)
        elif kind == 'scratch':
            pat = _scratch(h, w, g)
        else:
            pat = _cut(h, w, g)
        pat = pat / (pat.max() + 1e-8)
        sign = 1.0 if float(torch.rand(1, generator=g).item()) > 0.5 else -1.0
        s = strength * (1.0 - jitter + 2 * jitter * float(torch.rand(1, generator=g).item()))
        out[i] = torch.clamp(out[i] + sign * s * pat.unsqueeze(0), 0, 1)
    return out


# ------------------------------------------------------------------- policies
def simplex(step):
    out = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((round(i * step, 6), round(j * step, 6), round(k * step, 6)))
    return out


def build_policies():
    P = []
    for w in simplex(0.1):
        P.append(('%.1f_%.1f_%.1f_topk' % w, w, 'topk'))
    for w in simplex(0.2):
        P.append(('%.1f_%.1f_%.1f_mean' % w, w, 'mean'))
        P.append(('%.1f_%.1f_%.1f_max' % w, w, 'max'))
    # canonical names used elsewhere in the project
    P.append(('sum_mean', (1 / 3, 1 / 3, 1 / 3), 'mean'))
    P.append(('sum_topk', (1 / 3, 1 / 3, 1 / 3), 'topk'))
    P.append(('sum_max', (1 / 3, 1 / 3, 1 / 3), 'max'))
    seen, uniq = set(), []
    for name, w, pm in P:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, w, pm))
    return uniq


POOL = build_policies()


def apply_policy(maps, w, pm, k=POOL_K):
    m = w[0] * maps[0] + w[1] * maps[1] + w[2] * maps[2]
    flat = m.reshape(m.shape[0], -1)
    if pm == 'mean':
        return flat.mean(axis=1)
    if pm == 'max':
        return flat.max(axis=1)
    n = max(1, int(flat.shape[1] * k))
    return np.sort(flat, axis=1)[:, -n:].mean(axis=1)


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


def auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def ranknorm(s):
    r = np.argsort(np.argsort(s)).astype(float)
    return r / max(len(r) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=CATS)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--backbone', default='efficientnet_b0')
    ap.add_argument('--holdout', type=float, default=0.3)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--topk_ens', type=int, nargs='+', default=[5, 10, 20])
    ap.add_argument('--train-seed', type=int, default=42,
                    help='seed for the STUDENT training run (the pseudo-anomaly '
                         'variants have their own --seeds parameter); vary this to '
                         'check that the result survives a change of training seed')
    ap.add_argument('--tag', default='',
                    help='suffix for the output files, so several training seeds '
                         'can be run without overwriting each other')
    a = ap.parse_args()

    global RESFILE, SUMFILE
    if a.tag:
        RESFILE = os.path.join(OUT, 'pseudo_select_%s.json' % a.tag)
        SUMFILE = os.path.join(OUT, 'pseudo_summary_%s.json' % a.tag)
    log('train_seed=%d  tag=%s  -> %s' % (a.train_seed, a.tag, RESFILE))

    device = torch.device('cuda')
    res = json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}

    for cat in a.categories:
        if cat in res and res[cat].get('done'):
            continue
        t0 = time.time()
        log('=== %s ===' % cat)
        train_imgs, test_imgs, test_labels = load_category_tensors(cat)
        n_fit = int(len(train_imgs) * (1 - a.holdout))
        fit, hold = train_imgs[:n_fit], train_imgs[n_fit:]
        model = train_student(fit, a.backbone, a.epochs, device, seed=a.train_seed)

        mh = scale_maps(model, hold, device)
        mt = scale_maps(model, test_imgs, device)
        mp = {}
        for s_i in range(a.seeds):
            for k_i, kind in enumerate(KINDS):
                mp['%s%d' % (kind, s_i)] = scale_maps(
                    model, make_pseudo(hold, kind, 0.35, seed=1000 * s_i + 13 * k_i + 1),
                    device)
        n_ps = len(mp)
        log('  trained; fit=%d holdout=%d test=%d  pseudo variants=%d'
            % (len(fit), len(hold), len(test_imgs), n_ps))

        y = np.asarray(test_labels)
        rows = {}
        for name, w, pm in POOL:
            sc_test = apply_policy(mt, w, pm)
            clean = apply_policy(mh, w, pm)
            ps_aucs = []
            for kk, mm in mp.items():
                ps_aucs.append(auc(np.r_[np.zeros(len(hold)), np.ones(len(hold))],
                                   np.r_[clean, apply_policy(mm, w, pm)]))
            rows[name] = {'weights': list(w), 'pool': pm,
                          'auroc': auc(y, sc_test),
                          'pseudo_auc': float(np.nanmean(ps_aucs)),
                          'pseudo_auc_std': float(np.nanstd(ps_aucs))}
            rows[name]['_scores'] = sc_test

        # rank correlation between pseudo criterion and true AUROC over policies
        pv = np.array([rows[k]['pseudo_auc'] for k in rows])
        tv = np.array([rows[k]['auroc'] for k in rows])
        ok = np.isfinite(pv) & np.isfinite(tv)
        rho = float(spearmanr(pv[ok], tv[ok])[0]) if ok.sum() > 2 else float('nan')

        order = sorted(rows, key=lambda k: -np.nan_to_num(rows[k]['auroc'], nan=-1))
        ord_ps = sorted(rows, key=lambda k: -np.nan_to_num(rows[k]['pseudo_auc'], nan=-1))
        best, best_ps = order[0], ord_ps[0]

        # ensembles of the top-K policies by the pseudo criterion
        ens = {}
        for K in a.topk_ens:
            picks = ord_ps[:K]
            ens['top%d' % K] = auc(y, np.mean([ranknorm(rows[p]['_scores']) for p in picks],
                                              axis=0))
        # reference ensembles (use the true metric; upper bound only)
        ens['oracle_top%d' % a.topk_ens[0]] = auc(
            y, np.mean([ranknorm(rows[p]['_scores']) for p in order[:a.topk_ens[0]]], axis=0))

        for k in rows:
            rows[k].pop('_scores')

        res[cat] = {'done': True, 'n_fit': len(fit), 'n_holdout': len(hold),
                    'n_test': len(test_imgs), 'pseudo_variants': n_ps,
                    'policies': rows,
                    'default': rows['sum_mean']['auroc'],
                    'oracle_policy': best, 'oracle_auroc': rows[best]['auroc'],
                    'pseudo_policy': best_ps, 'pseudo_selected_auroc': rows[best_ps]['auroc'],
                    'pseudo_top1_is_oracle': bool(best_ps == best),
                    'spearman_pseudo_vs_auroc': rho,
                    'ensembles': ens,
                    'minutes': round((time.time() - t0) / 60, 1)}
        json.dump(res, open(RESFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        log('  default=%.4f oracle=%.4f  pseudo-pick=%s->%.4f  ens_top5=%.4f  rho=%.2f (%.1f min)'
            % (rows['sum_mean']['auroc'], rows[best]['auroc'], best_ps,
               rows[best_ps]['auroc'], ens['top%d' % a.topk_ens[0]], rho,
               res[cat]['minutes']))
        del model
        torch.cuda.empty_cache()

    summarize(res, a)


def summarize(res, a):
    done = {k: v for k, v in res.items() if v.get('done')}
    if not done:
        return
    cats = sorted(done)
    d = np.mean([done[c]['default'] for c in cats])
    o = np.mean([done[c]['oracle_auroc'] for c in cats])
    p = np.mean([done[c]['pseudo_selected_auroc'] for c in cats])
    hit = np.mean([bool(done[c]['pseudo_top1_is_oracle']) for c in cats])
    log('')
    log('=' * 74)
    log('%-30s %12s %11s' % ('strategy', 'mean AUROC', 'vs default'))
    log('-' * 74)
    log('%-30s %11.4f%% %10s' % ('default (sum/mean, paper)', 100 * d, '--'))
    log('%-30s %11.4f%% %9.2f pp  <== per-category upper bound'
        % ('oracle policy', 100 * o, 100 * (o - d)))
    log('%-30s %11.4f%% %9.2f pp  (top-1 hit %d/%d)'
        % ('pseudo-anomaly selection', 100 * p, 100 * (p - d),
           int(round(hit * len(cats))), len(cats)))
    ens_rows = {}
    for K in a.topk_ens:
        v = np.mean([done[c]['ensembles']['top%d' % K] for c in cats])
        ens_rows['top%d' % K] = v
        log('%-30s %11.4f%% %9.2f pp  (label-free ensemble)'
            % ('pseudo top-%d ensemble' % K, 100 * v, 100 * (v - d)))
    orc = np.mean([done[c]['ensembles']['oracle_top%d' % a.topk_ens[0]] for c in cats])
    log('%-30s %11.4f%% %9.2f pp  (oracle ensemble, upper bound)'
        % ('oracle top-%d ensemble' % a.topk_ens[0], 100 * orc, 100 * (orc - d)))
    log('=' * 74)
    json.dump({'per_category': done, 'default': float(d), 'oracle': float(o),
               'pseudo_selected': float(p), 'oracle_hit': float(hit),
               'ensembles': {k: float(v) for k, v in ens_rows.items()},
               'oracle_ensemble': float(orc),
               'spearman_pseudo_vs_auroc_mean': float(
                   np.nanmean([done[c]['spearman_pseudo_vs_auroc'] for c in cats]))},
              open(SUMFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
