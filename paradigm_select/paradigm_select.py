# -*- coding: utf-8 -*-
"""Label-free selection between detection paradigms.

The manuscript reports that the DETECTION PARADIGM (feature-prototype vs
reverse distillation) swings MVTec-AD AUROC by 10.2 points on the same trained
student, while the backbone search axis swings it by only 1.2 points, and that
"automatically determining which paradigm to apply per category from unlabeled
training data alone remains an open direction".

This script tests candidate LABEL-FREE selection rules. Both paradigms reuse the
same trained student (only the scoring rule differs), so one training run per
category yields both.

Protocol per category:
  1. split the normal training set 80/20 into fit / holdout
  2. train the RD student on `fit` only (500 epochs, seed 42, ResNet-18 teacher)
  3. build the prototype from `fit` features
  4. score the HOLDOUT NORMALS with both paradigms  -> selection criteria
     (no anomaly labels are touched here)
  5. score the test set with both paradigms -> true AUROC
  6. report: oracle | always-prototype | always-RD | each label-free rule

Usage: python paradigm_select.py [--categories ...] [--epochs 500]
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

OUT = _os.path.join(_KGD_ROOT, 'paradigm_select')
os.makedirs(OUT, exist_ok=True)

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']


def train_student(fit_imgs, backbone, epochs, device, seed=42, bs=64, log=None):
    set_seed(seed)
    model = RDDetector(student_backbone=backbone, teacher_arch='resnet18').to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(fit_imgs)
    for ep in range(1, epochs + 1):
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
        if log is not None and (ep % 100 == 0 or ep == epochs):
            log('      epoch %d loss=%.4f' % (ep, loss.item()))
    return model


@torch.no_grad()
def build_prototype(model, fit_imgs, device, bs=64):
    feats = []
    model.eval()
    for i in range(0, len(fit_imgs), bs):
        z = model.encoder(NORM(fit_imgs[i:i + bs].to(device)))
        feats.append(z.reshape(z.shape[0], z.shape[1], -1).mean(dim=2).cpu())
    return torch.cat(feats, dim=0).mean(dim=0).to(device)


@torch.no_grad()
def score(model, imgs, paradigm, proto, device, bs=64):
    out = []
    model.eval()
    for i in range(0, len(imgs), bs):
        x = NORM(imgs[i:i + bs].to(device))
        if paradigm == 'rd':
            s = model.anomaly_score(x)
        else:
            s = model.prototype_score(x, proto)
        out.extend(s.cpu().numpy())
    return np.asarray(out)


def auroc(labels, scores):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(roc_auc_score(labels, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=CATS)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--backbone', default='efficientnet_b0')
    ap.add_argument('--holdout', type=float, default=0.2)
    a = ap.parse_args()

    device = torch.device('cuda')
    results = {}
    resfile = os.path.join(OUT, 'paradigm_selection.json')
    if os.path.exists(resfile):
        results = json.load(open(resfile, encoding='utf-8'))

    def log(m):
        print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)

    for cat in a.categories:
        if cat in results and results[cat].get('done'):
            log('skip %s (done)' % cat)
            continue
        t0 = time.time()
        log('=== %s ===' % cat)
        train_imgs, test_imgs, test_labels = load_category_tensors(cat)
        n_fit = int(len(train_imgs) * (1 - a.holdout))
        fit, holdout = train_imgs[:n_fit], train_imgs[n_fit:]
        log('  fit=%d holdout=%d test=%d' % (len(fit), len(holdout), len(test_imgs)))

        model = train_student(fit, a.backbone, a.epochs, device, log=log)
        proto = build_prototype(model, fit, device)

        # --- label-free criteria: score the held-out NORMALS only ---
        h_rd = score(model, holdout, 'rd', proto, device)
        h_pr = score(model, holdout, 'proto', proto, device)

        # --- true performance on the labelled test set ---
        t_rd = score(model, test_imgs, 'rd', proto, device)
        t_pr = score(model, test_imgs, 'proto', proto, device)
        au_rd = auroc(test_labels.numpy(), t_rd)
        au_pr = auroc(test_labels.numpy(), t_pr)

        results[cat] = {
            'done': True, 'backbone': a.backbone, 'epochs': a.epochs,
            'n_fit': len(fit), 'n_holdout': len(holdout), 'n_test': len(test_imgs),
            'auroc_rd': au_rd, 'auroc_proto': au_pr,
            'best': 'rd' if au_rd >= au_pr else 'proto',
            'holdout': {
                'rd': {'mean': float(h_rd.mean()), 'std': float(h_rd.std()),
                       'p90': float(np.percentile(h_rd, 90)),
                       'p95': float(np.percentile(h_rd, 95)),
                       'max': float(h_rd.max())},
                'proto': {'mean': float(h_pr.mean()), 'std': float(h_pr.std()),
                          'p90': float(np.percentile(h_pr, 90)),
                          'p95': float(np.percentile(h_pr, 95)),
                          'max': float(h_pr.max())},
            },
            'minutes': round((time.time() - t0) / 60, 1),
        }
        json.dump(results, open(resfile, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        log('  rd=%.4f proto=%.4f best=%s (%.1f min)'
            % (au_rd, au_pr, results[cat]['best'], results[cat]['minutes']))
        del model
        torch.cuda.empty_cache()

    # ---------------- aggregate ----------------
    done = {k: v for k, v in results.items() if v.get('done')}
    if not done:
        return
    cats = sorted(done)
    oracle = np.mean([max(done[c]['auroc_rd'], done[c]['auroc_proto']) for c in cats])
    always_rd = np.mean([done[c]['auroc_rd'] for c in cats])
    always_pr = np.mean([done[c]['auroc_proto'] for c in cats])

    RULES = {
        'holdout_p95_min': lambda h: min(('rd', h['rd']['p95']), ('proto', h['proto']['p95']), key=lambda t: t[1])[0],
        'holdout_p90_min': lambda h: min(('rd', h['rd']['p90']), ('proto', h['proto']['p90']), key=lambda t: t[1])[0],
        'holdout_mean_std_min': lambda h: min(
            ('rd', h['rd']['mean'] + h['rd']['std']),
            ('proto', h['proto']['mean'] + h['proto']['std']), key=lambda t: t[1])[0],
        'holdout_max_min': lambda h: min(('rd', h['rd']['max']), ('proto', h['proto']['max']), key=lambda t: t[1])[0],
    }

    log('')
    log('=' * 74)
    log('%-11s %9s %9s %9s   %s' % ('category', 'RD', 'proto', 'true best', 'label-free picks'))
    log('-' * 74)
    summary = {}
    for c in cats:
        picks = {name: fn(done[c]['holdout']) for name, fn in RULES.items()}
        log('%-11s %9.4f %9.4f %9s   %s'
            % (c, done[c]['auroc_rd'], done[c]['auroc_proto'], done[c]['best'],
               ' '.join('%s=%s' % (n.split('_')[0], p) for n, p in picks.items())))
        summary[c] = picks

    log('')
    log('=' * 74)
    log('%-24s %10s %10s' % ('strategy', 'mean AUROC', 'vs oracle'))
    log('-' * 74)
    log('%-24s %9.4f%% %10s' % ('oracle (upper bound)', 100 * oracle, '--'))
    log('%-24s %9.4f%% %9.1f%%' % ('always RD', 100 * always_rd, 100 * (always_rd / oracle - 1)))
    log('%-24s %9.4f%% %9.1f%%' % ('always prototype', 100 * always_pr, 100 * (always_pr / oracle - 1)))
    for name, fn in RULES.items():
        vals = [done[c]['auroc_rd'] if summary[c][name] == 'rd' else done[c]['auroc_proto'] for c in cats]
        acc = np.mean([summary[c][name] == done[c]['best'] for c in cats])
        log('%-24s %9.4f%% %9.1f%%   (picks true best in %d/%d)'
            % (name, 100 * np.mean(vals), 100 * (np.mean(vals) / oracle - 1), int(acc * len(cats)), len(cats)))

    json.dump({'per_category': done, 'summary': summary,
               'aggregate': {'oracle': oracle, 'always_rd': always_rd, 'always_proto': always_pr}},
              open(os.path.join(OUT, 'paradigm_selection_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    log('')
    log('wrote ' + os.path.join(OUT, 'paradigm_selection_summary.json'))


if __name__ == '__main__':
    main()
