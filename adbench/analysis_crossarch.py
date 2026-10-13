# -*- coding: utf-8 -*-
"""Do the evaluation findings depend on the backbone?

Compares the ResNet-50 matrix against the ViT (deiT-tiny) matrix on the three
locality settings they share, for:
  (a) the per-metric seed noise,
  (b) the variance decomposition (category / configuration / seed / interaction),
  (c) the absolute metric levels.

A finding that holds on both a convolutional and an attention backbone is much
harder to dismiss as an artefact of one feature extractor.
"""
import io, json, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np

METRICS = ['i_auroc', 'p_auroc', 'pro', 'r_at_1fpr', 'ap_at_1fpr']
SHORT = {'i_auroc': 'I-AUROC', 'p_auroc': 'P-AUROC', 'pro': 'PRO',
         'r_at_1fpr': 'R@1%FPR', 'ap_at_1fpr': 'AP@1%FPR'}
RADII = ['global', 2, 0]           # the settings both matrices share


def load(fn):
    p = os.path.join(HERE, 'results', fn)
    d = json.load(open(p, encoding='utf-8'))
    return [v for v in d.values() if v.get('ok')]


def cube(cells, metric):
    cats = sorted({v['category'] for v in cells})
    seeds = sorted({v['seed'] for v in cells})
    idx = {(v['category'], v['radius'], v['seed']): v.get(metric) for v in cells}
    Y = np.full((len(RADII), len(cats), len(seeds)), np.nan)
    for a, r in enumerate(RADII):
        for b, c in enumerate(cats):
            for d, s in enumerate(seeds):
                val = idx.get((c, r, s))
                if val is not None:
                    Y[a, b, d] = val
    return Y


def vardecomp(Y):
    gm = np.nanmean(Y)
    sc = Y.shape[0] * Y.shape[2] * np.nansum((np.nanmean(Y, axis=(0, 2)) - gm) ** 2)
    sm = Y.shape[1] * Y.shape[2] * np.nansum((np.nanmean(Y, axis=(1, 2)) - gm) ** 2)
    ss = Y.shape[1] * Y.shape[0] * np.nansum((np.nanmean(Y, axis=(0, 1)) - gm) ** 2)
    rs = np.nansum((Y
                    - np.nanmean(Y, axis=(0, 2))[None, :, None]
                    - np.nanmean(Y, axis=(1, 2))[:, None, None]
                    - np.nanmean(Y, axis=(0, 1))[None, None, :] + 2 * gm) ** 2)
    t = sc + sm + ss + rs + 1e-12
    dof = (Y.shape[0] - 1) * (Y.shape[1] - 1) * Y.shape[2]
    F = (sm / (Y.shape[0] - 1)) / ((rs / dof) + 1e-12)
    return 100 * sc / t, 100 * sm / t, 100 * ss / t, 100 * rs / t, F


res = load('matrix.json')
vit = load('matrix_deit.json')
print('cells: ResNet-50 %d | ViT %d' % (len(res), len(vit)))

print()
print('=' * 78)
print('(a) Per-metric seed noise (mean SD across seeds, over configurations x categories)')
print('=' * 78)
print('%-9s %14s %14s %8s' % ('metric', 'ResNet-50', 'ViT (deiT)', 'ratio'))
print('-' * 50)
for m in METRICS:
    a = float(np.nanmean(np.nanstd(cube(res, m), axis=2, ddof=1)))
    b = float(np.nanmean(np.nanstd(cube(vit, m), axis=2, ddof=1)))
    print('%-9s %14.4f %14.4f %8.2f' % (SHORT[m], a, b, (b / a) if a else float('nan')))

print()
print('=' * 78)
print('(b) Variance decomposition: does the CATEGORY dominate on both backbones?')
print('=' * 78)
print('%-9s %-22s %-22s' % ('metric', 'ResNet-50', 'ViT (deiT)'))
print('%-9s %11s %11s %11s %11s' % ('', 'cat', 'config', 'cat', 'config'))
print('-' * 56)
for m in METRICS:
    ra = vardecomp(cube(res, m))
    rb = vardecomp(cube(vit, m))
    print('%-9s %10.1f%% %10.1f%% %10.1f%% %10.1f%%'
          % (SHORT[m], ra[0], ra[1], rb[0], rb[1]))

print()
print('=' * 78)
print('(c) Absolute levels - is the ViT better or worse?')
print('=' * 78)
print('%-9s %12s %12s %10s' % ('metric', 'ResNet-50', 'ViT', 'diff'))
print('-' * 46)
for m in METRICS:
    a = float(np.nanmean(cube(res, m)))
    b = float(np.nanmean(cube(vit, m)))
    print('%-9s %12.4f %12.4f %+10.4f' % (SHORT[m], a, b, b - a))

print()
print('=' * 78)
print('(d) Sample size needed to detect a claimed improvement, per backbone')
print('=' * 78)
print('%-9s %-26s %-26s' % ('metric', 'ResNet-50  (+0.5pp / +0.3pp)', 'ViT  (+0.5pp / +0.3pp)'))
print('-' * 64)
for m in ['i_auroc', 'pro', 'r_at_1fpr']:
    out = []
    for cells in (res, vit):
        sd = float(np.nanmean(np.nanstd(cube(cells, m), axis=2, ddof=1)))
        ns = [int(np.ceil(2 * (2.8 * sd / c) ** 2)) for c in (0.005, 0.003)]
        out.append('%d / %d' % (ns[0], ns[1]))
    print('%-9s %-26s %-26s' % (SHORT[m], out[0], out[1]))
