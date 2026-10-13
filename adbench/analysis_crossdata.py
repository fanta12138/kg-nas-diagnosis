# -*- coding: utf-8 -*-
"""Do the evaluation findings survive a change of dataset?

Compares MVTec-AD (15 categories) against MVTec-LOCO (5 categories, logical +
structural anomalies) on the three locality settings they share:

  (a) per-metric seed noise,
  (b) the variance decomposition (category / configuration / seed / interaction),
  (c) absolute levels and the logical-vs-structural gap.

Caveat: LOCO has only five categories, so its category-level variance component is
estimated from fewer levels than MVTec's. That is reported rather than hidden.
"""
import io, json, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np

RADII = ['global', 2, 0]
SHORT = {'i_auroc': 'I-AUROC', 'p_auroc': 'P-AUROC', 'pro': 'PRO', 'spro': 'sPRO',
         'spro_log': 'sPRO-logical', 'spro_str': 'sPRO-structural',
         'r_at_1fpr': 'R@1%FPR', 'ap_at_1fpr': 'AP@1%FPR',
         'i_auroc_log_vs_good': 'I-AUROC log/good',
         'i_auroc_str_vs_good': 'I-AUROC str/good'}


def load(fn):
    d = json.load(open(os.path.join(HERE, 'results', fn), encoding='utf-8'))
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
                if isinstance(val, (int, float)):
                    Y[a, b, d] = val
    return Y


def vardecomp(Y):
    if np.all(np.isnan(Y)) or Y.shape[1] < 2:
        return (float('nan'),) * 3
    gm = np.nanmean(Y)
    sc = Y.shape[0] * Y.shape[2] * np.nansum((np.nanmean(Y, axis=(0, 2)) - gm) ** 2)
    sm = Y.shape[1] * Y.shape[2] * np.nansum((np.nanmean(Y, axis=(1, 2)) - gm) ** 2)
    rs = np.nansum((Y
                    - np.nanmean(Y, axis=(0, 2))[None, :, None]
                    - np.nanmean(Y, axis=(1, 2))[:, None, None]
                    + gm) ** 2)
    t = sc + sm + rs + 1e-12
    return 100 * sc / t, 100 * sm / t, 100 * rs / t


mv = load('matrix.json')
lo = load('matrix_loco.json')
print('cells: MVTec-AD %d | MVTec-LOCO %d' % (len(mv), len(lo)))

print()
print('=' * 84)
print('(a) Per-metric seed noise -- is the deployment metric noisier than AUROC?')
print('=' * 84)
print('%-9s %16s %16s %8s' % ('metric', 'MVTec-AD (15 cat)', 'LOCO (5 cat)', 'LOCO/MV'))
print('-' * 54)
for m in ['i_auroc', 'p_auroc', 'pro', 'spro', 'r_at_1fpr', 'ap_at_1fpr']:
    a = float(np.nanmean(np.nanstd(cube(mv, m), axis=2, ddof=1))) if not np.all(np.isnan(cube(mv, m))) else float('nan')
    b = float(np.nanmean(np.nanstd(cube(lo, m), axis=2, ddof=1))) if not np.all(np.isnan(cube(lo, m))) else float('nan')
    print('%-9s %16s %16s %8s'
          % (SHORT[m], ('%.4f' % a) if a == a else '--', ('%.4f' % b) if b == b else '--',
             ('%.2f' % (b / a)) if (a == a and a) and b == b else '--'))
print()
print('LOCO-specific: are the logical and structural metrics noisier still?')
for m in ['spro_log', 'spro_str', 'i_auroc_log_vs_good', 'i_auroc_str_vs_good']:
    b = float(np.nanmean(np.nanstd(cube(lo, m), axis=2, ddof=1)))
    print('   %-22s seed SD = %.4f' % (SHORT[m], b))

print()
print('=' * 84)
print('(b) Variance decomposition: does the CATEGORY still dominate?')
print('=' * 84)
print('%-19s %-24s %-24s' % ('metric', 'MVTec-AD  cat/conf/seed', 'LOCO  cat/conf/seed'))
print('-' * 70)
for m in ['i_auroc', 'p_auroc', 'pro', 'spro', 'r_at_1fpr', 'ap_at_1fpr',
          'spro_log', 'spro_str']:
    ra = vardecomp(cube(mv, m))
    rb = vardecomp(cube(lo, m))
    fmt = lambda r: ('  --        ' if r[0] != r[0]
                     else '%5.1f%% / %4.1f%% / %4.1f%%' % r)
    print('%-19s %-24s %-24s' % (SHORT[m], fmt(ra), fmt(rb)))

print()
print('=' * 84)
print('(c) Absolute levels on LOCO (mean over 5 categories x 3 configurations x 3 seeds)')
print('=' * 84)
for m in ['i_auroc', 'p_auroc', 'spro', 'spro_log', 'spro_str',
          'i_auroc_log_vs_good', 'i_auroc_str_vs_good', 'r_at_1fpr']:
    Y = cube(lo, m)
    if np.all(np.isnan(Y)):
        continue
    print('   %-22s %.4f' % (SHORT[m], float(np.nanmean(Y))))
print()
print('   logical-vs-structural gap in localisation:')
Yl = float(np.nanmean(cube(lo, 'spro_log')))
Ys = float(np.nanmean(cube(lo, 'spro_str')))
print('   sPRO_log=%.4f  sPRO_str=%.4f  gap=%.4f (%.0f%% relative)'
      % (Yl, Ys, Ys - Yl, 100 * (Ys - Yl) / Ys))
