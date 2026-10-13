# -*- coding: utf-8 -*-
"""Three-dataset synthesis: MVTec-AD (15 cat), VisA (12 cat), MVTec-LOCO (5 cat).

All three use the same harness, same backbone (ResNet-50), same three locality
settings and the same three seeds, so the only thing that changes is the data.

Reports
  (a) per-metric seed noise,
  (b) the variance decomposition (category / configuration / seed),
  (c) absolute levels,
  (d) the sample size a claimed improvement would need.
"""
import io, json, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np

RADII = ['global', 2, 0]
SHORT = {'i_auroc': 'I-AUROC', 'p_auroc': 'P-AUROC', 'pro': 'PRO', 'spro': 'sPRO',
         'r_at_1fpr': 'R@1%FPR', 'ap_at_1fpr': 'AP@1%FPR',
         'spro_log': 'sPRO-logical', 'spro_str': 'sPRO-structural'}
DATASETS = [('MVTec-AD (15)', 'matrix.json', ['i_auroc', 'p_auroc', 'pro', 'r_at_1fpr', 'ap_at_1fpr']),
            ('VisA (12)', 'matrix_visa.json', ['i_auroc', 'p_auroc', 'pro', 'r_at_1fpr', 'ap_at_1fpr']),
            ('LOCO (5)', 'matrix_loco.json', ['i_auroc', 'p_auroc', 'spro', 'r_at_1fpr', 'ap_at_1fpr'])]


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
    """Category / configuration / residual shares.

    No separate 'seed' main effect is reported: each seed's mean is taken over all
    category x configuration cells, so it is ~the grand mean by construction and the
    share would be a meaningless 0%. The seed component is instead the *within-cell*
    variance, reported directly in section (a).
    """
    gm = np.nanmean(Y)
    sc = Y.shape[0] * Y.shape[2] * np.nansum((np.nanmean(Y, axis=(0, 2)) - gm) ** 2)
    sm = Y.shape[1] * Y.shape[2] * np.nansum((np.nanmean(Y, axis=(1, 2)) - gm) ** 2)
    rs = np.nansum((Y - np.nanmean(Y, axis=(0, 2))[None, :, None]
                    - np.nanmean(Y, axis=(1, 2))[:, None, None] + gm) ** 2)
    t = sc + sm + rs + 1e-12
    return 100 * sc / t, 100 * sm / t, 100 * rs / t


print('=' * 92)
print('(a) Per-metric seed noise (mean SD across seeds over configurations x categories)')
print('=' * 92)
print('%-9s %14s %14s %14s' % ('metric', 'MVTec-AD', 'VisA', 'LOCO'))
print('-' * 54)
store = {}
for label, fn, mets in DATASETS:
    store[label] = load(fn)
for m in ['i_auroc', 'p_auroc', 'pro', 'spro', 'r_at_1fpr', 'ap_at_1fpr']:
    row = []
    for label, fn, mets in DATASETS:
        if m not in mets:
            row.append('     --')
            continue
        sd = float(np.nanmean(np.nanstd(cube(store[label], m), axis=2, ddof=1)))
        row.append('%7.4f' % sd)
    print('%-9s %14s %14s %14s' % (SHORT[m], row[0], row[1], row[2]))

print()
print('=' * 92)
print('(b) Variance decomposition: category / configuration  (residual not shown)')
print('=' * 92)
print('%-14s %-24s %-24s %-24s' % ('metric', 'MVTec-AD  cat/conf', 'VisA  cat/conf', 'LOCO  cat/conf'))
print('-' * 88)
for m in ['i_auroc', 'p_auroc', 'pro', 'spro', 'r_at_1fpr', 'ap_at_1fpr']:
    row = []
    for label, fn, mets in DATASETS:
        if m not in mets:
            row.append('          --            ')
            continue
        r = vardecomp(cube(store[label], m))
        row.append('%13.1f%% / %4.1f%%   ' % (r[0], r[1]))
    print('%-14s %-24s %-24s %-24s' % (SHORT[m], row[0], row[1], row[2]))

print()
print('=' * 92)
print('(c) Absolute levels')
print('=' * 92)
for m in ['i_auroc', 'p_auroc', 'pro', 'spro', 'r_at_1fpr', 'ap_at_1fpr']:
    parts = []
    for label, fn, mets in DATASETS:
        if m not in mets:
            parts.append('--')
            continue
        parts.append('%.4f' % float(np.nanmean(cube(store[label], m))))
    print('%-9s MVTec %s | VisA %s | LOCO %s' % (SHORT[m], parts[0], parts[1], parts[2]))

print()
print('=' * 92)
print('(d) Runs per arm needed to detect a claimed improvement (80% power, alpha=.05)')
print('=' * 92)
print('%-9s %-30s %-30s' % ('claim', 'MVTec-AD', 'VisA'))
print('-' * 72)
for m in ['i_auroc', 'pro']:
    row = []
    for label, fn, mets in DATASETS[:2]:
        sd = float(np.nanmean(np.nanstd(cube(store[label], m), axis=2, ddof=1)))
        ns = [int(np.ceil(2 * (2.8 * sd / c) ** 2)) for c in (0.005, 0.003)]
        row.append('+0.5pp: %4d   +0.3pp: %4d' % (ns[0], ns[1]))
    print('%-9s %-30s %-30s' % (SHORT[m], row[0], row[1]))

print()
print('=' * 92)
print('SUMMARY')
print('=' * 92)
for label, fn, mets in DATASETS:
    Y = cube(store[label], 'i_auroc')
    cat, conf, resid = vardecomp(Y)
    sd = float(np.nanmean(np.nanstd(Y, axis=2, ddof=1)))
    print('  %-13s categories=%2d  I-AUROC: category %5.1f%%  configuration %4.1f%%  '
          'residual %4.1f%%  (seed SD within cell %.4f)'
          % (label, Y.shape[1], cat, conf, resid, sd))
