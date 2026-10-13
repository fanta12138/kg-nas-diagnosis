# -*- coding: utf-8 -*-
"""What would it take to believe a reported improvement?

Uses the measured per-seed variance of each metric to compute the number of
independent runs a claim would need. Also reports the metric-dependent variance
decomposition (category / method / seed / interaction).
"""
import io, json, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np

METRICS = ['i_auroc', 'p_auroc', 'pro', 'r_at_1fpr', 'ap_at_1fpr']
SHORT = {'i_auroc': 'I-AUROC', 'p_auroc': 'P-AUROC', 'pro': 'PRO',
         'r_at_1fpr': 'R@1%FPR', 'ap_at_1fpr': 'AP@1%FPR'}
METHODS = ['global', 4, 2, 1, 0]
SEEDS = [0, 1, 2]


def load(fn):
    d = json.load(open(os.path.join(HERE, 'results', fn), encoding='utf-8'))
    return [v for v in d.values() if v.get('ok')]


cells = load('matrix.json')
cats = sorted({v['category'] for v in cells})

idx = {}
for v in cells:
    idx[(v['category'], v['radius'], v['seed'])] = v


def cube(metric):
    """(n_method, n_cat, n_seed) array."""
    Y = np.full((len(METHODS), len(cats), len(SEEDS)), np.nan)
    for a, meth in enumerate(METHODS):
        for b, c in enumerate(cats):
            for d, s in enumerate(SEEDS):
                v = idx.get((c, meth, s))
                if v is not None and v.get(metric) is not None:
                    Y[a, b, d] = v[metric]
    return Y


print('=' * 84)
print('1.  Per-metric seed noise, and the sample size a claim would need')
print('=' * 84)
print('%-9s %10s %11s %14s %18s' % ('metric', 'seed SD', 'one step', 'N for 80% power', 'half a step'))
print('-' * 70)
summary = {}
for m in METRICS:
    Y = cube(m)
    sds = np.nanstd(Y, axis=2, ddof=1)                 # sd across seeds, per (method, cat)
    sd = float(np.nanmean(sds))
    means = np.nanmean(Y, axis=(1, 2))
    one_step = float(np.nanmax(means) - np.nanmin(means)) / (len(METHODS) - 1)
    n1 = int(np.ceil(2 * (2.8 * sd / one_step) ** 2)) if one_step > 0 else -1
    n2 = int(np.ceil(2 * (2.8 * sd / (one_step / 2)) ** 2)) if one_step > 0 else -1
    summary[m] = dict(sd=sd, one_step=one_step, n_full=n1, n_half=n2)
    print('%-9s %10.4f %11.4f %14d %18d' % (SHORT[m], sd, one_step, n1, n2))

print()
print('For reference, the improvements the field actually claims on MVTec-AD:')
print('  PatchCore (2022) 99.1  ->  best 2026 ~99.6   = +0.5 pp  (I-AUROC)')
print('  the 2026 survey calls anything under +0.3 pp "insufficient evidence"')
for m in ['i_auroc', 'pro', 'r_at_1fpr']:
    sd = summary[m]['sd']
    for claim in (0.005, 0.003):
        if sd > 0:
            n = int(np.ceil(2 * (2.8 * sd / claim) ** 2))
            print('  to detect +%.1f pp on %-8s at 80%% power : %d runs per arm'
                  % (claim * 100, SHORT[m], n))

print()
print('=' * 84)
print('2.  Metric-dependent variance decomposition (5 configurations x 3 seeds x 15 categories)')
print('=' * 84)
print('%-9s %11s %11s %11s %11s %12s' % ('metric', 'category', 'method', 'seed', 'interact', 'F(method)'))
print('-' * 70)
for m in METRICS:
    Y = cube(m)
    gm = np.nanmean(Y)
    sc = len(METHODS) * len(SEEDS) * np.nansum((np.nanmean(Y, axis=(0, 2)) - gm) ** 2)
    sm = len(cats) * len(SEEDS) * np.nansum((np.nanmean(Y, axis=(1, 2)) - gm) ** 2)
    ss = len(cats) * len(METHODS) * np.nansum((np.nanmean(Y, axis=(0, 1)) - gm) ** 2)
    res = np.nansum((Y
                     - np.nanmean(Y, axis=(0, 2))[None, :, None]
                     - np.nanmean(Y, axis=(1, 2))[:, None, None]
                     - np.nanmean(Y, axis=(0, 1))[None, None, :] + 2 * gm) ** 2)
    tot = sc + sm + ss + res
    dof = (len(METHODS) - 1) * (len(cats) - 1) * len(SEEDS)
    F = (sm / (len(METHODS) - 1)) / (res / dof)
    print('%-9s %10.1f%% %10.1f%% %10.1f%% %10.1f%% %12.1f'
          % (SHORT[m], 100 * sc / tot, 100 * sm / tot, 100 * ss / tot,
             100 * res / tot, F))

json.dump(summary, open(os.path.join(HERE, 'results', 'power_summary.json'), 'w'), indent=2)
print()
print('saved -> results/power_summary.json')
