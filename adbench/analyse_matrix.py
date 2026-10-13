# -*- coding: utf-8 -*-
"""Analyse the method x locality x seed x category matrix.

Answers three questions that decide where to go next:

  P1  Does locality cost anything, and on which metric?  (the locality axis)
  P2  What is the seed-noise floor of each metric, and how does it compare with
      the effects people report?  (is the leaderboard resolvable?)
  P3  Do the metrics agree on the *ranking* of configurations?  (is mean AUROC a
      faithful proxy for deployment behaviour?)
"""
import io, json, os, sys, itertools

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from scipy import stats

METRICS = ['i_auroc', 'p_auroc', 'pro', 'r_at_1fpr', 'ap_at_1fpr', 'r_at_5fpr']
SHORT = {'i_auroc': 'I-AUROC', 'p_auroc': 'P-AUROC', 'pro': 'PRO',
         'r_at_1fpr': 'R@1%FPR', 'ap_at_1fpr': 'AP@1%FPR', 'r_at_5fpr': 'R@5%FPR'}

d = json.load(open(os.path.join(HERE, 'results', 'matrix.json'), encoding='utf-8'))
cells = [v for v in d.values() if v.get('ok')]
cats = sorted({v['category'] for v in cells})
methods = ['global', 4, 2, 1, 0, 'knn_global']


def lab(meth):
    """Label for a locality setting. (Note: dict.get's default is evaluated
    eagerly, so it cannot be used with a %-format on an int-only branch.)"""
    if meth == 'global':
        return 'global'
    if meth == 'knn_global':
        return 'knn-global'
    return 'radius=%d' % meth


def get(cat, meth, seed, metric):
    for v in cells:
        if v['category'] == cat and v['radius'] == meth and v['seed'] == seed:
            return v.get(metric, float('nan'))
    return float('nan')


def arr(meth, metric):
    """(n_cat, n_seed) array for one method x metric."""
    seeds = sorted({v['seed'] for v in cells if v['radius'] == meth})
    return np.array([[get(c, meth, s, metric) for s in seeds] for c in cats])


print('=' * 100)
print('P1  Does locality cost anything?   (mean over 15 categories x 3 seeds)')
print('=' * 100)
hdr = '%-12s' % 'locality' + ''.join('%11s' % SHORT[m] for m in METRICS)
print(hdr)
print('-' * len(hdr))
tab = {}
for meth in methods:
    row = []
    for m in METRICS:
        a = arr(meth, m)
        row.append(np.nanmean(a))
    tab[meth] = row
    print('%-12s' % lab(meth) + ''.join('%11.4f' % v for v in row))
print()
print('cost of making the model strictly local (global -> radius=0), in points:')
for i, m in enumerate(METRICS):
    print('   %-9s %+8.4f' % (SHORT[m], tab[0][i] - tab['global'][i]))

print()
print('=' * 100)
print('P2  Seed-noise floor vs the effects people report')
print('=' * 100)
print('%-12s' % 'locality' + ''.join('%14s' % SHORT[m] for m in ['i_auroc', 'pro', 'r_at_1fpr']))
print('-' * 60)
seed_sd = {}
for meth in methods:
    line = []
    for m in ['i_auroc', 'pro', 'r_at_1fpr']:
        a = arr(meth, m)
        sd = np.nanstd(a, axis=1, ddof=1)          # sd across the 3 seeds, per category
        seed_sd.setdefault(m, []).append(np.nanmean(sd))
        line.append(np.nanmean(sd))
    print('%-12s' % lab(meth) + ''.join('%14.4f' % v for v in line))
print()
for m in ['i_auroc', 'pro', 'r_at_1fpr']:
    print('  mean seed SD over configs, %-9s = %.4f   (max %.4f)'
          % (SHORT[m], np.mean(seed_sd[m]), np.max(seed_sd[m])))

# variance decomposition per metric: category / method / seed / residual
print()
print('Variance decomposition of each metric (balanced 15 cat x 6 method x 3 seed):')
print('%-12s %10s %10s %10s %10s %10s' % ('metric', 'category', 'method', 'seed', 'interact', 'F(method)'))
print('-' * 70)
for m in METRICS:
    Y = np.stack([arr(meth, m) for meth in methods])       # (M, C, S)
    gm = np.nanmean(Y)
    ss_cat = methods.__len__() * 3 * np.nansum((np.nanmean(Y, axis=(0, 2)) - gm) ** 2)
    ss_m = len(cats) * 3 * np.nansum((np.nanmean(Y, axis=(1, 2)) - gm) ** 2)
    ss_s = len(cats) * len(methods) * np.nansum((np.nanmean(Y, axis=(0, 1)) - gm) ** 2)
    resid = np.nansum((Y - np.nanmean(Y, axis=(0, 2))[None, :, None]
                       - np.nanmean(Y, axis=(1, 2))[:, None, None]
                       - np.nanmean(Y, axis=(0, 1))[None, None, :] + 2 * gm) ** 2)
    tot = ss_cat + ss_m + ss_s + resid
    F = (ss_m / (len(methods) - 1)) / (resid / ((len(methods) - 1) * (len(cats) - 1) * 3))
    print('%-12s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %10.2f'
          % (SHORT[m], 100 * ss_cat / tot, 100 * ss_m / tot, 100 * ss_s / tot,
             100 * resid / tot, F))

print()
print('=' * 100)
print('P3  Do the metrics agree on the ranking of configurations?')
print('=' * 100)
R = {m: np.array([tab[meth][i] for meth in methods]) for i, m in enumerate(METRICS)}
pairs = list(itertools.combinations(METRICS, 2))
print('%-28s %8s %8s' % ('metric pair', 'Spearman', 'p'))
print('-' * 46)
for a, b in pairs:
    rho, p = stats.spearmanr(R[a], R[b])
    flag = '   <-- DISAGREE' if rho < 0.7 else ''
    print('%-28s %8.2f %8.3f%s' % ('%s vs %s' % (SHORT[a], SHORT[b]), rho, p, flag))
print()
for m in ['i_auroc', 'pro', 'r_at_1fpr']:
    best = methods[int(np.nanargmax(R[m]))]
    print('  winner by %-9s : %s' % (SHORT[m], lab(best)))

print()
print('Per-category: which locality is best, and how much does the choice matter?')
print('%-12s %-10s %-10s %-10s %8s' % ('category', 'best I-AUROC', 'best PRO', 'best R@1%', 'spread PRO'))
print('-' * 60)
for ci, c in enumerate(cats):
    vals = {mm: np.nanmean([get(c, mm, sd, 'pro') for sd in range(3)]) for mm in methods}
    vi = {mm: np.nanmean([get(c, mm, sd, 'i_auroc') for sd in range(3)]) for mm in methods}
    vr = {mm: np.nanmean([get(c, mm, sd, 'r_at_1fpr') for sd in range(3)]) for mm in methods}
    def f(k):
        if k == 'global':
            return 'gl'
        if k == 'knn_global':
            return 'knn'
        return 'r=%d' % k
    print('%-12s %-10s %-10s %-10s %8.3f' % (c, f(max(vi, key=vi.get)), f(max(vals, key=vals.get)),
                                            f(max(vr, key=vr.get)), max(vals.values()) - min(vals.values())))
json.dump({'locality_table': {str(k): v for k, v in tab.items()},
           'seed_sd': {SHORT[k]: float(np.mean(v)) for k, v in seed_sd.items()}},
          open(os.path.join(HERE, 'results', 'analysis_p1_p3.json'), 'w'), indent=2)
