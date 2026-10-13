# -*- coding: utf-8 -*-
"""Is per-category configuration worth more than one global configuration?

This is the decision-theoretic implication of the variance decomposition. If the
category explains ~90% of the variance, the highest-leverage design question might
be "adapt per category" rather than "pick a better method". This script answers it
from the matrix already on disk -- no new runs.

Compared:
  global      one configuration, chosen as the best on the *average* category
  oracle      the per-category best configuration (upper bound on any adaptive rule)
  label-free  left as a separate experiment; here we only bound the payoff

Also reports the same for the backbone axis on MVTec-AD (ResNet-50 vs ViT).
"""
import io, json, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np

METRICS = ['i_auroc', 'p_auroc', 'pro', 'spro', 'r_at_1fpr', 'ap_at_1fpr']
SHORT = {'i_auroc': 'I-AUROC', 'p_auroc': 'P-AUROC', 'pro': 'PRO', 'spro': 'sPRO',
         'r_at_1fpr': 'R@1%FPR', 'ap_at_1fpr': 'AP@1%FPR'}
LOCALITIES = ['global', 4, 2, 1, 0]        # the five PatchCore settings on MVTec
LOCALITIES_3 = ['global', 2, 0]            # the three shared with ViT / VisA / LOCO


def load(fn):
    d = json.load(open(os.path.join(HERE, 'results', fn), encoding='utf-8'))
    return [v for v in d.values() if v.get('ok')]


def table(cells, configs, metric):
    """(n_config, n_cat) array of means over seeds."""
    cats = sorted({v['category'] for v in cells})
    seeds = sorted({v['seed'] for v in cells})
    T = np.full((len(configs), len(cats)), np.nan)
    for a, r in enumerate(configs):
        for b, c in enumerate(cats):
            vals = [v.get(metric) for v in cells
                    if v['category'] == c and v['radius'] == r and v['seed'] in seeds]
            vals = [x for x in vals if isinstance(x, (int, float))]
            if vals:
                T[a, b] = np.mean(vals)
    return T, cats


def analyse(label, cells, configs, metrics):
    print('=' * 86)
    print(label)
    print('=' * 86)
    print('%-10s %10s %10s %10s %10s %9s' %
          ('metric', 'global', 'oracle', 'gain', 'best-config', 'gain %'))
    print('-' * 70)
    for m in metrics:
        T, cats = table(cells, configs, m)
        if np.all(np.isnan(T)):
            continue
        means = np.nanmean(T, axis=1)
        gi = int(np.nanargmax(means))
        glob = means[gi]
        oracle = np.nanmean(np.nanmax(T, axis=0))
        gain = oracle - glob
        rel = 100 * gain / glob if glob else float('nan')
        cfg = configs[gi]
        print('%-10s %10.4f %10.4f %10.4f %10s %8.2f%%'
              % (SHORT[m], glob, oracle, gain,
                 'global' if cfg == 'global' else 'r=%s' % cfg, rel))
    print()


mv = load('matrix.json')
vis = load('matrix_visa.json')
lo = load('matrix_loco.json')
vit = load('matrix_deit.json')

analyse('MVTec-AD, 15 categories, 5 configurations (same backbone)',
        mv, LOCALITIES, ['i_auroc', 'p_auroc', 'pro', 'r_at_1fpr', 'ap_at_1fpr'])
analyse('VisA, 12 categories, 3 configurations',
        vis, LOCALITIES_3, ['i_auroc', 'p_auroc', 'pro', 'r_at_1fpr', 'ap_at_1fpr'])
analyse('MVTec-LOCO, 5 categories, 3 configurations',
        lo, LOCALITIES_3, ['i_auroc', 'p_auroc', 'spro', 'r_at_1fpr', 'ap_at_1fpr'])

# backbone axis: ResNet-50 vs ViT on MVTec-AD
print('=' * 86)
print('MVTec-AD: is per-category BACKBONE choice worth anything? (ResNet-50 vs ViT)')
print('=' * 86)
print('%-10s %22s %22s %9s' % ('metric', 'ResNet-50 / ViT best-global', 'per-category oracle', 'gain'))
print('-' * 68)
for m in ['i_auroc', 'pro', 'r_at_1fpr']:
    Tr, cats = table(mv, ['global'], m)
    Tv, _ = table(vit, ['global'], m)
    if np.all(np.isnan(Tr)) or np.all(np.isnan(Tv)):
        continue
    T = np.vstack([Tr[0], Tv[0]])                  # 2 configs x n_cat
    means = np.nanmean(T, axis=1)
    gi = int(np.nanargmax(means))
    glob = means[gi]
    oracle = np.nanmean(np.nanmax(T, axis=0))
    name = 'ResNet-50' if gi == 0 else 'ViT'
    print('%-10s %22s %22s %+9.4f   (global=%s)'
          % (SHORT[m], '%.4f / %.4f' % (means[0], means[1]), '%.4f' % oracle,
             oracle - glob, name))

print()
print('=' * 86)
print('INTERPRETATION')
print('=' * 86)
print('''The "global" column is what the field reports: one configuration, chosen on the
category average. The "oracle" column is the best achievable if every category were
given its own configuration chosen WITH ground-truth labels. The gap between them is
therefore the entire payoff available to any per-category adaptation rule.

If that gap is small compared with the 90% category share, then per-category
adaptation is not where the leverage is either -- which means the dominant factor is
not a design choice at all, and the actionable conclusion is about *reporting*
rather than about designing.''')
