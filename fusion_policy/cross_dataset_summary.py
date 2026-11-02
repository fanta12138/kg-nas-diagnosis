# -*- coding: utf-8 -*-
"""Cross-dataset summary of the scale-fusion sweep: MVTec-AD (15 cats) vs VisA (12).

Reads
    fusion_policy/fusion_policy.json        -- MVTec, 82 policies per category
    fusion_policy/fusion_policy_visa.json   -- VisA, the same 82 policies

Reports, per dataset
    * mean AUROC under the published default (sum of the three scales, mean pooling)
    * mean AUROC under the fixed top-k convention (k = 1%)
    * the paired per-category difference, with a bootstrap 95% CI over categories
    * how many categories improve, and by how much at the extremes
    * whether every one of the eight strongest policies uses top-k pooling
      (the structural claim made for MVTec in Section 5.2)

It also lists the policies whose rank correlation between the two datasets can be
computed, so the manuscript can say whether the *ordering* of policies transfers,
not only the mean.

Writes
    fusion_policy/cross_dataset_summary.json
"""
import json, os, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

OUT = os.path.join(os.environ.get('KGD_WORK_ROOT', r'D:/autopaper'), 'fusion_policy')
MVTEC = os.path.join(OUT, 'fusion_policy.json')
VISA = os.path.join(OUT, 'fusion_policy_visa.json')
DEST = os.path.join(OUT, 'cross_dataset_summary.json')

K = 'sum_topk'          # fixed top-k (k = 1%)
MEAN = 'sum_mean'       # published default


def load(path, n_expected):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding='utf-8'))
    cats = [c for c, v in d.items() if v.get('done') and 'policies' in v]
    print('  %-46s %2d/%2d categories' % (os.path.basename(path), len(cats), n_expected))
    return d, cats


def summarise(name, d, cats, n_expected):
    if not cats:
        print('  %s: no completed categories, skipped' % name)
        return None
    means = np.array([d[c]['policies'][MEAN] for c in cats])
    topk = np.array([d[c]['policies'][K] for c in cats])
    delta = topk - means

    rng = np.random.default_rng(42)
    boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # do the eight strongest policies all use top-k? (pooled over categories)
    per_policy = {}
    for p in d[cats[0]]['policies']:
        vals = [d[c]['policies'][p] for c in cats if p in d[c]['policies']]
        if len(vals) == len(cats):
            per_policy[p] = float(np.mean(vals))
    top8 = sorted(per_policy, key=per_policy.get, reverse=True)[:8]
    topk_share = sum('topk' in p for p in top8)

    gains = sorted(delta, reverse=True)
    out = {
        'dataset': name,
        'n_categories': len(cats),
        'n_expected': n_expected,
        'categories': cats,
        'mean_default': float(means.mean()),
        'mean_topk': float(topk.mean()),
        'delta_mean': float(delta.mean()),
        'delta_ci95': [float(lo), float(hi)],
        'n_improved': int((delta > 0).sum()),
        'n_worse': int((delta < 0).sum()),
        'largest_gain': float(gains[0]),
        'largest_loss': float(gains[-1]),
        'top8_all_topk': int(topk_share),
        'per_category': {c: {'default': float(d[c]['policies'][MEAN]),
                             'topk': float(d[c]['policies'][K]),
                             'delta': float(d[c]['policies'][K] - d[c]['policies'][MEAN])}
                         for c in cats},
        'policy_means': per_policy,
    }
    print('  %s: %d/%d cats | default %.2f | top-k %.2f | delta %+.2f '
          '[%.2f, %.2f] | improved %d/%d | top-8 top-k %d/8 | largest %+.2f / %+.2f'
          % (name, len(cats), n_expected, out['mean_default'], out['mean_topk'],
             out['delta_mean'], lo, hi, out['n_improved'], len(cats),
             topk_share, gains[0], gains[-1]))
    return out


def main():
    print('=== loading ===')
    mv = load(MVTEC, 15)
    vi = load(VISA, 12)
    if vi is None:
        print('\nVisA results not found -- run fusion_policy_visa.py first.')
        return 1

    print('\n=== summary ===')
    res = {}
    if mv:
        res['MVTec-AD'] = summarise('MVTec-AD', *mv, 15)
    res['VisA'] = summarise('VisA', *vi, 12)

    # policy-ordering transfer between datasets
    if mv:
        a = res['MVTec-AD']['policy_means']
        b = res['VisA']['policy_means']
        common = sorted(set(a) & set(b))
        if len(common) > 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr([a[k] for k in common], [b[k] for k in common])
            res['policy_rank_correlation'] = {'rho': float(rho), 'p': float(p),
                                              'n_policies': len(common)}
            print('\n  policy ordering, MVTec vs VisA: Spearman rho = %.3f (p = %.3g, %d policies)'
                  % (rho, p, len(common)))

    # is top-k above mean on both, and by how much
    for k, v in res.items():
        if isinstance(v, dict) and 'delta_mean' in v:
            print('  %-9s delta = %+.2f points, %d/%d categories improved'
                  % (k, v['delta_mean'], v['n_improved'], v['n_categories']))

    json.dump(res, open(DEST, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\nwritten: %s' % DEST)
    return 0


if __name__ == '__main__':
    sys.exit(main())
