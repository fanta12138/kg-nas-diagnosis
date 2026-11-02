# -*- coding: utf-8 -*-
"""Re-derive every number the manuscript states, from the released result files.

Each entry is (label, value_as_printed_in_the_paper, recomputed_value, tolerance).
A FAIL means either the paper or this script is wrong; both are fixed before
submission, never papered over.

Run:  python verify_numbers.py
"""
import io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from scipy.stats import spearmanr, norm
import numpy.linalg as la

REPO = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return json.load(open(os.path.join(REPO, p), encoding='utf-8'))


def sp(x, y):
    r, p = spearmanr(x, y)
    return r, p


def partial(x, y, z):
    def rank(a):
        return np.argsort(np.argsort(a)).astype(float)

    def resid(v, u):
        X = np.c_[np.ones_like(u), u]
        return v - X @ la.lstsq(X, u, rcond=None)[0]
    rx, ry, rz = rank(x), rank(y), rank(z)
    return sp(resid(rx, rz), resid(ry, rz))


rows = []


def check(label, printed, computed, tol):
    ok = abs(float(printed) - float(computed)) <= tol
    rows.append((label, printed, computed, ok))


# ------------------------------------------------------------------ sources
truth = load('proxy_diag/phaseB_truth.json')
fin = load('proxy_diag/final_truth_analysis.json')
samp = load('proxy_diag/samplesize_analysis.json')
rho_c = load('proxy_diag/rho_wellbehaved_control.json')
b0 = load('fusion_policy/fusion_policy.json')
rn = load('fusion_policy/fusion_policy_resnet18.json')
vi = load('fusion_policy/fusion_policy_visa.json')
area = load('fusion_policy/defect_area.json')

POL = 'sum_topk'
DEF = 'sum_mean'


def pol(d, c, k):
    v = d[c]
    if 'policies' in v:
        return 100 * v['policies'][k]
    return 100 * v[k]


def delta(d, c):
    return pol(d, c, POL) - pol(d, c, DEF)


# ============================================================ §4 diagnosis
check('variance: backbone %', 0.54, fin['variance']['backbone'], 0.005)
check('variance: category %', 95.32, fin['variance']['category'], 0.005)
check('variance: interaction %', 4.13, fin['variance']['interaction'], 0.005)
check('F(13,182)', 1.844, fin['F'], 0.001)
check('p of backbone term', 0.039, fin['p'], 0.001)
check('spread of 14 backbones (pp)', 3.98, samp['headline']['backbone_spread_pp'], 0.005)
check('pooled seed sd (pp)', 1.96, samp['seed_noise_pp']['pooled_sd'], 0.005)
check('mean of cell sd (pp)', 1.27, samp['seed_noise_pp']['mean_of_cell_sd'], 0.005)
check('median of cell sd (pp)', 0.58, samp['seed_noise_pp']['median_of_cell_sd'], 0.005)
check('max of cell sd (pp)', 7.74, samp['seed_noise_pp']['max_of_cell_sd'], 0.005)
check('mdd, 15 cat paired (pp)', 2.98, samp['resolvability']['delta_0.5pp']['mdd_with_15_categories_paired'], 0.005)
check('runs per arm for 0.5 pp', 241, samp['resolvability']['delta_0.5pp']['n_per_arm_single_category_unpaired'], 1.0)
check('categories for 0.5 pp', 533, samp['resolvability']['delta_0.5pp']['categories_needed_paired'], 1.0)
check('mdd with 3 seeds (pp)', 2.95, samp['resolvability']['delta_0.5pp']['mdd_with_15cat_x3seed_paired'], 0.005)
check('mean cross-category rho', 0.215, fin['cross_cat_mean_spearman'], 0.001)
check('category pairs reversed %', 31.4, 100 * fin['reversed_pairs'] / fin['total_pairs'], 0.2)

# ranking / headline accuracy
rank = dict((k, v) for k, v in fin['ranking'])
check('AUROC of efficientnet_b0', 82.47, 100 * rank['efficientnet_b0'], 0.05)
check('AUROC of resnet18 (best)', 83.34, 100 * rank['resnet18'], 0.05)
check('AUROC of convnextv2_base (worst)', 79.36, 100 * rank['convnextv2_base'], 0.05)
order = sorted(rank.values(), reverse=True)
check('gap 1st vs 13th', 3.23, 100 * (order[0] - order[12]), 0.02)

# rho controls
check('rho, all 14 backbones', 0.215, rho_c['all_14_backbones']['mean_rho'], 0.002)
check('rho, excluding 2 collapse-prone', 0.256, rho_c['excl_2_collapse_prone']['mean_rho'], 0.002)
check('rho, excluding 3 backbones', 0.278, rho_c['excl_collapse_and_b0']['mean_rho'], 0.002)

# ============================================================ §5.2 pooling
mv_cats = [c for c in b0 if 'policies' in b0[c]]
vi_cats = [c for c in vi if 'policies' in vi[c]]
rn_cats = [c for c in rn if 'policies' in rn[c]]
check('MVTec: default mean AUROC', 82.80, np.mean([pol(b0, c, DEF) for c in mv_cats]), 0.02)
check('MVTec: top-k mean AUROC', 90.21, np.mean([pol(b0, c, POL) for c in mv_cats]), 0.02)
check('MVTec: delta (pp)', 7.42, np.mean([delta(b0, c) for c in mv_cats]), 0.02)
check('MVTec: categories improved', 10, sum(delta(b0, c) > 0 for c in mv_cats), 0)
check('MVTec: screw delta', 41.8, delta(b0, 'screw'), 0.1)
check('MVTec: cable delta', -5.06, delta(b0, 'cable'), 0.02)
check('MVTec: pill delta', -4.80, delta(b0, 'pill'), 0.02)
check('MVTec: tile delta', -4.62, delta(b0, 'tile'), 0.02)
check('MVTec: zipper delta', -3.02, delta(b0, 'zipper'), 0.02)
check('MVTec: transistor delta', -2.00, delta(b0, 'transistor'), 0.02)

# ============================================================ §5.3 VisA
check('VisA: default mean AUROC', 82.35, np.mean([pol(vi, c, DEF) for c in vi_cats]), 0.02)
check('VisA: top-k mean AUROC', 90.57, np.mean([pol(vi, c, POL) for c in vi_cats]), 0.02)
check('VisA: delta (pp)', 8.22, np.mean([delta(vi, c) for c in vi_cats]), 0.02)
check('VisA: categories improved', 9, sum(delta(vi, c) > 0 for c in vi_cats), 0)
check('VisA: candle delta', -6.6, delta(vi, 'candle'), 0.06)
check('VisA: capsules delta', -4.8, delta(vi, 'capsules'), 0.06)
check('VisA: fryum delta', -0.3, delta(vi, 'fryum'), 0.06)

# policy ordering across datasets
common = [p for p in b0[mv_cats[0]]['policies'] if all(p in vi[c]['policies'] for c in vi_cats)]
a = [np.mean([b0[c]['policies'][p] for c in mv_cats]) for p in common]
b = [np.mean([vi[c]['policies'][p] for c in vi_cats]) for p in common]
r, p = sp(a, b)
check('policy ordering rho (MVTec vs VisA)', 0.77, r, 0.005)
check('policy ordering -log10 p', 16.5, -np.log10(p), 0.4)
top8 = [common[i] for i in np.argsort(a)[-8:]]
check('MVTec: of 8 strongest, top-k based', 8, sum('topk' in x for x in top8), 0)
top8v = [common[i] for i in np.argsort(b)[-8:]]
check('VisA: of 8 strongest, top-k based', 8, sum('topk' in x for x in top8v), 0)

# second student family
check('ResNet-18: default mean AUROC', 83.14, np.mean([pol(rn, c, DEF) for c in rn_cats]), 0.02)
check('ResNet-18: top-k mean AUROC', 91.14, np.mean([pol(rn, c, POL) for c in rn_cats]), 0.02)
check('ResNet-18: delta (pp)', 8.00, np.mean([delta(rn, c) for c in rn_cats]), 0.02)
check('ResNet-18: categories improved', 10, sum(delta(rn, c) > 0 for c in rn_cats), 0)
shared = [c for c in mv_cats if c in rn_cats]
r_bb, p_bb = sp([delta(b0, c) for c in shared], [delta(rn, c) for c in shared])
check('per-category rho between backbones', 0.98, r_bb, 0.005)
check('both backbones positive, same cats', 10,
      sum((delta(b0, c) > 0) and (delta(rn, c) > 0) for c in shared), 0)
check('both backbones negative, same cats', 5,
      sum((delta(b0, c) < 0) and (delta(rn, c) < 0) for c in shared), 0)

# pooled 42 runs
runs = []
for tag, d, key in [('MVTec-B0', b0, 'MVTec-AD'), ('MVTec-RN18', rn, 'MVTec-AD'), ('VisA-B0', vi, 'VisA')]:
    for c in d:
        if 'policies' not in d[c]:
            continue
        runs.append((tag, c, pol(d, c, DEF), delta(d, c),
                     100 * area[key][c]['median_frac']))
check('total category runs', 42, len(runs), 0)
base = np.array([r[2] for r in runs]); dlt = np.array([r[3] for r in runs])
are = np.array([r[4] for r in runs])
r1, p1 = sp(base, dlt)
check('pooled rho(delta, baseline)', -0.67, r1, 0.005)
check('pooled -log10 p', 6, -np.log10(p1), 0.35)
r2, p2 = sp(are, dlt)
check('pooled rho(delta, area)', -0.29, r2, 0.005)
check('area p', 0.06, p2, 0.01)
_rb, _rd, _ra = (np.argsort(np.argsort(v)).astype(float) for v in (base, dlt, are))
_rp = spearmanr(_rd - np.c_[np.ones_like(_rb), _rb] @ la.lstsq(np.c_[np.ones_like(_rb), _rb], _rd, rcond=None)[0],
                _ra - np.c_[np.ones_like(_rb), _rb] @ la.lstsq(np.c_[np.ones_like(_rb), _rb], _ra, rcond=None)[0])
rp, pp = _rp.statistic, _rp.pvalue
check('partial rho(delta, area | baseline)', 0.06, rp, 0.01)
check('partial p', 0.71, pp, 0.02)
check('runs improved', 29, int((dlt > 0).sum()), 0)
check('runs not improved', 13, int((dlt < 0).sum()), 0)
check('runs gaining >16 pp', 11, int((dlt > 16).sum()), 0)

edges = [40.0, 70.0, 85.0, 95.0, 101.0]
bands = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (base >= lo) & (base < hi)
    bands.append(dlt[m].mean() if m.any() else float('nan'))
for i, (want, got) in enumerate(zip([28.9, 10.3, -0.7, 0.9], bands)):
    check('baseline band %d mean delta' % (i + 1), want, got, 0.06)

check('largest gain (pp)', 41.8, dlt.max(), 0.1)
check('largest loss (pp)', -6.6, dlt.min(), 0.06)
check('gain/loss ratio, "six times"', 6, abs(dlt.max() / dlt.min()), 0.5)

# derived constants printed in the text
c1 = norm.pdf(norm.ppf(0.99)) / 0.01
c5 = norm.pdf(norm.ppf(0.95)) / 0.05
check('c(k) at 1%%', 2.67, c1, 0.005)
check('c(k) at 5%%', 2.06, c5, 0.005)
for s, want in [(3, 1.1), (6, 5.6), (10, 7.3)]:
    check('a* at s=%d (%% of image)' % s, want, 100 * 0.1 * (1 - c1 / s), 0.06)
check('screw median area %', 0.29, 100 * area['MVTec-AD']['screw']['median_frac'], 0.01)
check('cashew median area %', 0.15, 100 * area['VisA']['cashew']['median_frac'], 0.05)

# ============================================================ report
bad = [r for r in rows if not r[3]]
print('%-46s %12s %12s' % ('claim', 'paper', 'recomputed'))
print('-' * 74)
for label, printed, computed, ok in rows:
    print('%-46s %12.4g %12.4g  %s' % (label, printed, computed, 'ok' if ok else 'FAIL'))
print('-' * 74)
print('%d checks, %d failed' % (len(rows), len(bad)))
if bad:
    print('\nFAILING:')
    for label, printed, computed, _ in bad:
        print('  %-44s paper %g   computed %g' % (label, printed, computed))
sys.exit(1 if bad else 0)
