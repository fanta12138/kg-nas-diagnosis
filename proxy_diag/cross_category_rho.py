# -*- coding: utf-8 -*-
"""Cross-category transfer of the backbone ordering (paper §4.3).

Statistic
    For every pair of categories, rank the fourteen backbones by their seed-42
    AUROC and correlate the two orderings (Spearman). Report the mean over the
    105 category pairs and the share of pairs whose correlation is negative.

Why a permutation test
    The 105 pairs are not independent: each category appears in fourteen of them.
    A pair-level t-test therefore overstates the evidence. The null used here
    shuffles the backbone order independently within each category, which keeps
    each category's marginal intact and destroys only the cross-category
    agreement, and recomputes the same mean.

Reads   proxy_diag/phaseB_truth.json
Writes  proxy_diag/cross_category_rho.json
"""
import io, itertools, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = os.path.join(HERE, 'phaseB_truth.json')
DEST = os.path.join(HERE, 'cross_category_rho.json')
DRAWS = 20000
SEED = 0
COLLAPSE_PRONE = {'efficientnet_b1', 'mobilenetv3_small_100'}


def matrix(drop=()):
    rows = [v for v in json.load(open(TRUTH, encoding='utf-8')).values()
            if v.get('seed') == 42]
    cats = sorted({r['category'] for r in rows})
    bbs = sorted({r['backbone'] for r in rows} - set(drop))
    M = np.full((len(bbs), len(cats)), np.nan)
    for r in rows:
        if r['backbone'] in bbs:
            M[bbs.index(r['backbone']), cats.index(r['category'])] = r['auroc']
    return M, cats, bbs


def mean_pair_rho(M):
    v = np.array([spearmanr(M[:, i], M[:, j])[0]
                  for i, j in itertools.combinations(range(M.shape[1]), 2)])
    return float(v.mean()), float((v < 0).mean()), v


def permutation_p(M, observed, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for b in range(draws):
        Mp = np.empty_like(M)
        for j in range(M.shape[1]):
            Mp[:, j] = M[rng.permutation(M.shape[0]), j]
        null[b] = mean_pair_rho(Mp)[0]
    return float((null >= observed).mean()), float(null.mean()), float(null.std())


def main():
    out = {}
    for name, drop in [('all_14', ()),
                       ('excl_collapse_prone', COLLAPSE_PRONE),
                       ('excl_collapse_and_b0', COLLAPSE_PRONE | {'efficientnet_b0'})]:
        M, cats, bbs = matrix(drop)
        rho, frac_neg, pairs = mean_pair_rho(M)
        rec = {'n_backbones': len(bbs), 'n_category_pairs': len(pairs),
               'mean_rho': round(rho, 6), 'frac_negative': round(frac_neg, 4)}
        if name == 'all_14':
            p, mu, sd = permutation_p(M, rho)
            rec.update({'permutation_p': p, 'null_mean': round(mu, 6),
                        'null_sd': round(sd, 6), 'permutation_draws': DRAWS,
                        'null_scheme': 'backbone order shuffled within each category',
                        'z_above_null': round((rho - mu) / sd, 2)})
        out[name] = rec
        print('%-22s rho=%.4f  %.1f%% of %d pairs reverse%s'
              % (name, rho, 100 * frac_neg, len(pairs),
                 ('  | permutation p=%.2g, z=%.1f' % (p, (rho - mu) / sd)) if name == 'all_14' else ''))
    json.dump(out, open(DEST, 'w', encoding='utf-8'), indent=2)
    print('written:', DEST)


if __name__ == '__main__':
    main()
