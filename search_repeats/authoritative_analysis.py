# -*- coding: utf-8 -*-
"""Authoritative search-vs-random analysis.

Why this exists: `search_repeats.py --analyse-only` (pipeline stage 35) recomputes
its summary from whatever is in proxy_diag/phaseB_truth.json at the time. Once
stage 33 added seeds 43/44 for six backbones, that summary silently switched from
a single-seed matrix to a mixed one (six backbones averaged over three seeds, the
other eight over one), which is not a like-for-like comparison and no longer
matches the ranking table in the manuscript.

The manuscript's ranking, variance decomposition and controlled comparison all
use the single-seed (seed 42) matrix in proxy_diag/final_truth_analysis.json,
where all 14 backbones are on equal footing. This script re-derives the
search-vs-random comparison from that same matrix, so the figure, the tables and
the text all quote one source.

Writes search_repeats/search_vs_random_final.json.
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

import json, os

import numpy as np

D = _KGD_ROOT
FT = os.path.join(D, 'proxy_diag', 'final_truth_analysis.json')
SR = os.path.join(D, 'search_repeats', 'search_repeats.json')
OUT = os.path.join(D, 'search_repeats', 'search_vs_random_final.json')
DRAWS = 200000
SEED = 0

COLLAPSE_PRONE = {'efficientnet_b1', 'mobilenetv3_small_100'}

ft = json.load(open(FT, encoding='utf-8'))
sr = json.load(open(SR, encoding='utf-8'))

truth = dict(ft['ranking'])                       # seed 42, all 14 backbones
order = [b for b, _ in ft['ranking']]             # best -> worst
rank = {b: i for i, b in enumerate(order, 1)}
sel = [v['selected']['backbone'] for v in sr.values() if v.get('done')]
n = len(sel)

pool = np.array([truth[b] for b in order])
search_mean = float(np.mean([truth[b] for b in sel]))
random_mean = float(pool.mean())
oracle = float(pool.max())

rng = np.random.default_rng(SEED)
boot = np.array([rng.choice(pool, size=n, replace=True).mean() for _ in range(DRAWS)])
p = float((boot <= search_mean).mean())

n_collapse = sum(1 for b in sel if b in COLLAPSE_PRONE)
from math import comb
p_collapse = sum(comb(n, i) * (len(COLLAPSE_PRONE) / 14) ** i
                 * (1 - len(COLLAPSE_PRONE) / 14) ** (n - i)
                 for i in range(n_collapse, n + 1))

out = {
    'source_truth_matrix': 'proxy_diag/final_truth_analysis.json (seed 42, 15 categories, 14 backbones)',
    'n_runs': n,
    'selected_backbones': sel,
    'true_rank_of_selected': [rank[b] for b in sel],
    'mean_true_rank_of_selected': float(np.mean([rank[b] for b in sel])),
    'random_expected_rank': float((len(order) + 1) / 2),
    'search_mean_true_auroc': search_mean,
    'random_mean_true_auroc': random_mean,
    'oracle_true_auroc': oracle,
    'oracle_backbone': order[0],
    'bootstrap_draws': DRAWS,
    'bootstrap_seed': SEED,
    'bootstrap_sd': float(boot.std(ddof=0)),
    'percentile_of_search_in_random': 100 * p,
    'p_random_leq_search': p,
    'n_collapse_prone_selected': n_collapse,
    'p_all_in_collapse_subset': p_collapse,
    'runs_selecting_top4': int(sum(1 for b in sel if rank[b] <= 4)),
}
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2)

print('runs                      : %d' % n)
print('selected backbones        : %s' % {b: sel.count(b) for b in sorted(set(sel))})
print('mean true rank            : %.2f / %d  (random expectation %.1f)'
      % (out['mean_true_rank_of_selected'], len(order), out['random_expected_rank']))
print('search mean AUROC         : %.4f%%' % (100 * search_mean))
print('uniform random mean AUROC : %.4f%%' % (100 * random_mean))
print('oracle (%s)      : %.4f%%' % (order[0], 100 * oracle))
print('percentile in random dist : %.2f  (p = %.4f, %d draws, seed %d)'
      % (out['percentile_of_search_in_random'], p, DRAWS, SEED))
print('collapse-prone selections : %d / %d   (p = %.2e)'
      % (n_collapse, n, p_collapse))
print('top-4 selections          : %d / %d' % (out['runs_selecting_top4'], n))
print('saved -> %s' % OUT)
