# -*- coding: utf-8 -*-
"""Tier selection for the capacity experiment.

Five non-overlapping size bands, defined on the FINAL model's size
(encoder+decoder at the fixed 256-d bottleneck). For each tier we take the
backbone with the lowest mean valid search score among that tier's backbones,
using the narrowest search that still covers the tier.

A degenerate trial (numerical-validity failure) collapses the score to exactly
0 and a divergent run blows up to ~1e20; both are discarded before ranking.
The author's engine uses a plain single-trial argmin over all draws
(methods/nas_engine.py:273) and is therefore exposed to them; we apply the rule
the paper STATES — lowest average distillation loss — instead.
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

import json, os, statistics

EXP = _os.path.join(_KGD_ROOT, 'capacity_experiment')
NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
TRACE = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')

TIERS = [('1', 0, 3), ('2', 3, 6), ('3', 6, 9), ('4', 9, 15), ('5', 15, 31)]
# which budget search to read for each tier (the narrowest that covers it)
TIER_SOURCE = {'1': 3, '2': 6, '3': 8, '4': 13, '5': 30}

params = json.load(open(os.path.join(TRACE, 'params_by_backbone.json'), encoding='utf-8'))
SIZE = {bb: params[bb]['256']['total_m'] for bb in params if '256' in params[bb]}


def valid_scores(bb):
    """mean/std of the valid search scores for a backbone, pooled over its dims"""
    pass


def load_scores(budget):
    """mean valid score per (backbone, feature_dim) pair.

    The search scores a (backbone, feature_dim) pair, but the FINAL detector is
    sized solely by the backbone (bottleneck is fixed at 256), so we rank pairs
    and then read off the backbone. Pooling dims into one mean per backbone is
    wrong: it mixes configs the search never compared.
    """
    p = os.path.join(NAS, 'experiments', 'cap_budget_%dM' % budget, 'search_results.json')
    d = json.load(open(p, encoding='utf-8'))
    g, dropped = {}, 0
    for t in (d.get('all_results') or []):
        a = t.get('architecture') or {}
        bb, dim, sc = a.get('backbone'), a.get('feature_dim'), t.get('score')
        if bb is None or dim is None or sc is None:
            continue
        sc = float(sc)
        if sc <= 0.0 or sc > 1e6:
            dropped += 1
            continue
        g.setdefault((bb, int(dim)), []).append(sc)
    return g, dropped


out, diag = {}, {}
for label, lo, hi in TIERS:
    members = set(bb for bb, m in SIZE.items() if lo < m <= hi)
    src = TIER_SOURCE[label]
    g, dropped = load_scores(src)
    ranked = []
    for (bb, dim), scs in g.items():
        if bb in members:
            ranked.append({'backbone': bb, 'search_dim': dim, 'size_m': SIZE[bb],
                           'mean_score': statistics.mean(scs),
                           'std': statistics.pstdev(scs) if len(scs) > 1 else 0.0,
                           'n': len(scs)})
    if not ranked:
        print('tier %s (%g-%gM): NO scored pair in the source search' % (label, lo, hi))
        continue
    best = min(ranked, key=lambda r: r['mean_score'])
    out[label] = best
    diag[label] = {'range_m': [lo, hi], 'source_budget_M': src,
                   'dropped_degenerate': dropped,
                   'all_ranked': sorted(ranked, key=lambda r: r['mean_score'])}
    print('tier %s (%g-%gM) <- search %2dM | %d scored pairs, %d degenerate dropped'
          % (label, lo, hi, src, len(ranked), dropped))
    for r in sorted(ranked, key=lambda r: r['mean_score'])[:6]:
        mark = '  <== chosen' if (r['backbone'] == best['backbone'] and r['search_dim'] == best['search_dim']) else ''
        print('      %-32s %4dd  final %.2fM  mean %.4f  std %6.3f  n=%d%s'
              % (r['backbone'], r['search_dim'], r['size_m'], r['mean_score'], r['std'], r['n'], mark))
    print()

json.dump({'selected': out, 'diagnostics': diag},
          open(os.path.join(EXP, 'results', 'tier_selection.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('wrote tier_selection.json')
