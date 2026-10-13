# -*- coding: utf-8 -*-
"""Select, per budget, the best-scoring configuration that RESPECTS the budget.

The search ran over a backbone list filtered by each backbone's cheapest
configuration, so individual (backbone, dim) draws can exceed the budget.
This step enforces the cap strictly using the measured encoder+decoder counts.
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

EXP = _os.path.join(_KGD_ROOT, 'capacity_experiment')
NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
TRACE = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')
BUDGETS = [3, 6, 8, 13, 30]

params = json.load(open(os.path.join(TRACE, 'params_by_backbone.json'), encoding='utf-8'))
plan = json.load(open(os.path.join(EXP, 'results', 'budget_plan.json'), encoding='utf-8'))

selected = {}
summary = []
for B in BUDGETS:
    sr = os.path.join(NAS, 'experiments', 'cap_budget_%dM' % B, 'search_results.json')
    if not os.path.exists(sr):
        print('budget %2dM : search_results.json absent — search still running?' % B)
        continue
    d = json.load(open(sr, encoding='utf-8'))
    trials = d.get('all_results') or []
    rows = []
    for t in trials:
        a = t.get('architecture') or {}
        bb, dim, sc = a.get('backbone'), a.get('feature_dim'), t.get('score')
        if bb is None or dim is None or sc is None:
            continue
        info = params.get(bb, {}).get(str(dim))
        if not info or 'total_m' not in info:
            continue
        rows.append({'backbone': bb, 'dim': int(dim), 'score': float(sc),
                     'params_m': info['total_m']})
    in_budget = [r for r in rows if r['params_m'] <= B]
    if not in_budget:
        print('budget %2dM : NO in-budget trial (n=%d)' % (B, len(rows)))
        continue
    best = min(in_budget, key=lambda r: r['score'])
    n_cfg = len(set((r['backbone'], r['dim']) for r in in_budget))
    selected[B] = best
    summary.append({'budget_m': B, 'trials_total': len(rows),
                    'trials_in_budget': len(in_budget), 'configs_in_budget': n_cfg,
                    'chosen': best})
    print('budget %2dM : %3d trials, %3d in budget (%2d configs) -> %s + %dd  (%.2fM, score %.6g)'
          % (B, len(rows), len(in_budget), n_cfg, best['backbone'], best['dim'],
             best['params_m'], best['score']))

out = os.path.join(EXP, 'results', 'selected_configs.json')
json.dump({'selected': {str(k): v for k, v in selected.items()},
           'summary': summary, 'budget_plan': plan},
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nwrote', out)
