# -*- coding: utf-8 -*-
"""Generate per-budget search configs for the capacity experiment.

A backbone enters a budget's search space if its CHEAPEST (encoder+decoder)
configuration fits the budget. The budget is then enforced strictly at
selection time, using the measured per-(backbone, dim) parameter counts.
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

import json, os, sys

import yaml

ROOT = _os.path.join(_KGD_ROOT, 'capacity_experiment')
PARAMS = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')
BASE_CFG = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'configs', 'stage2_nas_search.yaml')

BUDGETS = [3, 6, 8, 13, 30]
TRIALS = {3: 30, 6: 60, 8: 100, 13: 100, 30: 100}

params = json.load(open(PARAMS, encoding='utf-8'))
base = yaml.safe_load(open(BASE_CFG, encoding='utf-8'))

backbone_min = {}
for bb, dims in params.items():
    vals = [v['total_m'] for v in dims.values() if 'total_m' in v]
    if vals:
        backbone_min[bb] = min(vals)

plan = {}
for B in BUDGETS:
    bbs = sorted(bb for bb, mn in backbone_min.items() if mn <= B)
    if not bbs:
        print('!! budget %dM admits no backbone' % B)
        continue

    cfg = json.loads(json.dumps(base))          # deep copy
    cfg['nas']['search_space']['backbones'] = bbs
    cfg['nas']['num_trials'] = TRIALS[B]
    cfg['output']['dir'] = 'experiments/cap_budget_%dM' % B

    path = os.path.join(ROOT, 'configs', 'budget_%dM.yaml' % B)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    n_cfg = sum(1 for bb in bbs for v in params[bb].values()
                if 'total_m' in v and v['total_m'] <= B)
    plan[B] = {'backbones': bbs, 'trials': TRIALS[B], 'configs_in_budget': n_cfg}
    print('budget %2dM : %d backbones, %d configs in budget, %d trials' %
          (B, len(bbs), n_cfg, TRIALS[B]))
    print('            ' + ', '.join(bbs))

with open(os.path.join(ROOT, 'results', 'budget_plan.json'), 'w', encoding='utf-8') as f:
    json.dump({'budgets': BUDGETS, 'plan': plan, 'params': params}, f,
              ensure_ascii=False, indent=2)
print('\nwrote budget configs to', os.path.join(ROOT, 'configs'))
