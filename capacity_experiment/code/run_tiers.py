# -*- coding: utf-8 -*-
"""Sequential runner for the remaining capacity tiers (4 and 5).

Tier 1/2/3 are already complete: budget_3M = MobileNetV3-Small @256d,
budget_6M = EfficientNet-B0 @256d, budget_8M = DeiT-Tiny @256d (the earlier
worker ignored the search's feature_dim, which for these three happened to be
the correct fixed 256-d bottleneck the paper actually trains).

Usage: python run_tiers.py <tier_label> <backbone> [more tiers as label:backbone ...]
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

import json, os, subprocess, sys, time

EXP = _os.path.join(_KGD_ROOT, 'capacity_experiment')
NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
LOGS = os.path.join(EXP, 'logs', 'tier')
os.makedirs(LOGS, exist_ok=True)

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']
PY = os.environ.get('KGD_PYTHON', sys.executable)  # interpreter used for the child workers

PAIRS = [a.split(':') for a in sys.argv[1:]]
if not PAIRS:
    PAIRS = [['4', 'resnet18'], ['5', 'resnet50']]

env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')

jobs = []
for tier, bb in PAIRS:
    for c in CATS:
        ev = os.path.join(EXP, 'results', 'tier_%sM' % tier, c, 'final_evaluation.json')
        if not os.path.exists(ev):
            jobs.append((tier, bb, c))

print('pending jobs: %d (sequential)' % len(jobs), flush=True)
t0 = time.time()
ok, bad = 0, []
for i, (tier, bb, c) in enumerate(jobs, 1):
    fh = open(os.path.join(LOGS, 'tier%sM_%s.log' % (tier, c)), 'w', encoding='utf-8')
    s = time.time()
    p = subprocess.run([PY, '-u', os.path.join(EXP, 'code', 'train_worker.py'), tier, bb, c],
                       stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=NAS)
    fh.close()
    el = (time.time() - s) / 60
    if p.returncode == 0:
        ok += 1
        print('[%2d/%2d] ok   tier%s/%s (%.1f min, %.1f h elapsed)'
              % (i, len(jobs), tier, c, el, (time.time() - t0) / 3600), flush=True)
    else:
        bad.append((tier, c))
        print('[%2d/%2d] FAIL tier%s/%s rc=%d' % (i, len(jobs), tier, c, p.returncode), flush=True)

print('\nfinished in %.1f h | ok=%d failed=%d' % ((time.time() - t0) / 3600, ok, len(bad)))
if bad:
    print('failed:', bad)
