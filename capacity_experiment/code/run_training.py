# -*- coding: utf-8 -*-
"""Orchestrate the capacity experiment's final-training phase.

Trains every (budget, category) pair at 500 epochs with the architecture the
budget's search selected, N processes at a time. Resumable: train_category()
skips any category whose final_evaluation.json already exists.

Usage: python run_training.py [max_parallel]
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
LOGS = os.path.join(EXP, 'logs', 'train')
os.makedirs(LOGS, exist_ok=True)

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']

sel = json.load(open(os.path.join(EXP, 'results', 'selected_configs.json'), encoding='utf-8'))['selected']
PARALLEL = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PY = os.environ.get('KGD_PYTHON', sys.executable)  # interpreter used for the child workers

jobs = []
for b, info in sorted(sel.items(), key=lambda kv: int(kv[0])):
    for c in CATS:
        ev = os.path.join(EXP, 'results', 'budget_%sM' % b, c, 'final_evaluation.json')
        if os.path.exists(ev):
            continue
        jobs.append((b, info['backbone'], c))

print('budgets: %s' % ', '.join('%sM' % k for k in sorted(sel, key=int)))
print('pending jobs: %d (parallel=%d)' % (len(jobs), PARALLEL))

env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
running, done, failed = [], 0, []

t0 = time.time()
while jobs or running:
    while jobs and len(running) < PARALLEL:
        b, bb, c = jobs.pop(0)
        log = open(os.path.join(LOGS, '%sM_%s.log' % (b, c)), 'w', encoding='utf-8')
        p = subprocess.Popen([PY, '-u', os.path.join(EXP, 'code', 'train_worker.py'), b, bb, c],
                             stdout=log, stderr=subprocess.STDOUT, env=env,
                             cwd=_os.path.join(_KGD_ROOT, 'unsupervised_nas'))
        running.append((p, b, c, log, time.time()))
        print('  [start] %sM/%-11s %s' % (b, c, bb), flush=True)

    time.sleep(20)
    still = []
    for p, b, c, log, ts in running:
        if p.poll() is None:
            still.append((p, b, c, log, ts))
            continue
        log.close()
        el = (time.time() - ts) / 60
        if p.returncode == 0:
            done += 1
            print('  [ok]    %sM/%-11s %.1f min   (%d done, %d left)'
                  % (b, c, el, done, len(jobs) + len(still)), flush=True)
        else:
            failed.append((b, c))
            print('  [FAIL]  %sM/%-11s rc=%d' % (b, c, p.returncode), flush=True)
    running = still

print('\nfinished in %.1f min | ok=%d failed=%d' % ((time.time() - t0) / 60, done, len(failed)))
if failed:
    print('failed:', failed)
