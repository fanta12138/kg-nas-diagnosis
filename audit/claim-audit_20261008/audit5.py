# -*- coding: utf-8 -*-
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

import io, sys, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
EXP = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments')
EXPR = _os.path.join(_KGD_ROOT, 'experiments')
def load(p):
    with open(p, encoding='utf-8') as f: return json.load(f)
print('BTAD best-test-loss epochs:')
for p in ['01','02','03']:
    d=load(os.path.join(EXPR,'btad_stage3',p,'training_log.json'))
    tl=[(e['epoch'], e['test_loss']) for e in d]
    be=min(tl,key=lambda x:x[1])
    print('  Product %s: best test_loss=%.6f at epoch %d (n=%d)' % (p, be[1], be[0], len(tl)))
print()
print('VisA stage2 backbone pool (all_results.json):')
import glob
for c in ['candle','capsules','chewinggum','macaroni1','pcb1','pipe_fryum']:
    p=os.path.join(EXPR,'visa_stage2',c,'all_results.json')
    if os.path.exists(p):
        d=load(p)
        bbs=set(r['architecture']['backbone'] for r in d['all_results'])
        print('  %s: n=%d backbones=%s' % (c, len(d['all_results']), sorted(bbs)))
    else:
        print('  %s: no all_results.json' % c)
# search subdirs
for c in ['cashew','fryum','macaroni2','pcb2','pcb3','pcb4']:
    base=os.path.join(EXPR,'visa_stage2',c,c)
    if os.path.isdir(base):
        print('  %s/%s:' % (c,c), os.listdir(base))
print()
print('BTAD stage2 all-result trial counts / backbones:')
import glob
for p in glob.glob(os.path.join(EXPR,'btad_stage2','*','*','*.json')):
    print('  ', p)
# try to find btad config
for p in glob.glob(os.path.join(EXPR,'btad_stage2','**','*'), recursive=True):
    if os.path.isfile(p):
        print('   file:', p)
