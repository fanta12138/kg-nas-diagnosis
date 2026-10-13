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

import io, sys, os, json, glob, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
EXP = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments')
EXPR = _os.path.join(_KGD_ROOT, 'experiments')
def load(p):
    with open(p, encoding='utf-8') as f: return json.load(f)

print('='*70)
print('M. pooled per-category medians across 3 strategies')
for c in ['wood','hazelnut','capsule']:
    allv=[]
    for s in ['random','evolutionary','bayesian']:
        d=load(os.path.join(EXP,'p1_search_strategies',c,s,'results.json'))
        allv += [r['score'] for r in d['all_results'] if 0<r['score']<1000]
    print('%s: pooled n=%d median=%.4f' % (c, len(allv), np.median(allv)))

print('='*70)
print('N. max within-(backbone,dim) variation across strategies, per category')
for c in ['wood','hazelnut','capsule']:
    memo={}
    for s in ['random','evolutionary','bayesian']:
        d=load(os.path.join(EXP,'p1_search_strategies',c,s,'results.json'))
        for r in d['all_results']:
            k=(r['architecture']['backbone'], r['architecture']['feature_dim'])
            memo.setdefault(k,[]).append(r['score'])
    best=0; bkey=None
    for k,v in memo.items():
        vv=[x for x in v if x>0]
        if len(vv)>=2:
            ratio=max(vv)/min(vv)
            if ratio>best: best=ratio; bkey=(k, vv)
    print('%s: max ratio=%.2f  key=%s  vals=%s' % (c, best, bkey[0] if bkey else None, [round(x,5) for x in bkey[1]] if bkey else None))

print('='*70)
print('O. BTAD convergence epoch from training_log.json')
for p in ['01','02','03']:
    d=load(os.path.join(EXPR,'btad_stage3',p,'training_log.json'))
    if isinstance(d,dict):
        print(p, 'keys:', list(d.keys()))
    print(p, json.dumps(d)[:400])

print('='*70)
print('P. BTAD / VisA search-space evidence')
print('btad nas_results:')
print(load(os.path.join(EXPR,'btad_stage2','01','01','nas_results.json')))
print('visa stage2 dirs:')
print(os.listdir(os.path.join(EXPR,'visa_stage2')))
