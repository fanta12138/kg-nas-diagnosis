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
from scipy import stats

EXP = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments')
EXPR = _os.path.join(_KGD_ROOT, 'experiments')
CATS = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut',
        'pill','screw','tile','toothbrush','transistor','wood','zipper']

def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def jauroc(p):
    return load(p)['auroc']

print('='*70)
print('A. MVTec RD main (p2_rd/nas/ep500)')
rd = {c: jauroc(os.path.join(EXP,'p2_rd','nas','ep500',c,'final_evaluation.json')) for c in CATS}
print('rd mean = %.6f  -> %.2f%%' % (np.mean(list(rd.values())), np.mean(list(rd.values()))*100))
print('rd round1:', {c: round(v*100,1) for c,v in rd.items()})

print('='*70)
print('B. Prototype (p1_long500)')
proto = {c: jauroc(os.path.join(EXP,'p1_long500',c,'final_evaluation.json')) for c in CATS}
print('proto mean = %.6f -> %.2f%%' % (np.mean(list(proto.values())), np.mean(list(proto.values()))*100))
print('proto round1:', {c: round(v*100,1) for c,v in proto.items()})
# oracle per-category max
oracle = {c: max(proto[c], rd[c]) for c in CATS}
print('ORACLE mean = %.6f -> %.2f%%' % (np.mean(list(oracle.values())), np.mean(list(oracle.values()))*100))
# how many RD wins
rdwin = sum(1 for c in CATS if rd[c] > proto[c])
ci = np.mean(list(rd.values()))*100 - np.mean(list(proto.values()))*100
print('RD wins %d/15, RD-proto avg diff = %.2f, CFA 89.75 gap = %.2f' % (rdwin, ci, 89.75-np.mean(list(oracle.values()))*100))

print('='*70)
print('C. NAS ep200')
rd200 = {c: jauroc(os.path.join(EXP,'p2_rd','nas','ep200',c,'final_evaluation.json')) for c in CATS}
print('rd200 mean = %.6f -> %.2f%%' % (np.mean(list(rd200.values())), np.mean(list(rd200.values()))*100))

print('='*70)
print('D. Fixed 500 (p2_rd/teacher_resnet18/fixed/*/ep500)')
fixed500 = {}
for name, sub in [('resnet18','resnet18'),('swin_tiny','swin_tiny'),('mbv3_small','mbv3_small')]:
    d = {}
    for c in CATS:
        p = os.path.join(EXP,'p2_rd','teacher_resnet18','fixed',sub,'ep500',c,'final_evaluation.json')
        if os.path.exists(p):
            d[c] = jauroc(p)
    fixed500[name] = d
    if len(d)==15:
        print('%s: mean=%.4f (%.2f%%)' % (name, np.mean(list(d.values())), np.mean(list(d.values()))*100))
    else:
        print('%s: n=%d' % (name, len(d)))
# wins at 500 among 4 archs (nas + 3 fixed)
if all(len(v)==15 for v in fixed500.values()):
    wins = {'nas':0,'resnet18':0,'swin_tiny':0,'mbv3_small':0}
    for c in CATS:
        vals = {'nas':rd[c],'resnet18':fixed500['resnet18'][c],
                'swin_tiny':fixed500['swin_tiny'][c],'mbv3_small':fixed500['mbv3_small'][c]}
        m = max(vals.values())
        for k,v in vals.items():
            if v==m: wins[k]+=1
    print('500-ep wins (argmax, ties count all):', wins)
    # strict winner counts (ties -> none or split)
    wins_strict = {k:0 for k in wins}
    for c in CATS:
        vals = {'nas':rd[c],'resnet18':fixed500['resnet18'][c],
                'swin_tiny':fixed500['swin_tiny'][c],'mbv3_small':fixed500['mbv3_small'][c]}
        m = max(vals.values()); top=[k for k,v in vals.items() if v==m]
        for k in top: wins_strict[k]+=1
    print('500-ep winner counts:', wins_strict, 'nas 98.2% of resnet18 =', round(np.mean(list(rd.values()))/np.mean(list(fixed500['resnet18'].values()))*100,2))

print('='*70)
print('E. Fixed 200 (p2_rd/fixed/*/ep200)  +  NAS 200')
fixed200 = {}
for name, sub in [('resnet18','resnet18'),('swin_tiny','swin_tiny'),('mbv3_small','mbv3_small')]:
    d = {}
    for c in CATS:
        p = os.path.join(EXP,'p2_rd','fixed',sub,'ep200',c,'final_evaluation.json')
        if os.path.exists(p):
            d[c] = jauroc(p)
    fixed200[name] = d
    print('%s: n=%d mean=%.2f%%' % (name, len(d), np.mean(list(d.values()))*100 if d else float('nan')))
if all(len(v)==15 for v in fixed200.values()):
    wins200 = {'nas':0,'resnet18':0,'swin_tiny':0,'mbv3_small':0}
    for c in CATS:
        vals = {'nas':rd200[c],'resnet18':fixed200['resnet18'][c],
                'swin_tiny':fixed200['swin_tiny'][c],'mbv3_small':fixed200['mbv3_small'][c]}
        m = max(vals.values())
        for k,v in vals.items():
            if v==m: wins200[k]+=1
    print('200-ep winner counts:', wins200)

print('='*70)
print('F. Alignment ablation 4 backbones (ablation_4_dimension_alignment/*/ablation_summary.json)')
# inspect one file structure first
sample = load(os.path.join(EXP,'ablation_4_dimension_alignment','bottle','ablation_summary.json'))
print('sample keys:', list(sample.keys()))
print(json.dumps(sample, indent=1)[:1200])
