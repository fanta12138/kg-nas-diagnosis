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
CATS = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
def load(p):
    with open(p, encoding='utf-8') as f: return json.load(f)
def rd1(x): 
    import decimal
    return float(decimal.Decimal(str(x)).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP))

print('='*70)
print('G2. Loss ablation best-count detail')
rows=list(csv.DictReader(open(_os.path.join(_KGD_ROOT, 'unsupervised_nas', 'paper_results', 'table_ablation_3_loss_functions.csv'),encoding='utf-8-sig')))
rr=[r for r in rows if r['Category']!='Average']
for r in rr:
    mk=float(r['kd_only_AUROC']); mc=float(r['kd_consistency_AUROC']); mf=float(r['full_AUROC'])
    m=max(mk,mc,mf)
    tops=[n for n,v in [('kd',mk),('kc',mc),('full',mf)] if v==m]
    tag=' <-- tie' if len(tops)>1 else ''
    if 'full' in tops or len(tops)>1:
        print('  %-11s kd=%.4f kc=%.4f full=%.4f tops=%s%s' % (r['Category'],mk,mc,mf,tops,tag))

print('='*70)
print('I. Failure counts (AUROC<0.5) & CFA best counts (baselines_full_results.csv)')
raw={}
for row in csv.DictReader(open(_os.path.join(_KGD_ROOT, 'fig', 'baselines_full_results.csv'),encoding='utf-8-sig')):
    raw.setdefault(row['Method'],{})[row['Category']]=float(row['AUROC'])*100
disp={'autoencoder':'AE','vae':'VAE','padim':'PaDiM','patchcore':'PatchCore','stpm':'STPM','reverse_distillation':'RevDist','cfa':'CFA','spade':'SPADE','fastflow':'FastFlow','efficientad':'EffAD','simplenet':'SimpleNet'}
fail={disp[m]:sum(1 for v in raw[m].values() if v<50) for m in raw}
print('failures <50%:', fail, 'TOTAL=', sum(fail.values()))
# best per category within 0.02pp
best_counts={}
for c in CATS:
    vals={disp[m]:raw[m][c] for m in raw}
    top=max(vals.values())
    for m,v in vals.items():
        if top-v<=0.02: best_counts[m]=best_counts.get(m,0)+1
print('best-counts (tie<=0.02):', best_counts, 'sum', sum(best_counts.values()))
# strict best (no tie)
strict={}
for c in CATS:
    vals={disp[m]:raw[m][c] for m in raw}
    top=max(vals.values()); tops=[m for m,v in vals.items() if abs(v-top)<1e-9]
    for m in tops: strict[m]=strict.get(m,0)+1
print('strict best:', strict)
# CFA vs PaDiM leather diff
print('CFA-PaDiM leather diff pp = %.4f' % (raw['cfa']['leather']-raw['padim']['leather']))
print('CFA-RevDist bottle diff pp = %.4f' % (raw['cfa']['bottle']-raw['reverse_distillation']['bottle']))
# which categories CFA strictly best
print('CFA best cats (strict):', [c for c in CATS if abs(raw['cfa'][c]-max(disp[m] and raw[m][c] for m in raw))<1e-9])
# averages
for m in raw:
    print('avg %-8s = %.4f' % (disp[m], np.mean(list(raw[m].values()))))

print('='*70)
print('J. main table average row check (round-half-up of avg)')
for m in ['autoencoder','vae','padim','patchcore','stpm','reverse_distillation','cfa','spade','fastflow','efficientad','simplenet']:
    a=np.mean(list(raw[m].values()))
    print('  %-8s avg=%.4f -> %.1f' % (disp[m], a, rd1(a)))

print('='*70)
print('K. CI of KG-NAS main results (sigma of 15 cat AUROC)')
rd={c: load(os.path.join(EXP,'p2_rd','nas','ep500',c,'final_evaluation.json'))['auroc']*100 for c in CATS}
sd=np.std(list(rd.values()), ddof=1)
print('mean=%.4f sample_std=%.4f pop_std=%.4f' % (np.mean(list(rd.values())), sd, np.std(list(rd.values()))))
import math
print('CI = 82.4 +/- 2.145*%.4f/sqrt(15) = [%.2f, %.2f]' % (sd, 82.4-2.145*sd/math.sqrt(15), 82.4+2.145*sd/math.sqrt(15)))
print('min=%.1f (%s), max=%.1f (%s)' % (min(rd.values()), min(rd,key=rd.get), max(rd.values()), max(rd,key=rd.get)))

print('='*70)
print('L. proxy medians & 81x (p1_search_strategies random results)')
for c in ['wood','hazelnut','capsule']:
    p=os.path.join(EXP,'p1_search_strategies',c,'random','results.json')
    d=load(p)
    scores=[r['score'] for r in d['all_results']]
    v=[s for s in scores if 0<s<1000]
    print('%s: n_all=%d n_valid=%d median=%.4f min=%.4f max=%.4f ratio=%.1f' % (c, len(scores), len(v), np.median(v), min(v), max(v), max(v)/min(v)))
