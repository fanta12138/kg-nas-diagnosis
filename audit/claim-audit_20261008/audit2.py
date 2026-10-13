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

print('='*70)
print('F. Alignment ablation aggregates (n=60)')
BBS = ['resnet18','efficientnet_b0','vit_tiny_patch16_224','convnextv2_tiny']
fixed=[]; dyn=[]; percat={}
for c in CATS:
    s = load(os.path.join(EXP,'ablation_4_dimension_alignment',c,'ablation_summary.json'))
    fa = s['fixed_alignment']; da = s['dynamic_alignment']
    row=[]
    for b in BBS:
        fixed.append(fa[b]['auroc']); dyn.append(da[b]['auroc'])
        row.append(da[b]['auroc']-fa[b]['auroc'])
    percat[c]=np.mean(row)
fixed=np.array(fixed); dyn=np.array(dyn); diff=(dyn-fixed)*100
print('n=%d  mean diff=%.4f  std=%.4f' % (len(diff), diff.mean(), diff.std(ddof=1)))
t,p = stats.ttest_rel(dyn, fixed)
print('paired t=%.4f  p=%.5f' % (t,p))
se = diff.std(ddof=1)/np.sqrt(len(diff))
tc = stats.t.ppf(0.975, len(diff)-1)
print('95%% CI = [%.2f, %.2f]  (t_crit=%.4f)' % (diff.mean()-tc*se, diff.mean()+tc*se, tc))
print('positive categories (of 15):', sum(1 for c in CATS if percat[c]>0), [c for c in CATS if percat[c]<=0])
print('per-category avg improvement (x100):')
for c in CATS: print('  %-11s %.2f' % (c, percat[c]*100))
print('ResNet-18 avg impr = %.2f' % (np.mean([(load(os.path.join(EXP,'ablation_4_dimension_alignment',c,'ablation_summary.json'))['dynamic_alignment']['resnet18']['auroc']-load(os.path.join(EXP,'ablation_4_dimension_alignment',c,'ablation_summary.json'))['fixed_alignment']['resnet18']['auroc']) for c in CATS])*100))

print('='*70)
print('G. Loss ablation (table_ablation_3_loss_functions.csv)')
import csv as _csv
rows=list(_csv.DictReader(open(os.path.join(_os.path.join(_KGD_ROOT, 'unsupervised_nas', 'paper_results'),'table_ablation_3_loss_functions.csv'),encoding='utf-8-sig')))
for k in ['kd_only_AUROC','kd_consistency_AUROC','full_AUROC','kd_only_AP','kd_consistency_AP','full_AP']:
    vals=[float(r[k])*100 for r in rows if r['Category']!='Average']
    print('%s: mean=%.2f std(pop)=%.2f std(samp)=%.2f n=%d' % (k, np.mean(vals), np.std(vals), np.std(vals,ddof=1), len(vals)))
# best counts
mk=[float(r['kd_only_AUROC']) for r in rows if r['Category']!='Average']
mc=[float(r['kd_consistency_AUROC']) for r in rows if r['Category']!='Average']
mf=[float(r['full_AUROC']) for r in rows if r['Category']!='Average']
bc={'kd':0,'kc':0,'full':0}
for i in range(15):
    m=max(mk[i],mc[i],mf[i])
    for k,v in [('kd',mk[i]),('kc',mc[i]),('full',mf[i])]:
        if v==m: bc[k]+=1
print('best counts:', bc)

print('='*70)
print('H. CNN vs Transformer from nas_search_trials.csv (93 rows)')
rows=list(_csv.DictReader(open(_os.path.join(_KGD_ROOT, 'fig', 'nas_search_trials.csv'),encoding='utf-8-sig')))
CNN={'resnet18','resnet34','resnet50','mobileone_s0','efficientnet_b0','efficientnet_b1','convnext_tiny','convnext_small','convnextv2_tiny','convnextv2_base'}
TR={'vit_tiny_patch16_224','deit_tiny_patch16_224','swin_tiny_patch4_window7_224'}
cnn=[]; tr=[]
for r in rows:
    s=float(r['Score']); b=r['Backbone']
    if b in CNN: cnn.append(s)
    elif b in TR: tr.append(s)
cnn=np.array(cnn); tr=np.array(tr)
print('CNN n=%d types=%d median=%.4f std(samp)=%.3f' % (len(cnn), len(CNN), np.median(cnn), cnn.std(ddof=1)))
print('TR  n=%d types=%d median=%.4f std(samp)=%.3f' % (len(tr), len(TR), np.median(tr), tr.std(ddof=1)))
u,p=stats.mannwhitneyu(cnn,tr)
print('Mann-Whitney p=%.4f' % p)
# with divergent excluded (score<1000)
cnn2=np.array([float(r['Score']) for r in rows if r['Backbone'] in CNN and 0<float(r['Score'])<1000])
tr2=np.array([float(r['Score']) for r in rows if r['Backbone'] in TR and 0<float(r['Score'])<1000])
print('After excluding >1000: CNN n=%d median=%.4f std=%.3f ; TR n=%d median=%.4f std=%.3f' % (len(cnn2),np.median(cnn2),cnn2.std(ddof=1),len(tr2),np.median(tr2),tr2.std(ddof=1)))
u,p=stats.mannwhitneyu(cnn2,tr2); print('MW p(excl)=%.4f' % p)
