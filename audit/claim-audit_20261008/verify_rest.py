"""Executor verification — Phase 2, remaining findings F4-F13 and M1."""
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

import csv, os, glob, json, re, collections

FIG = _os.path.join(_KGD_ROOT, 'fig')
PR = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'paper_results')
EX = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments')


def rows(p):
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


print('=' * 70)
print('F4 — best trial score 0.0097 vs 0.0023')
print('=' * 70)
trials = rows(os.path.join(FIG, 'nas_search_trials.csv'))
best = min(trials, key=lambda r: float(r['Score']))
print('  nas_search_trials.csv: rows=%d  global min score = %s (%s, %s dim)' %
      (len(trials), best['Score'], best.get('Backbone'), best.get('Feature_Dim') or best.get('Dim')))
b = [r for r in trials if float(r['Score']) < 0.005]
print('  rows with score < 0.005: %d' % len(b))
for r in b:
    print('     ', r)

fd = os.path.join(PR, 'ablation_phase1', 'ablation_3_feature_dim.csv')
if os.path.exists(fd):
    print('\n  ablation_3_feature_dim.csv:')
    for r in rows(fd):
        print('     ', r)

print()
print('=' * 70)
print('F6/F7/F8 — rounding of PaDiM / carpet / CFA')
print('=' * 70)
bl = rows(os.path.join(FIG, 'baselines_full_results.csv'))
print('  baselines_full_results.csv columns:', list(bl[0].keys())[:8])
# average per method
agg = collections.defaultdict(list)
mcol = [c for c in bl[0] if 'ethod' in c or c.lower().startswith('method')]
mcol = mcol[0] if mcol else list(bl[0].keys())[0]
acol = [c for c in bl[0] if 'AUROC' in c.upper() or 'auroc' in c]
acol = acol[0] if acol else list(bl[0].keys())[1]
for r in bl:
    try:
        agg[r[mcol]].append(float(r[acol]))
    except (ValueError, KeyError):
        pass
for m in ['PaDiM', 'CFA', 'Reverse Distillation', 'SPADE']:
    if m in agg:
        v = sum(agg[m]) / len(agg[m])
        print('  %-22s n=%2d mean=%.4f  -> 1dp half-up = %.1f   2dp=%.2f' %
              (m, len(agg[m]), v, round(v + 1e-12, 1), round(v + 1e-12, 2)))

# carpet KG-NAS
cp = glob.glob(os.path.join(EX, 'p2_rd/nas/ep500/carpet/final_evaluation.json'))
if cp:
    print('\n  carpet final_evaluation.json raw:')
    print('     ', open(cp[0], encoding='utf-8').read()[:300].replace('\n', ' '))

print()
print('=' * 70)
print('F9 — CFA best-count ties and the leather margin')
print('=' * 70)
bc = os.path.join(FIG, 'baseline_best_counts.csv')
if os.path.exists(bc):
    for r in rows(bc):
        print('     ', r)

print()
print('=' * 70)
print('F10/F11 — backbone type count and trials-per-config')
print('=' * 70)
bs = os.path.join(FIG, 'backbone_distribution.csv')
if os.path.exists(bs):
    r = rows(bs)
    print('  backbone_distribution.csv rows=%d' % len(r))
    fams = collections.Counter(x.get('Family') or x.get('Type', '?') for x in r)
    print('  families:', dict(fams))

print()
print('=' * 70)
print('M1 — is 107 FPS recorded anywhere?')
print('=' * 70)
for p in [os.path.join(FIG, 'inference_efficiency.csv'),
          os.path.join(PR, 'efficiency_2_inference.csv'),
          os.path.join(FIG, 'efficiency_metrics.csv')]:
    if os.path.exists(p):
        print('  --', os.path.basename(p))
        for r in rows(p):
            print('     ', r)
