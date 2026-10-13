# -*- coding: utf-8 -*-
"""Final analysis of the capacity experiment: KG-NAS accuracy vs parameter budget."""
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

import json, os, statistics

EXP = _os.path.join(_KGD_ROOT, 'capacity_experiment')
TRACE = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')
CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']

TIERS = [
    (1, 'budget_3M', 'MobileNetV3-Small', 2.81, '0–3M'),
    (2, 'budget_6M', 'EfficientNet-B0',   5.41, '3–6M'),
    (3, 'budget_8M', 'DeiT-Tiny',         7.31, '6–9M'),
    (4, 'tier_4M',   'ResNet-18',        13.04, '9–15M'),
    (5, 'tier_5M',   'ResNet-50',        25.77, '15–31M'),
]

rows = []
for t, d, bb, m, band in TIERS:
    base = os.path.join(EXP, 'results', d)
    per = {}
    for c in CATS:
        f = os.path.join(base, c, 'final_evaluation.json')
        if os.path.exists(f):
            per[c] = json.load(open(f, encoding='utf-8'))['auroc']
    if per:
        rows.append({'tier': t, 'dir': d, 'backbone': bb, 'params_m': m, 'band': band,
                     'n': len(per), 'mean': statistics.mean(per.values()),
                     'std': statistics.pstdev(list(per.values())), 'per_category': per})

print('=' * 78)
print('KG-NAS capacity curve (MVTec-AD, 15 categories, 500 epochs, identical protocol)')
print('=' * 78)
print('%-5s %-20s %9s %11s %9s %6s' % ('tier', 'backbone', 'params', 'mean AUROC', 'std', 'n'))
for r in rows:
    print('T%-4d %-20s %8.2fM %10.2f%% %8.2f%% %4d/15'
          % (r['tier'], r['backbone'], r['params_m'], 100 * r['mean'], 100 * r['std'], r['n']))

print()
print('reference points')
print('  KG-NAS as published (full space, 5.41M) : 82.40%')
print('  EfficientAD                            : 97.50%  (8.0M)')
print('  CFA                                    : 89.75%  (12.0M)')
print('  Reverse Distillation                   : 80.49%  (15.0M)')

# per-category table
print()
print('per-category AUROC')
hdr = '%-11s' % 'category' + ''.join('%10s' % ('T%d' % r['tier']) for r in rows)
print(hdr)
for c in CATS:
    line = '%-11s' % c
    for r in rows:
        v = r['per_category'].get(c)
        line += '%9.4f ' % v if v is not None else '%10s' % '-'
    print(line)

# trend
fmt = [r for r in rows if r['n'] == 15]
if len(fmt) >= 2:
    lo, hi = fmt[0], fmt[-1]
    print()
    print('=' * 78)
    print('VERDICT')
    print('=' * 78)
    print('  smallest: %s  %.2fM -> %.2f%%' % (lo['backbone'], lo['params_m'], 100 * lo['mean']))
    print('  largest : %s  %.2fM -> %.2f%%' % (hi['backbone'], hi['params_m'], 100 * hi['mean']))
    print('  parameters grew %.2fx, accuracy changed %+.2f pp'
          % (hi['params_m'] / lo['params_m'], 100 * (hi['mean'] - lo['mean'])))
    best = max(fmt, key=lambda r: r['mean'])
    print('  best point: %s (%.2fM) at %.2f%% -- %s the smallest tier'
          % (best['backbone'], best['params_m'], 100 * best['mean'],
             'is' if best['tier'] == lo['tier'] else 'is NOT'))

json.dump({'curve': rows}, open(os.path.join(EXP, 'results', 'tier_curve.json'), 'w',
                                encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nwrote tier_curve.json')
