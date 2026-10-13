# -*- coding: utf-8 -*-
"""Capacity-experiment analysis: KG-NAS accuracy vs parameter budget.

Writes capacity_report.md with the curve, the comparison against the published
baselines, and the verdict on whether the gap to EfficientAD is a budget effect.
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

import json, os

EXP = _os.path.join(_KGD_ROOT, 'capacity_experiment')
CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']

sel = json.load(open(os.path.join(EXP, 'results', 'selected_configs.json'), encoding='utf-8'))
curve = json.load(open(os.path.join(EXP, 'results', 'capacity_curve.json'), encoding='utf-8'))['curve']

PUBLISHED = {'KG-NAS (paper, full space)': (5.41, 82.40),
             'EfficientAD': (8.0, 97.50),
             'SimpleNet': (12.0, 98.00),
             'CFA': (12.0, 89.75),
             'PaDiM': (12.0, 77.92),
             'Reverse Distillation': (15.0, 80.49)}

L = []
L.append('# KG-NAS 参数量扫描实验报告\n')
L.append('> 目的：判定与 EfficientAD（8M / 97.5%）的差距来自**方法**还是**参数预算**')
L.append('> 协议：受限搜索 → 500-epoch 全 15 类目训练（bs 64 / seed 42 / ResNet-18 教师）\n')

L.append('## 一、精度—参数量曲线\n')
L.append('| 预算 | 选中架构 | 参数量 | 均值 AUROC | 均值 AP | 类别数 |')
L.append('|---|---|---|---|---|---|')
for r in curve:
    L.append('| %dM | %s + %dd | %.2fM | **%.2f%%** | %.2f%% | %d/15 |'
             % (r['budget_m'], r['backbone'], r['dim'], r['params_m'],
                100 * r['mean_auroc'], 100 * r['mean_ap'], r['n_categories']))

L.append('\n## 二、与已发表方法对比\n')
L.append('| 方法 | 参数量 | AUROC |')
L.append('|---|---|---|')
for r in curve:
    L.append('| **本文 %dM 预算** | %.2fM | **%.2f%%** |' % (r['budget_m'], r['params_m'], 100 * r['mean_auroc']))
for k, (p, a) in sorted(PUBLISHED.items(), key=lambda kv: kv[1][0]):
    L.append('| %s | %.2fM | %.2f%% |' % (k, p, a))

L.append('\n## 三、逐类目\n')
L.append('| Category | ' + ' | '.join('%sM' % r['budget_m'] for r in curve) + ' |')
L.append('|---|' + '---|' * len(curve))
for c in CATS:
    cells = []
    for r in curve:
        v = r['per_category'].get(c)
        cells.append('%.4f' % v if v else '-')
    L.append('| %s | %s |' % (c, ' | '.join(cells)))

L.append('\n## 四、判读\n')
used = [(r['budget_m'], r['params_m'], 100 * r['mean_auroc']) for r in curve if r['n_categories'] == 15]
if len(used) >= 2:
    lo, hi = used[0], used[-1]
    L.append('- 最低预算 %dM（%.2fM 实参）→ %.2f%%' % lo)
    L.append('- 最高预算 %dM（%.2fM 实参）→ %.2f%%' % hi)
    L.append('- 参数增加 **%.2f×**，精度变化 **%+.2f pp**' % (hi[1] / lo[1], hi[2] - lo[2]))
L.append('- 论文全空间结果：5.41M → 82.40%')
L.append('- EfficientAD：8.00M → 97.50%')

out = os.path.join(EXP, 'results', 'capacity_report.md')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
print('\nwrote', out)
