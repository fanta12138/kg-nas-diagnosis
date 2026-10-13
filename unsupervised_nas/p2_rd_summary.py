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

import json
import glob
from pathlib import Path
import numpy as np

root = Path(_os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', 'p2_rd'))
CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

# 旧范式参照（P1 500ep）
OLD_500 = {'bottle': 0.8024, 'cable': 0.6563, 'capsule': 0.6434, 'carpet': 0.8355,
           'grid': 0.6896, 'hazelnut': 0.9812, 'leather': 0.7536, 'metal_nut': 0.6652,
           'pill': 0.6630, 'screw': 0.9038, 'tile': 0.5588, 'toothbrush': 0.5139,
           'transistor': 0.6650, 'wood': 0.7772, 'zipper': 0.7190}


def load(ep):
    d = {}
    for f in glob.glob(str(root / f'nas/ep{ep}/*/final_evaluation.json')):
        r = json.load(open(f, encoding='utf-8'))
        d[Path(f).parent.name] = r
    return d


ep200, ep500 = load(200), load(500)
print(f"{'类别':14s} {'旧500ep':>8s} {'RD200ep':>8s} {'RD500ep':>8s} {'RD500-旧':>8s}")
for c in CATS:
    v_old = OLD_500.get(c, 0)
    v200 = ep200[c]['auroc']
    v500 = ep500[c]['auroc']
    print(f"{c:14s} {v_old:8.4f} {v200:8.4f} {v500:8.4f} {v500-v_old:+8.4f}")

m_old = np.mean(list(OLD_500.values()))
m200 = np.mean([ep200[c]['auroc'] for c in CATS])
m500 = np.mean([ep500[c]['auroc'] for c in CATS])
print(f"{'平均':14s} {m_old:8.4f} {m200:8.4f} {m500:8.4f} {m500-m_old:+8.4f}")

# 与 baseline 对比
print("\n=== 与 baseline 对比（MVTec-AD 平均 AUROC）===")
print(f"RD-KG-NAS (500ep, efficientnet_b0 学生): {m500:.4f}")
print(f"Reverse Distillation (baseline):         0.8049")
print(f"PaDiM:                                   0.7792")
print(f"CFA:                                     0.8975")
print(f"vs RD 差距: {m500-0.8049:+.4f}")
print(f"vs CFA 差距: {m500-0.8975:+.4f}")
