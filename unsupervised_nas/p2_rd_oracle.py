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


def load_all():
    data = {}
    # NAS 200/500ep
    for ep in [200, 500]:
        for f in glob.glob(str(root / f'nas/ep{ep}/*/final_evaluation.json')):
            d = json.load(open(f, encoding='utf-8'))
            data[f'nas_{ep}ep'] = data.get(f'nas_{ep}ep', {})
            data[f'nas_{ep}ep'][Path(f).parent.name] = d['auroc']
    # fixed
    for name in ['resnet18', 'swin_tiny', 'mbv3_small']:
        for f in glob.glob(str(root / f'fixed/{name}/ep200/*/final_evaluation.json')):
            d = json.load(open(f, encoding='utf-8'))
            data[f'fixed_{name}'] = data.get(f'fixed_{name}', {})
            data[f'fixed_{name}'][Path(f).parent.name] = d['auroc']
    return data


data = load_all()
names = list(data.keys())

print("=" * 78)
print("RD 范式：逐类别学生架构对比（AUROC）")
print("=" * 78)
header = f"{'类别':14s}" + "".join(f"{n.split('_',1)[1]:>12s}" for n in names)
print(header)
oracle = {}
for c in CATS:
    row = {n: data[n].get(c) for n in names}
    best_n = max((n for n in names if row[n] is not None), key=lambda n: row[n])
    oracle[c] = (best_n, row[best_n])
    cells = "".join(f"{row[n]:12.4f}" if row[n] is not None else f"{'N/A':>12s}" for n in names)
    print(f"{c:14s}{cells}  <- {best_n.split('_',1)[1]}")

print()
for n in names:
    vals = [data[n][c] for c in CATS if c in data[n]]
    print(f"{n:16s}: 平均 {np.mean(vals):.4f}")

# oracle 组合
oracle_avg = np.mean([v for _, v in oracle.values()])
print(f"\nPer-category oracle 组合平均: {oracle_avg:.4f}")
print("各架构夺冠类别数:")
wins = {}
for c, (n, _) in oracle.items():
    wins[n] = wins.get(n, 0) + 1
for n, w in sorted(wins.items(), key=lambda x: -x[1]):
    print(f"  {n:16s}: {w} 类")

# screw 修复验证
print("\nscrew 类别各架构:")
for n in names:
    print(f"  {n:16s}: {data[n].get('screw')}")
