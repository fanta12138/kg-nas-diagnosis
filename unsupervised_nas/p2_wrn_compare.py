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


def load(path):
    d = {}
    for f in glob.glob(str(root / path)):
        r = json.load(open(f, encoding='utf-8'))
        d[Path(f).parent.name] = r
    return d


r18 = load('nas/ep200/*/final_evaluation.json')
wrn = load('teacher_wide_resnet50_2/nas/ep200/*/final_evaluation.json')

print(f"{'类别':14s} {'R18教师':>10s} {'WRN教师':>10s} {'差值':>8s}")
for c in CATS:
    a = r18[c]['auroc']
    b = wrn[c]['auroc']
    print(f"{c:14s} {a:10.4f} {b:10.4f} {b-a:+8.4f}")
print(f"{'平均':14s} {np.mean([r18[c]['auroc'] for c in CATS]):10.4f} "
      f"{np.mean([wrn[c]['auroc'] for c in CATS]):10.4f} "
      f"{np.mean([wrn[c]['auroc']-r18[c]['auroc'] for c in CATS]):+8.4f}")
