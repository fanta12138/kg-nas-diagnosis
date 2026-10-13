# -*- coding: utf-8 -*-
"""Probe: does NASSearchEngine.evaluate_architecture have a degenerate mode?

nas_engine.evaluate_architecture accumulates the validation score only when
`isfinite(score) and score < 1000`; otherwise it prints a warning and skips the
batch.  The final average is `total_score / max(count, 1)`.  Therefore, if EVERY
validation batch is rejected, count == 0 and the function silently returns 0.0 --
the minimum of the range it minimises over.  This probe counts the rejected
batches so we can tell a genuine 0.0 from a degenerate one.
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

import contextlib, io, json, os, random, sys, time, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
sys.path.insert(0, NAS)
os.chdir(NAS)

import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import MVTecWoodDataset
from models import ContrastiveModel
from methods.nas_engine import NASSearchEngine

OUT = _os.path.join(_KGD_ROOT, 'proxy_diag', 'probe_degenerate.json')
dev = torch.device('cuda')

ck = torch.load('experiments/stage1/best_teacher.pth', map_location=dev)
tm = ContrastiveModel(backbone_name=ck['config']['model']['backbone'],
                      projection_dim=ck['config']['model']['projection_dim'],
                      pretrained=False).to(dev)
tm.load_state_dict(ck['model_state_dict']); tm.eval()
for p in tm.parameters():
    p.requires_grad = False

tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
tr = MVTecWoodDataset(root='../datasets/mvtec_anomaly_detection/wood', split='train',
                      defect_type='good', transform=tf, return_pair=False)
va = MVTecWoodDataset(root='../datasets/mvtec_anomaly_detection/wood', split='test',
                      defect_type='good', transform=tf, return_pair=False)
tl = DataLoader(tr, batch_size=16, shuffle=True, num_workers=0)
vl = DataLoader(va, batch_size=16, shuffle=False, num_workers=0)
eng = NASSearchEngine(search_space={'backbones': ['efficientnet_b0'], 'feature_dims': [128, 256, 512]},
                      teacher_model=tm, device=dev)

rows = []
for dim in (128, 256, 512):
    for r in range(4):
        seed = 7000 + 17 * r
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = eng.evaluate_architecture({'backbone': 'efficientnet_b0',
                                           'feature_dim': int(dim)}, tl, vl,
                                          quick_epochs=3)
        txt = buf.getvalue()
        rec = {'dim': int(dim), 'seed': seed, 'score': float(s),
               'rejected_batches': txt.count('Invalid score'),
               'n_val_batches': len(vl),
               'align_layer': txt.count('添加维度对齐层') > 0}
        rec['degenerate'] = rec['rejected_batches'] >= rec['n_val_batches']
        rows.append(rec)
        print('dim=%3d seed=%5d score=%.8f rejected=%d/%d degenerate=%s'
              % (rec['dim'], seed, rec['score'], rec['rejected_batches'],
                 rec['n_val_batches'], rec['degenerate']), flush=True)

json.dump({'val_batches': len(vl), 'rows': rows}, open(OUT, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
nd = sum(r['degenerate'] for r in rows)
print('\n%d / %d evaluations hit the degenerate mode (returned 0.0 because every '
      'validation batch was rejected).' % (nd, len(rows)))
print('saved -> %s' % OUT)
