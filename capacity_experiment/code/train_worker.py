# -*- coding: utf-8 -*-
"""Capacity-experiment training worker.

Trains ONE (tier, category) pair with the backbone the tier's search selected,
using the authors' own train_category() so the protocol is identical to the
paper's main results (500 epochs, bs 64, seed 42, ResNet-18 teacher,
reverse distillation).

IMPORTANT — the feature dimension is fixed at 256, matching what the paper
actually trains:
  * the search's `feature_dim` is passed to ContrastiveModel.projection_dim
    (methods/nas_engine.py), a search-time construct — it does NOT size the
    final detector;
  * StudentEncoder.forward lazily builds its bottleneck with a hard-coded 256
    (methods/rd_detector.py:108), so a non-256 bottleneck_dim would make the
    encoder and decoder disagree in channel count;
  * p2_rd_train.py:102 therefore constructs RDDetector with default args.
So a budget changes only WHICH BACKBONE is trained, never the bottleneck.

Usage: python train_worker.py <tier_M> <backbone> <category>
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

import io, json, os, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

NAS_DIR = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
sys.path.insert(0, NAS_DIR)
os.chdir(NAS_DIR)

import importlib.util

# scripts/ is not a package, so load the author's module by path (file untouched).
_spec = importlib.util.spec_from_file_location(
    'p2_rd_train', os.path.join(NAS_DIR, 'scripts', 'p2_rd_train.py'))
_p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p2)
train_category = _p2.train_category

import torch

EXP = _os.path.join(_KGD_ROOT, 'capacity_experiment', 'results')


def main():
    tier, backbone, category = sys.argv[1], sys.argv[2], sys.argv[3]
    device = torch.device('cuda')
    out = os.path.join(EXP, 'tier_%sM' % tier, category)
    t0 = time.time()
    auroc, ap, minutes = train_category(
        category=category, student_backbone=backbone,
        label='tier%s' % tier, epochs=500, seed=42,
        output_dir=out, device=device, teacher='resnet18')
    print('DONE tier=%s cat=%s %s AUROC=%.4f AP=%.4f (%.1f min wall)'
          % (tier, category, backbone, auroc, ap, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
