"""Ground-truth parameter measurement for the RD student (encoder + decoder).

Instantiates the authors' own RDDetector from
<REPO_ROOT>/unsupervised_nas/methods/rd_detector.py and counts parameters,
instead of trusting any hard-coded number in the manuscript or the scripts.
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

import sys, os
sys.path.insert(0, _os.path.join(_KGD_ROOT, 'unsupervised_nas'))

import torch
from methods.rd_detector import RDDetector


def n(mod):
    return sum(p.numel() for p in mod.parameters()) / 1e6


CASES = [
    ('discovered (efficientnet_b0, 256d)', 'efficientnet_b0', 256),
    ('resnet18 (fixed)', 'resnet18', 256),
    ('mobilenetv3_small (fixed)', 'mobilenetv3_small_100', 256),
    ('swin_tiny (fixed)', 'swin_tiny_patch4_window7_224', 256),
]

print('%-38s %10s %10s %10s' % ('case', 'encoder', 'decoder', 'TOTAL'))
print('-' * 72)
rows = {}
for label, bb, bd in CASES:
    try:
        m = RDDetector(student_backbone=bb, teacher_arch='resnet18',
                       bottleneck_dim=bd, pretrained=False)
        e, d = n(m.encoder), n(m.decoder)
        rows[label] = (e, d, e + d)
        print('%-38s %10.2f %10.2f %10.2f' % (label, e, d, e + d))
    except Exception as ex:
        print('%-38s  ERR %s' % (label, str(ex)[:60]))

print()
print('manuscript says   : 5.3 / 12.9 / 2.7 / 29.3   (labelled "student encoder+decoder")')
print('figure data says  : 6.0 / 14.9 / 2.5 / 28.0')
print()
for label, (e, d, t) in rows.items():
    print('  %-38s measured total = %.2f M' % (label, t))

# ratio check
if 'discovered (efficientnet_b0, 256d)' in rows and 'resnet18 (fixed)' in rows:
    disc = rows['discovered (efficientnet_b0, 256d)'][2]
    r18 = rows['resnet18 (fixed)'][2]
    print()
    print('  ratio ResNet-18 / discovered = %.3f  (manuscript: 2.4x; 14.9/6.0 = %.2fx)' %
          (r18 / disc, 14.9 / 6.0))
    print('  discovered as %% of 12-15M baseline band: %.0f%% - %.0f%%' %
          (100 * disc / 15, 100 * disc / 12))
