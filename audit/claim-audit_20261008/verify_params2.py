"""Corrected ground-truth parameter measurement.

StudentEncoder lazily builds its 1x1 bottleneck on the first forward pass, so a
naive parameter count under-reports. This script runs a dummy forward pass
before counting.
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

import sys
sys.path.insert(0, _os.path.join(_KGD_ROOT, 'unsupervised_nas'))

import torch
from methods.rd_detector import RDDetector


def total(mod):
    return sum(p.numel() for p in mod.parameters()) / 1e6


CASES = [
    ('efficientnet_b0', 'efficientnet_b0', 128),
    ('efficientnet_b0', 'efficientnet_b0', 256),
    ('efficientnet_b0', 'efficientnet_b0', 512),
    ('resnet18', 'resnet18', 256),
    ('resnet34', 'resnet34', 256),
    ('resnet50', 'resnet50', 256),
    ('mobilenetv3_small_100', 'mobilenetv3_small_100', 128),
    ('swin_tiny_patch4_window7_224', 'swin_tiny_patch4_window7_224', 256),
    ('convnext_tiny', 'convnext_tiny', 256),
    ('convnextv2_tiny', 'convnextv2_tiny', 256),
]

print('%-32s %5s %10s %10s %10s %10s' %
      ('backbone', 'dim', 'encoder', 'decoder', 'TOTAL', 'paper fig'))
print('-' * 84)
res = {}
x = torch.randn(1, 3, 224, 224)
for label, bb, bd in CASES:
    try:
        m = RDDetector(student_backbone=bb, teacher_arch='resnet18',
                       bottleneck_dim=bd, pretrained=False)
        m.eval()
        with torch.no_grad():
            _ = m.encoder(x)          # materialise the lazy bottleneck
        e, d = total(m.encoder), total(m.decoder)
        res[(bb, bd)] = e + d
        print('%-32s %5d %10.2f %10.2f %10.2f' % (bb, bd, e, d, e + d))
    except Exception as ex:
        print('%-32s %5d  ERR %s' % (bb, bd, str(ex)[:40]))

print()
d = res.get(('efficientnet_b0', 256))
r = res.get(('resnet18', 256))
if d and r:
    print('  discovered/256d = %.2f M   (manuscript 5.3 ; figure csv 6.0)' % d)
    print('  resnet18/256d   = %.2f M   (manuscript 12.9 ; figure csv 14.9)' % r)
    print('  ratio           = %.3f      (manuscript 2.4x)' % (r / d))
m = res.get(('mobilenetv3_small_100', 256)) or res.get(('mobilenetv3_small_100', 128))
if m:
    print('  mobilenetv3     = %.2f M   (manuscript 2.7 ; figure csv 2.5)' % m)
s = res.get(('swin_tiny_patch4_window7_224', 256))
if s:
    print('  swin_tiny       = %.2f M   (manuscript 29.3 ; figure csv 28.0)' % s)
