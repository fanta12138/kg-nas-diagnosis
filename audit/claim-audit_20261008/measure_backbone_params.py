# -*- coding: utf-8 -*-
"""Measure encoder+decoder parameters for every backbone in the search space.

Needed to define the parameter budgets of the capacity experiment.
Writes params_by_backbone.json next to the audit trace.
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

import io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, _os.path.join(_KGD_ROOT, 'unsupervised_nas'))

import torch
from methods.rd_detector import RDDetector

OUT = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')
DEV = torch.device('cuda')
x = torch.randn(1, 3, 224, 224, device=DEV)

BACKBONES = [
    'convnextv2_tiny', 'convnextv2_base', 'mobileone_s0',
    'convnext_tiny', 'convnext_small',
    'vit_tiny_patch16_224', 'deit_tiny_patch16_224', 'swin_tiny_patch4_window7_224',
    'efficientnet_b0', 'efficientnet_b1', 'mobilenetv3_small_100',
    'resnet18', 'resnet34', 'resnet50',
]
DIMS = [128, 256, 512]

res = {}
print('%-34s %7s %10s %10s' % ('backbone', 'dim', 'encoder', 'TOTAL'))
print('-' * 66)
for bb in BACKBONES:
    res[bb] = {}
    for d in DIMS:
        try:
            m = RDDetector(student_backbone=bb, teacher_arch='resnet18',
                           bottleneck_dim=d, pretrained=False).to(DEV).eval()
            with torch.inference_mode():
                _ = m.encoder(x)
            e = sum(p.numel() for p in m.encoder.parameters()) / 1e6
            dc = sum(p.numel() for p in m.decoder.parameters()) / 1e6
            res[bb][d] = {'encoder_m': round(e, 3), 'decoder_m': round(dc, 3),
                          'total_m': round(e + dc, 3)}
            print('%-34s %7d %10.3f %10.3f' % (bb, d, e, e + dc))
            del m
            torch.cuda.empty_cache()
        except Exception as ex:
            res[bb][d] = {'error': str(ex)[:60]}
            print('%-34s %7d  ERR %s' % (bb, d, str(ex)[:40]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nwritten:', OUT)

print('\n--- backbone minimum across dims (cheapest config) ---')
for bb, d in sorted(res.items(), key=lambda kv: min(v['total_m'] for v in kv[1].values() if 'total_m' in v)):
    vals = [v for v in d.values() if 'total_m' in v]
    if vals:
        lo = min(v['total_m'] for v in vals)
        hi = max(v['total_m'] for v in vals)
        print('  %-34s %.2f - %.2f M' % (bb, lo, hi))
