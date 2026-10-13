# -*- coding: utf-8 -*-
"""FILL THE EVIDENCE GAP: measure the full teacher-student detector throughput.

The manuscript formerly claimed "the full teacher--student detector runs at
~107 FPS" with no artefact behind it (audit finding M1). This script produces
that measurement, and also re-measures the student-only throughput so the
existing 142 FPS figure can be checked against the same protocol.

Protocol (matches the manuscript footnote and eps/fps_remeasure2.py):
  batch size 1, 224x224 input, fp32, idle RTX 3090, warmup 100, iters 500,
  rounds 3, median reported.

Evidence written to:
  <REPO_ROOT>/audit/claim-audit_20261008/fps_benchmark.json
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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, _os.path.join(_KGD_ROOT, 'unsupervised_nas'))

import torch
from methods.rd_detector import RDDetector

OUT = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')
DEV = torch.device('cuda')

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True   # steady-state inference
x = torch.randn(1, 3, 224, 224, device=DEV)


def bench(fn, warmup=100, iters=500, rounds=3):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            ts.append(iters / (time.perf_counter() - t0))
    ts.sort()
    return dict(median=round(ts[len(ts) // 2], 2),
                min=round(ts[0], 2), max=round(ts[-1], 2))


CASES = [
    ('discovered (efficientnet_b0 + 256d)', 'efficientnet_b0', 256),
    ('fixed resnet18 + 256d', 'resnet18', 256),
    ('fixed mobilenetv3_small + 256d', 'mobilenetv3_small_100', 256),
    ('fixed swin_tiny + 256d', 'swin_tiny_patch4_window7_224', 256),
]

results = {'protocol': {'batch_size': 1, 'input': '3x224x224', 'dtype': 'fp32',
                        'device': torch.cuda.get_device_name(0),
                        'warmup': 100, 'iters': 500, 'rounds': 3,
                        'metric': 'median FPS', 'cudnn_benchmark': True},
           'models': {}}

print('device: %s' % torch.cuda.get_device_name(0))
print('%-38s %10s %10s %10s' % ('model', 'student', 'teacher', 'FULL'))
print('-' * 74)

for label, bb, bd in CASES:
    try:
        m = RDDetector(student_backbone=bb, teacher_arch='resnet18',
                       bottleneck_dim=bd, pretrained=False).to(DEV).eval()
        with torch.inference_mode():
            _ = m.encoder(x)                       # materialise lazy bottleneck
            _ = m.anomaly_score(x)                 # warm the scoring path

        st = bench(lambda: m.forward_student(x))
        tc = bench(lambda: m.teacher(x))
        full = bench(lambda: m.anomaly_score(x))
        results['models'][label] = {
            'backbone': bb, 'feature_dim': bd,
            'student_fps': st, 'teacher_fps': tc, 'full_pipeline_fps': full,
            'student_params_m': round(sum(p.numel() for p in m.encoder.parameters())
                                      + sum(p.numel() for p in m.decoder.parameters())) / 1e6,
            'decoder_params_m': round(sum(p.numel() for p in m.decoder.parameters())) / 1e6,
            'teacher_params_m': round(sum(p.numel() for p in m.teacher.parameters())) / 1e6,
        }
        print('%-38s %10.1f %10.1f %10.1f' % (label, st['median'], tc['median'], full['median']))
    except Exception as ex:
        print('%-38s  ERR %s' % (label, str(ex)[:50]))
        results['models'][label] = {'error': str(ex)}

d = results['models'].get('discovered (efficientnet_b0 + 256d)', {})
if 'full_pipeline_fps' in d:
    print()
    print('>>> discovered student-only : %.1f FPS  (manuscript claimed 142)' % d['student_fps']['median'])
    print('>>> FULL teacher+student    : %.1f FPS  (manuscript claimed ~107, no artefact)' % d['full_pipeline_fps']['median'])
    print('>>> teacher alone           : %.1f FPS' % d['teacher_fps']['median'])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('\nwritten:', OUT)
