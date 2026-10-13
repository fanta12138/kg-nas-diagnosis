"""Executor verification — Phase 2.

Independently recompute the numbers the zero-context reviewer flagged.
Evidence root: <REPO_ROOT>/unsupervised_nas/experiments/p2_rd
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

import json, os, glob, collections

ROOT = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', 'p2_rd')
CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']


def read_auroc(path):
    """final_evaluation.json may nest the metric; find the AUROC value."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)

    def hunt(o, depth=0):
        if depth > 5:
            return None
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if isinstance(v, (int, float)) and ('auroc' in kl or kl in ('image_auroc', 'img_auroc')):
                    return float(v)
            for v in o.values():
                r = hunt(v, depth + 1)
                if r is not None:
                    return r
        return None
    return hunt(d)


def load_arm(arm_glob, label):
    out = {}
    for cat in CATS:
        hits = glob.glob(os.path.join(ROOT, arm_glob, cat, 'final_evaluation.json'))
        if hits:
            out[cat] = read_auroc(hits[0])
    print('  %-14s categories found: %d' % (label, len(out)))
    return out


print('=' * 68)
print('VERIFY F1 / F2 — 500-epoch winner counts and mean AUROC')
print('=' * 68)
arms = {
    'nas':        load_arm('nas/ep500', 'nas/ep500'),
    'resnet18':   load_arm('teacher_resnet18/fixed/resnet18/ep500', 'resnet18'),
    'mbv3_small': load_arm('teacher_resnet18/fixed/mbv3_small/ep500', 'mbv3_small'),
    'swin_tiny':  load_arm('teacher_resnet18/fixed/swin_tiny/ep500', 'swin_tiny'),
}

print()
for name, d in arms.items():
    if d:
        vals = list(d.values())
        print('  %-11s n=%2d  mean AUROC = %.4f (%.1f%%)' %
              (name, len(vals), sum(vals) / len(vals), 100 * sum(vals) / len(vals)))

# winner per category
common = set.intersection(*[set(d) for d in arms.values() if d])
print('\n  common categories: %d' % len(common))
win = collections.Counter()
detail = []
for cat in CATS:
    if cat not in common:
        continue
    scores = {k: v[cat] for k, v in arms.items() if cat in v}
    top = max(scores.values())
    winners = [k for k, v in scores.items() if abs(v - top) < 1e-12]
    for w in winners:
        win[w] += 1
    detail.append((cat, scores, winners))
print('\n  500-epoch winner counts (ties counted for each):')
for k, v in win.most_common():
    print('     %-11s %d' % (k, v))
print('\n  per-category detail (AUROC, higher better):')
for cat, sc, w in detail:
    print('     %-11s  ' % cat + '  '.join('%s=%.4f' % (k, v) for k, v in sorted(sc.items(), key=lambda x: -x[1])) + '   -> ' + ','.join(w))
