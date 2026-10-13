"""Recompute the 'vary by up to Nx within a category' claim (open item U3)."""
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

import json, glob, collections, os

ROOT = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', 'p1_search_strategies')
NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', 'stage2_topjournal_15backbones_fixed', 'search_results.json')


def trials_from(path):
    d = json.load(open(path, encoding='utf-8'))
    ar = d.get('all_results') or d.get('results') or []
    out = []
    for r in ar:
        a = r.get('architecture') if isinstance(r.get('architecture'), dict) else None
        bb = (a or {}).get('backbone') or r.get('backbone')
        dim = (a or {}).get('feature_dim') or r.get('feature_dim')
        s = r.get('score')
        if bb is not None and s is not None:
            out.append((bb, dim, float(s)))
    return out


print('=' * 72)
print('U3 — within (category, backbone, dim) score spread')
print('=' * 72)

worst = []
for f in sorted(glob.glob(os.path.join(ROOT, '*', '*', 'results.json'))):
    parts = f.replace('\\', '/').split('/')
    cat, strat = parts[-3], parts[-2]
    ts = trials_from(f)
    if not ts:
        continue
    grp = collections.defaultdict(list)
    for bb, dim, s in ts:
        grp[(bb, dim)].append(s)
    rep = {k: v for k, v in grp.items() if len(v) > 1}
    print('\n%-10s %-12s trials=%2d  distinct configs=%2d  repeated=%2d' %
          (cat, strat, len(ts), len(grp), len(rep)))
    for k, v in sorted(rep.items(), key=lambda x: -(max(x[1]) / max(min(x[1]), 1e-12))):
        lo, hi = min(v), max(v)
        ratio = hi / lo if lo > 1e-12 else float('inf')
        print('    %-38s n=%d  min=%.6g max=%.6g  ratio=%.1fx' %
              ('%s/%s' % k, len(v), lo, hi, ratio))
        if lo > 1e-12:
            worst.append((ratio, cat, strat, k, lo, hi, len(v)))

print('\n' + '=' * 72)
print('U3 — main 93-trial MVTec search (no repeat: trials are distinct draws)')
print('=' * 72)
mt = trials_from(NAS)
grp = collections.defaultdict(list)
for bb, dim, s in mt:
    grp[(bb, dim)].append(s)
rep = {k: v for k, v in grp.items() if len(v) > 1}
print('  trials=%d  distinct (backbone,dim)=%d  repeated=%d' % (len(mt), len(grp), len(rep)))
for k, v in sorted(rep.items(), key=lambda x: -(max(x[1]) / max(min(x[1]), 1e-12)))[:5]:
    lo, hi = min(v), max(v)
    print('    %-40s n=%d  ratio=%.1fx' % ('%s/%s' % k, len(v), hi / lo if lo > 1e-12 else float('inf')))

print('\n' + '=' * 72)
print('SUMMARY')
print('=' * 72)
if worst:
    worst.sort(reverse=True)
    r, cat, strat, k, lo, hi, n = worst[0]
    print('  max within-config ratio across search-strategy repeats: %.1fx  (%s/%s %s/%s, n=%d)' % (r, cat, strat, k[0], k[1], n))
    ratios = [w[0] for w in worst]
    print('  min: %.1fx   median: %.1fx   count: %d' % (min(ratios), sorted(ratios)[len(ratios)//2], len(ratios)))
