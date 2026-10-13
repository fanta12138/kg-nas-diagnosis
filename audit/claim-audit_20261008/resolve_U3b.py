"""List every within-config score ratio to locate the claim 'vary by up to 81x'."""
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

import json, glob, collections

def trials(p):
    d = json.load(open(p, encoding='utf-8'))
    for r in (d.get('all_results') or []):
        a = r.get('architecture') if isinstance(r.get('architecture'), dict) else {}
        bb, dm, s = a.get('backbone'), a.get('feature_dim'), r.get('score')
        if bb and s is not None:
            yield bb, dm, float(s)

allr = []
for f in glob.glob(_os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', 'p1_search_strategies', '*', '*', 'results.json')):
    parts = f.replace('\\', '/').split('/')
    cat, st = parts[-3], parts[-2]
    g = collections.defaultdict(list)
    for bb, dm, s in trials(f):
        g[(bb, dm)].append(s)
    for k, v in g.items():
        if len(v) > 1 and min(v) > 1e-12:
            allr.append((max(v) / min(v), '%s/%s' % (cat, st), '%s/%s' % k, min(v), max(v), len(v)))

f = _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', 'stage2_topjournal_15backbones_fixed', 'search_results.json')
g = collections.defaultdict(list)
for bb, dm, s in trials(f):
    g[(bb, dm)].append(s)
for k, v in g.items():
    if len(v) > 1 and min(v) > 1e-12:
        allr.append((max(v) / min(v), 'mvtec93', '%s/%s' % k, min(v), max(v), len(v)))

allr.sort(reverse=True)
print('%-10s  %-24s  %-34s  %-12s  %-12s  %3s' %
      ('ratio', 'category/strategy', 'config', 'min', 'max', 'n'))
for r, c, k, lo, hi, n in allr[:24]:
    mark = '   <<< near 81' if 78 <= r <= 84 else ''
    print('%-10.1f  %-24s  %-34s  %-12.6g  %-12.6g  %3d%s' % (r, c, k, lo, hi, n, mark))
print()
print('groups with repeats (finite): %d   max ratio: %.1fx   median: %.1fx' %
      (len(allr), allr[0][0], sorted(x[0] for x in allr)[len(allr) // 2]))
print('any group in [78,84]x:', [x for x in allr if 78 <= x[0] <= 84][:3] or 'NONE')
