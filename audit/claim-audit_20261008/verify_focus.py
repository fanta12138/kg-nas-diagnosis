"""Focused recomputation for F6/F8/F9/F10/F13 and the 107 FPS question."""
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

import csv, os, collections, json, glob

FIG = _os.path.join(_KGD_ROOT, 'fig')

bl = list(csv.DictReader(open(os.path.join(FIG, 'baselines_full_results.csv'), encoding='utf-8-sig')))
print('baselines_full_results.csv: %d rows' % len(bl))
agg = collections.defaultdict(list)
for r in bl:
    agg[r['Method']].append(float(r['AUROC']))
print('\n--- method averages (half-up to 1dp) ---')
for m, v in sorted(agg.items(), key=lambda x: -sum(x[1]) / len(x[1])):
    mean = sum(v) / len(v)
    print('  %-22s n=%2d  mean=%.4f   ->1dp %.1f  ->2dp %.2f' % (m, len(v), mean, round(mean + 1e-12, 1), round(mean + 1e-12, 2)))

print('\n--- F9: CFA vs runner-up, category by category ---')
per = collections.defaultdict(dict)
for r in bl:
    per[r['Category']][r['Method']] = float(r['AUROC'])
ties = 0
for cat, d in sorted(per.items()):
    top = max(d.values())
    win = [m for m, v in d.items() if abs(v - top) < 1e-12]
    second = sorted(d.values())[-2]
    if len(win) > 1 or (top - second) < 0.001:
        ties += 1
        print('  %-11s top=%s (%.4f)  2nd=%.4f  margin=%.4f pp  ties=%s' %
              (cat, ','.join(win), top, second, (top - second) * 100, len(win) > 1))
print('  categories where the top is a tie or margin < 0.1pp: %d' % ties)
print('  CFA best-count:', sum(1 for d in per.values() if max(d.values()) == d.get('CFA')))
print('  CFA strict wins (no tie):',
      sum(1 for d in per.values() if d.get('CFA') == max(d.values()) and list(d.values()).count(max(d.values())) == 1))
print('  CFA tied wins:',
      sum(1 for d in per.values() if d.get('CFA') == max(d.values()) and list(d.values()).count(max(d.values())) > 1))

print('\n--- F10: backbone_distribution.csv ---')
bd = list(csv.DictReader(open(os.path.join(FIG, 'backbone_distribution.csv'), encoding='utf-8-sig')))
print('  columns:', list(bd[0].keys()))
print('  rows:', len(bd))
for r in bd:
    print('    ', r)

print('\n--- search-space table (from the manuscript) ---')
import re
src = open(_os.path.join(_KGD_PAPER, 'sections', '3_method.tex'), encoding='utf-8').read()
tb = re.search(r'\\begin\{tabular\}\{llccc\}(.*?)\\end\{tabular\}', src, re.S)
if tb:
    rows = [r for r in tb.group(1).split(r'\\') if '&' in r]
    cnn = sum(1 for r in rows if 'CNN' in r)
    tr = sum(1 for r in rows if 'Transformer' in r)
    print('  rows=%d  CNN rows=%d  Transformer rows=%d' % (len(rows), cnn, tr))

print('\n--- M1: any "107" in evidence? ---')
hits = []
for root in [FIG, _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'paper_results')]:
    for ext in ('*.csv', '*.json', '*.txt'):
        for p in glob.glob(os.path.join(root, ext)):
            try:
                t = open(p, encoding='utf-8', errors='ignore').read()
            except OSError:
                continue
            for m in re.finditer(r'\b107\b', t):
                s = max(0, m.start() - 60)
                hits.append((os.path.basename(p), t[s:m.end() + 30].replace('\n', ' ')))
print('  hits: %d' % len(hits))
for h in hits[:12]:
    print('    ', h)

print('\n--- F13: BTAD training sample counts (manuscript + any artefact) ---')
for p in glob.glob(_os.path.join(_KGD_ROOT, 'experiments', 'btad_stage2', '*', '*', 'best_architecture.yaml')):
    print('  ', p)
    print('     ', open(p, encoding='utf-8', errors='ignore').read().replace('\n', ' | ')[:200])
