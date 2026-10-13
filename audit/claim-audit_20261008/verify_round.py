"""Precise rounding + best-count recomputation."""
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

import csv, os, collections, re
from decimal import Decimal, ROUND_HALF_UP

FIG = _os.path.join(_KGD_ROOT, 'fig')
bl = list(csv.DictReader(open(os.path.join(FIG, 'baselines_full_results.csv'), encoding='utf-8-sig')))


def hu(x, nd):
    return Decimal(repr(x)).quantize(Decimal('1.' + '0' * nd), rounding=ROUND_HALF_UP)


agg = collections.defaultdict(list)
per = collections.defaultdict(dict)
for r in bl:
    v = float(r['AUROC']) * 100
    agg[r['Method']].append(v)
    per[r['Category']][r['Method']] = v

print('=== exact means and correct 1-dp / 2-dp rounding (half-up) ===')
paper = {'cfa': 89.8, 'reverse_distillation': 80.5, 'padim': 78.0, 'spade': 78.7,
         'fastflow': 71.3, 'efficientad': 60.1, 'autoencoder': 58.0, 'stpm': 51.1,
         'patchcore': 50.6, 'simplenet': 50.5, 'vae': 47.6}
for m, v in sorted(agg.items(), key=lambda x: -sum(x[1]) / len(x[1])):
    mean = sum(v) / len(v)
    p = paper.get(m)
    flag = ''
    if p is not None:
        flag = 'OK' if abs(float(hu(mean, 1)) - p) < 1e-9 else '<<< paper says %.1f' % p
    print('  %-22s mean=%.6f   1dp=%s  2dp=%s  %s' % (m, mean, hu(mean, 1), hu(mean, 2), flag))

print()
print('=== CFA best-category accounting ===')
cats = sorted(per)
cfabest_strict = 0
cfabest_tied = 0
other = collections.Counter()
detail = []
for c in cats:
    d = per[c]
    top = max(d.values())
    winners = sorted([m for m, v in d.items() if abs(v - top) < 1e-9])
    for w in winners:
        other[w] += 1
    if 'cfa' in winners:
        if len(winners) == 1:
            cfabest_strict += 1
        else:
            cfabest_tied += 1
    detail.append((c, winners, top, sorted(d.values())[-2]))
print('  CFA strict wins : %d' % cfabest_strict)
print('  CFA tied wins   : %d' % cfabest_tied)
print('  CFA total best  : %d' % (cfabest_strict + cfabest_tied))
print('  all best counts :', dict(other))
print()
print('  ties / near-ties (margin < 0.1 pp):')
for c, w, top, second in detail:
    if len(w) > 1 or (top - second) < 0.1:
        print('     %-11s winners=%-28s margin vs runner-up = %.4f pp' %
              (c, ','.join(w), top - second))

print()
print('=== manuscript main-table Average row vs recomputation ===')
src = open(_os.path.join(_KGD_PAPER, 'sections', '4_experiments.tex'), encoding='utf-8').read()
m = re.search(r'\\textbf\{Average\}(.*?)\\\\', src, re.S)
if m:
    print('  paper row:', ' | '.join(x.strip() for x in m.group(1).split('&')))
print('  recomputed:', {k: str(hu(sum(v) / len(v), 1)) for k, v in
                       [('AE', agg['autoencoder']), ('VAE', agg['vae']), ('PaDiM', agg['padim']),
                        ('PatchCore', agg['patchcore']), ('STPM', agg['stpm']),
                        ('RevDist', agg['reverse_distillation']), ('CFA', agg['cfa']),
                        ('SPADE', agg['spade']), ('FastFlow', agg['fastflow']),
                        ('EffAD', agg['efficientad']), ('SimpleNet', agg['simplenet'])]})
