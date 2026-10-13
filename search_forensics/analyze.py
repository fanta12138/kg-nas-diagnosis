# -*- coding: utf-8 -*-
"""Search-space forensics: what the recorded NAS runs actually show.

Reads every recorded search result under unsupervised_nas/experiments and
quantifies four things, with no re-training:

 1. degenerate trials       -- the proxy silently returns exactly 0.0 when every
                               validation batch is rejected; the search minimises,
                               so 0.0 is the best attainable score.
 2. within-configuration    -- the same (backbone, feature_dim) cell evaluated more
    variability                than once inside one run.
 3. inert-axis swing        -- the spread the DEPLOYMENT-INERT feature_dim axis
                               induces in the proxy score.
 4. scale across categories -- how far the proxy's absolute level moves between
                               categories, which makes cross-category scores
                               incomparable.

Writes search_forensics.json and prints a readable digest.
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

import collections, glob, io, json, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
OUT = _os.path.join(_KGD_ROOT, 'search_forensics')
os.makedirs(OUT, exist_ok=True)

import numpy as np


def load_run(path):
    d = json.load(open(path, encoding='utf-8'))
    return d.get('all_results', []), d.get('best_architecture'), d.get('num_trials')


def collect():
    runs, trials = [], []
    for f in sorted(glob.glob(os.path.join(NAS, 'experiments', '*', '**', 'results.json'),
                              recursive=True)) + \
            sorted(glob.glob(os.path.join(NAS, 'experiments', '*', 'search_results.json'))) + \
            sorted(glob.glob(os.path.join(NAS, 'experiments', '*', '*', 'search_results.json'))):
        rel = os.path.relpath(f, NAS).replace('\\', '/')
        res, best, declared = load_run(f)
        if not res:
            continue
        parts = rel.split('/')
        cat = parts[-3] if parts[-1] == 'results.json' else 'wood'
        strat = parts[-2] if parts[-1] == 'results.json' else 'random'
        runs.append({'file': rel, 'category': cat, 'strategy': strat,
                     'recorded': len(res), 'declared': declared, 'best': best})
        for x in res:
            a = x.get('architecture', {})
            trials.append({'run': rel, 'category': cat, 'strategy': strat,
                           'backbone': a.get('backbone'), 'dim': a.get('feature_dim'),
                           'score': x.get('score')})
    return runs, trials


def main():
    runs, trials = collect()
    sc = np.array([t['score'] for t in trials], dtype=float)
    out = {'n_runs': len(runs), 'n_trials': len(trials), 'runs': runs}

    # ---- 1. degenerate trials -------------------------------------------------
    zeros = int((sc == 0.0).sum())
    out['degenerate'] = {
        'exact_zero_trials': zeros, 'total_trials': len(trials),
        'share_pct': 100.0 * zeros / len(trials) if trials else None,
        'per_run': {r['file']: {'recorded': r['recorded'], 'declared': r['declared']}
                    for r in runs},
    }
    # Per-run: does the naive argmin (nas_engine.py:273 min(self.results, key=score))
    # select a degenerate zero-score trial?
    byrun = collections.defaultdict(list)
    for t in trials:
        byrun[t['run']].append(t)
    argue = []
    for f, ts in sorted(byrun.items()):
        pos = [t for t in ts if np.isfinite(t['score']) and t['score'] > 0]
        nz = sum(1 for t in ts if t['score'] == 0.0)
        arg = min(ts, key=lambda t: t['score'])
        best_pos = min(pos, key=lambda t: t['score']) if pos else None
        argue.append({
            'run': f, 'n_trials': len(ts), 'zeros': nz,
            'argmin_backbone': arg['backbone'], 'argmin_dim': arg['dim'],
            'argmin_score': arg['score'],
            'argmin_is_degenerate': bool(arg['score'] == 0.0),
            'best_nonzero_backbone': best_pos['backbone'] if best_pos else None,
            'best_nonzero_dim': best_pos['dim'] if best_pos else None,
            'best_nonzero_score': best_pos['score'] if best_pos else None,
        })
    out['argmin_degeneracy'] = {
        'runs': argue,
        'runs_with_degenerate_argmin': int(sum(a['argmin_is_degenerate'] for a in argue)),
        'runs_total': len(argue),
    }
    # ---- 4. scale across categories ------------------------------------------
    bycat = collections.defaultdict(list)
    for t in trials:
        if np.isfinite(t['score']) and t['score'] > 0:
            bycat[t['category']].append(t['score'])
    if bycat:
        med = {c: float(np.median(v)) for c, v in bycat.items()}
        pos = [v for v in med.values() if v > 0]
        out['category_scale'] = {
            'median_proxy_per_category': med,
            'max_over_min': float(max(pos) / min(pos)) if pos else None,
        }

    # ---- 2. within-configuration variability ---------------------------------
    cells = collections.defaultdict(list)
    for t in trials:
        cells[(t['run'], t['backbone'], t['dim'])].append(t['score'])
    rep = {k: np.array(v, dtype=float) for k, v in cells.items() if len(v) > 1}
    cv_rows = []
    for k, v in rep.items():
        m = v.mean()
        cv_rows.append({'run': k[0], 'backbone': k[1], 'dim': k[2], 'n': int(len(v)),
                        'min': float(v.min()), 'max': float(v.max()),
                        'cv': float(v.std(ddof=0) / abs(m)) if m else None,
                        'collapsed': bool((v == 0).any())})
    cvs = np.array([r['cv'] for r in cv_rows if r['cv'] is not None and np.isfinite(r['cv'])])
    out['within_config'] = {
        'cells_with_repeats': len(rep), 'cells_total': len(cells),
        'cv_median': float(np.median(cvs)) if len(cvs) else None,
        'cv_p90': float(np.percentile(cvs, 90)) if len(cvs) else None,
        'cv_max': float(cvs.max()) if len(cvs) else None,
        'cells_containing_a_zero': int(sum(r['collapsed'] for r in cv_rows)),
        'rows': sorted(cv_rows, key=lambda r: -(r['cv'] or 0)),
    }

    # ---- 3. inert-axis swing --------------------------------------------------
    bybb = collections.defaultdict(dict)
    for t in trials:
        bybb[(t['run'], t['backbone'])].setdefault(t['dim'], []).append(t['score'])
    swing = []
    for k, dd in bybb.items():
        if len(dd) < 2:
            continue
        m = {d: float(np.mean(v)) for d, v in dd.items()}
        pos = [x for x in m.values() if x > 0]
        if len(pos) < 2:
            continue
        swing.append({'run': k[0], 'backbone': k[1], 'dim_means': m,
                      'max_over_min': float(max(pos) / min(pos))})
    ratios = np.array([s['max_over_min'] for s in swing])
    out['inert_axis'] = {
        'n_backbone_comparisons': len(swing),
        'median_ratio': float(np.median(ratios)) if len(ratios) else None,
        'p90_ratio': float(np.percentile(ratios, 90)) if len(ratios) else None,
        'max_ratio': float(ratios.max()) if len(ratios) else None,
        'rows': sorted(swing, key=lambda s: -s['max_over_min']),
        'note': 'feature_dim cannot reach the deployed detector (StudentEncoder '
                'hard-codes a 256-d bottleneck), so any spread here is pure artefact.',
    }

    json.dump(out, open(os.path.join(OUT, 'search_forensics.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    # ---- digest ---------------------------------------------------------------
    print('runs=%d  trials=%d' % (len(runs), len(trials)))
    print('\n[1] degenerate trials (score == 0.0): %d / %d = %.1f%%'
          % (zeros, len(trials), 100.0 * zeros / len(trials)))
    ag = out['argmin_degeneracy']
    print('    runs whose naive argmin is a degenerate trial: %d / %d'
          % (ag['runs_with_degenerate_argmin'], ag['runs_total']))
    print('    %-46s %5s %5s %10s %-30s %s' % ('run', 'n', 'zer', 'argmin', 'argmin_cfg', 'degen'))
    for a in ag['runs']:
        print('    %-46s %5d %5d %10.5g %-30s %s'
              % (a['run'][-44:], a['n_trials'], a['zeros'], a['argmin_score'],
                 '%s@%s' % (a['argmin_backbone'], a['argmin_dim']),
                 'YES' if a['argmin_is_degenerate'] else ''))
    print('\n[2] within-configuration CV')
    print('    repeated cells: %d / %d' % (len(rep), len(cells)))
    if len(cvs):
        print('    median %.1f%%   p90 %.1f%%   max %.1f%%'
              % (100 * np.median(cvs), 100 * np.percentile(cvs, 90), 100 * cvs.max()))
        print('    cells containing an exact zero: %d' % out['within_config']['cells_containing_a_zero'])
    print('\n[3] inert-axis (feature_dim) swing of the proxy')
    if len(ratios):
        print('    %d backbones compared; median %.2fx  p90 %.1fx  max %.1fx'
              % (len(swing), np.median(ratios), np.percentile(ratios, 90), ratios.max()))
        for s in out['inert_axis']['rows'][:5]:
            print('      %-46s %-28s max/min=%.1fx' % (s['run'][-34:], s['backbone'],
                                                        s['max_over_min']))
    print('\n[4] category scale')
    if bycat:
        m = out['category_scale']['median_proxy_per_category']
        for c in sorted(m):
            print('      %-10s median proxy = %.4g' % (c, m[c]))
        print('    max/min across categories = %.1fx' % out['category_scale']['max_over_min'])
    print('\nsaved -> %s' % os.path.join(OUT, 'search_forensics.json'))


if __name__ == '__main__':
    main()
