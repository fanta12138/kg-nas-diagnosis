# -*- coding: utf-8 -*-
"""
Aggregate the label-free scoring-policy selection over training seeds.

Each seed trains its own students (different initialisation) and then runs the
same label-free selection procedure on them. This script pools the per-category
results and reports the gain over the published default policy together with its
spread, which is the number quoted in Section 5.5.

Inputs (whichever exist):
  pseudo_select/pseudo_summary.json        <- seed 42 (un-suffixed, historical)
  pseudo_select/pseudo_summary_s43.json
  pseudo_select/pseudo_summary_s44.json
Output:
  pseudo_select/pseudo_multiseed.json
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'pseudo_multiseed.json')

CANDS = [
    ('42', 'pseudo_summary.json'),
    ('43', 'pseudo_summary_s43.json'),
    ('44', 'pseudo_summary_s44.json'),
]

ROWS = [
    ('default', 'published default policy (mean, uniform)'),
    ('pseudo_selected', 'pseudo-anomaly separation, best policy'),
    ('oracle', 'per-category oracle (upper bound)'),
]


def row_value(summary, key):
    v = summary.get(key)
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        return v.get('mean') or v.get('auroc')
    return None


def main():
    seeds = {}
    for tag, fn in CANDS:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            seeds[tag] = json.load(open(p, encoding='utf-8'))
    if not seeds:
        raise SystemExit('no summary files found')

    res = {'seeds': sorted(seeds), 'per_seed': {}, 'aggregate': {}}

    for tag, s in sorted(seeds.items()):
        n_cats = len(s.get('per_category') or {})
        row = {'n_categories': n_cats}
        for key, _ in ROWS:
            row[key] = row_value(s, key)
        ens = s.get('ensembles') or {}
        for name in ('top5', 'top10', 'top20'):
            if name in ens:
                row['ens_%s' % name] = ens[name].get('mean') if isinstance(ens[name], dict) \
                    else ens[name]
        if s.get('oracle_ensemble') is not None:
            oe = s['oracle_ensemble']
            row['oracle_ensemble'] = oe.get('mean') if isinstance(oe, dict) else oe
        row['oracle_hit'] = s.get('oracle_hit')
        row['spearman'] = s.get('spearman_pseudo_vs_auroc_mean')
        res['per_seed'][tag] = row

    for key, _ in ROWS:
        vals = [res['per_seed'][t][key] for t in res['seeds']
                if res['per_seed'][t].get(key) is not None]
        if vals:
            res['aggregate'][key] = {
                'n_seeds': len(vals),
                'mean': sum(vals) / len(vals),
                'min': min(vals), 'max': max(vals),
                'spread': max(vals) - min(vals),
            }
    for key in ('ens_top5', 'ens_top10', 'ens_top20', 'oracle_ensemble', 'oracle_hit',
                'spearman'):
        vals = [res['per_seed'][t][key] for t in res['seeds']
                if res['per_seed'][t].get(key) is not None]
        if vals:
            res['aggregate'][key] = {
                'n_seeds': len(vals),
                'mean': sum(vals) / len(vals),
                'min': min(vals), 'max': max(vals),
            }

    gains = []
    for t in res['seeds']:
        r = res['per_seed'][t]
        if r.get('pseudo_selected') is not None and r.get('default') is not None:
            gains.append(100.0 * (r['pseudo_selected'] - r['default']))
    res['aggregate']['gain_pp'] = {
        'per_seed': gains,
        'mean': sum(gains) / len(gains) if gains else None,
        'min': min(gains) if gains else None,
        'max': max(gains) if gains else None,
    }
    oracle_gaps = []
    for t in res['seeds']:
        r = res['per_seed'][t]
        if None not in (r.get('oracle'), r.get('default')):
            oracle_gaps.append(100.0 * (r['oracle'] - r['default']))
    res['aggregate']['oracle_gap_pp'] = {
        'per_seed': oracle_gaps,
        'mean': sum(oracle_gaps) / len(oracle_gaps) if oracle_gaps else None,
    }
    if gains and oracle_gaps:
        res['aggregate']['recovery_pct'] = 100.0 * (
            (sum(gains) / len(gains)) / (sum(oracle_gaps) / len(oracle_gaps)))

    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print('seeds:', res['seeds'])
    print('%-42s %s' % ('row', '  '.join('seed %s' % t for t in res['seeds'])))
    for key, label in ROWS:
        cells = []
        for t in res['seeds']:
            v = res['per_seed'][t].get(key)
            cells.append('  --  ' if v is None else '%6.2f' % (100 * v))
        print('%-42s %s' % (label[:42], '  '.join(cells)))
    g = res['aggregate']['gain_pp']
    print('\ngain over default: mean %.2f pp  (range %.2f .. %.2f; n=%d seeds)'
          % (g['mean'], g['min'], g['max'], len(g['per_seed'])))
    if 'recovery_pct' in res['aggregate']:
        print('oracle gap: mean %.2f pp  -> recovery %.0f%%'
              % (res['aggregate']['oracle_gap_pp']['mean'],
                 res['aggregate']['recovery_pct']))
    for t in res['seeds']:
        r = res['per_seed'][t]
        print('  seed %s: oracle_hit %s, spearman %s'
              % (t, r.get('oracle_hit'), r.get('spearman')))
    print('\nwritten -> %s' % OUT)


if __name__ == '__main__':
    main()
