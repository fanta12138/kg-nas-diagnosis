# -*- coding: utf-8 -*-
"""Does the search's selection survive filtering degenerate evaluations?

The paper's mechanistic claim is that the failure is an *interaction* between an
objective that can silently return the optimum of its own range and a minimising
selection rule. The direct test is to re-run the same selection with trials that
scored exactly 0.0 removed, and see what the search would have chosen.

If filtering the degenerates yields ordinary backbones, the claim is that the
space is fine and the objective/rule pair is what breaks.

Output: degenerate_filter_control.json
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'search_repeats.json')


def main():
    with open(SRC, encoding='utf-8') as fh:
        runs = json.load(fh)

    out, picked_all, picked_clean = {}, collections.Counter(), collections.Counter()
    for rn, r in sorted(runs.items()):
        best_all, best_clean = {}, {}
        for t in r['all_results']:
            cfg = (t['architecture']['backbone'], t['architecture']['feature_dim'])
            best_all[cfg] = min(best_all.get(cfg, 1e18), t['score'])
            if t['score'] > 0.0:
                best_clean[cfg] = min(best_clean.get(cfg, 1e18), t['score'])
        a = min(best_all, key=lambda c: best_all[c])
        c = min(best_clean, key=lambda k: best_clean[k]) if best_clean else None
        picked_all[a[0]] += 1
        if c:
            picked_clean[c[0]] += 1
        out[rn] = {
            'argmin_with_degenerates': {'backbone': a[0], 'dim': a[1],
                                        'score': best_all[a], 'true_auroc': None},
            'argmin_filtering_zeros': ({'backbone': c[0], 'dim': c[1],
                                        'score': best_clean[c]} if c else None),
        }
        print(f'{rn}: argmin(all)={a[0]:24s} {best_all[a]:6.3f} | '
              f'filtered={c[0] if c else "-":24s} '
              f'{best_clean[c] if c else float("nan"):6.3f}')

    out['_summary'] = {
        'backbones_chosen_with_degenerates': dict(picked_all),
        'backbones_chosen_filtering_zeros': dict(picked_clean),
    }
    print('with degenerates :', dict(picked_all))
    print('filtering zeros  :', dict(picked_clean))
    with open(os.path.join(HERE, 'degenerate_filter_control.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print('wrote degenerate_filter_control.json')


if __name__ == '__main__':
    main()
