# -*- coding: utf-8 -*-
"""How much of the label-free gain is the pooling rule itself?

The label-free selection experiment reports a gain over the pipeline's default
scoring policy (an unweighted mean over all pixels). That default is the weakest
member of the policy family, so the reported gain mixes two effects:

  (a) adopting top-k pooling at all -- a *static* choice that needs no selection
      procedure and no labels either, and
  (b) choosing *which* policy to use per category, which is what the label-free
      criterion actually does.

This script separates them, on the same trained models and the same 70/30 split
that pseudo_select.py uses. It reads the per-policy AUROC already stored by that
script, so it needs no retraining.

Output: baseline_decomposition.json
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = {
    42: 'pseudo_summary.json',
    43: 'pseudo_summary_s43.json',
    44: 'pseudo_summary_s44.json',
}


def load(fn):
    with open(os.path.join(HERE, fn), encoding='utf-8') as fh:
        return json.load(fh)['per_category']


def main():
    out = {}
    for seed, fn in SEEDS.items():
        pc = load(fn)
        cats = sorted(pc)

        def mean_of(key):
            return 100 * statistics.mean(pc[c]['policies'][key]['auroc'] for c in cats)

        selected = 100 * statistics.mean(
            pc[c]['policies'][max(pc[c]['policies'],
                                  key=lambda p: pc[c]['policies'][p]['pseudo_auc'])]['auroc']
            for c in cats)
        oracle = 100 * statistics.mean(
            max(pc[c]['policies'][p]['auroc'] for p in pc[c]['policies']) for c in cats)

        base = mean_of('sum_mean')
        topk = mean_of('sum_topk')
        out[seed] = {
            'n_categories': len(cats),
            'default_mean': base,
            'static_topk': topk,
            'label_free_selected': selected,
            'per_category_oracle': oracle,
            'gain_default_to_topk': topk - base,
            'gain_topk_to_selected': selected - topk,
            'gain_default_to_selected': selected - base,
            'oracle_gap_from_default': oracle - base,
        }
        r = out[seed]
        print(f'seed {seed}: default {base:.2f} | static top-k {topk:.2f} '
              f'(+{r["gain_default_to_topk"]:.2f}) | selected {selected:.2f} '
              f'({r["gain_topk_to_selected"]:+.2f}) | oracle {oracle:.2f}')
        print(f'   of the {r["oracle_gap_from_default"]:.2f}-point oracle gap, '
              f'top-k alone recovers {r["gain_default_to_topk"]:.2f} '
              f'({100 * r["gain_default_to_topk"] / r["oracle_gap_from_default"]:.0f}%) '
              f'and the criterion adds {r["gain_topk_to_selected"]:+.2f}')

    for key in ('gain_default_to_topk', 'gain_topk_to_selected'):
        vals = [out[s][key] for s in SEEDS]
        out.setdefault('summary', {})[key] = {
            'mean': statistics.mean(vals), 'values': vals,
        }
        print(f'{key}: mean {statistics.mean(vals):+.2f} over seeds {vals}')

    with open(os.path.join(HERE, 'baseline_decomposition.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print('wrote baseline_decomposition.json')


if __name__ == '__main__':
    main()
