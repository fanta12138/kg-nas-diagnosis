# -*- coding: utf-8 -*-
"""Is Table 1's 14-way backbone ranking a seed-42 artefact? (reviewer M7)

Table 1 reports a rank column from a single seed, with adjacent ranks separated by
0.08 points -- far below the paper's own 2.98-point resolvability floor. This script
reports the ranking under every seed available in the truth matrix, and under the
all-seed mean, so the reader can see whether the ordering is stable or arbitrary.

Only 6 of the 14 backbones have all three seeds, so the all-seed comparison is
restricted to those six; the full 14-way ordering is reported per seed where
available.

Output: rank_stability_control.json
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'phaseB_truth.json')


def main():
    with open(SRC, encoding='utf-8') as fh:
        raw = json.load(fh)
    cells, seeds = {}, set()
    for k, v in raw.items():
        c, b, s = k.split('|')
        cells[(b, c, s)] = v['auroc'] * 100
        seeds.add(s)

    bbs = sorted({b for b, _, _ in cells})
    cats = sorted({c for _, c, _ in cells})
    out = {}

    for s in sorted(seeds):
        means = {}
        for b in bbs:
            vals = [cells[(b, c, s)] for c in cats if (b, c, s) in cells]
            if vals:
                means[b] = statistics.mean(vals)
        order = sorted(means, key=lambda b: -means[b])
        out[f'seed_{s}'] = {'n_backbones': len(order), 'order': order,
                            'mean_auroc': {b: means[b] for b in order},
                            'spread': max(means.values()) - min(means.values())}
        print(f'seed {s}: n={len(order):2d} spread={out[f"seed_{s}"]["spread"]:.2f}')
        print('   ' + ' > '.join(f'{b}({means[b]:.2f})' for b in order[:8]))

    # who is replicated across all seeds
    full = [b for b in bbs if all((b, cats[0], s) in cells for s in seeds)]
    out['n_backbones_with_all_seeds'] = len(full)
    out['backbones_with_all_seeds'] = full
    print(f'\nreplicated across all seeds: {len(full)} of {len(bbs)} -> {full}')

    # seed dispersion per replicated backbone (over category means)
    disp = {}
    for b in full:
        per = [statistics.mean(cells[(b, c, s)] for c in cats) for s in sorted(seeds)]
        disp[b] = {'mean': statistics.mean(per), 'sd': statistics.stdev(per),
                   'per_seed': per}
        print(f'   {b:30s} mean {disp[b]["mean"]:6.2f} sd {disp[b]["sd"]:.3f}')
    out['replicated_seed_dispersion'] = disp

    # rank moves among the replicated six
    ranks = {}
    for s in sorted(seeds):
        order = sorted(full, key=lambda b: -statistics.mean(cells[(b, c, s)] for c in cats))
        for i, b in enumerate(order, 1):
            ranks.setdefault(b, []).append(i)
    out['rank_by_seed_replicated_six'] = ranks
    print('\nranks among the six replicated backbones, per seed:', ranks)
    moves = {b: max(r) - min(r) for b, r in ranks.items()}
    print('max rank move across seeds:', max(moves.values()), moves)

    with open(os.path.join(HERE, 'rank_stability_control.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print('\nwrote rank_stability_control.json')


if __name__ == '__main__':
    main()
