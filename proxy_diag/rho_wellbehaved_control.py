# -*- coding: utf-8 -*-
"""Rank transfer on the well-behaved backbones (the reviewer-requested control).

Section 4 reports a mean Spearman rho of 0.215 between two categories' orderings of
the 14 backbones, with 31% of pairs reversing. Two objections are natural:

  (a) pooled vs per-category -- it matters, so state which. Here: per backbone.
  (b) tie handling -- exact duplicates exist in the AUROC matrix (equal image
      counts can give equal AUROC), and a naive rank assignment inflates or
      deflates rho. Here: average ranks for ties (the standard convention).
  (c) the two collapse-prone backbones have no special role in the *truth*
      matrix, but a reviewer may want the control that excludes them anyway.

Output: rho_wellbehaved_control.json
"""
import itertools
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'phaseB_truth.json')
COLLAPSE = ('efficientnet_b1', 'mobilenetv3_small_100')


def average_ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float('nan')


def spearman(a, b):
    return pearson(average_ranks(a), average_ranks(b))


def main():
    with open(SRC, encoding='utf-8') as fh:
        raw = json.load(fh)
    cells = {}
    for k, v in raw.items():
        c, b, s = k.split('|')
        if s == '42':
            cells[(b, c)] = v['auroc']
    bbs = sorted({b for b, _ in cells})
    cats = sorted({c for _, c in cells})

    out = {}
    for label, subset in (('all_14_backbones', bbs),
                          ('excl_2_collapse_prone', [b for b in bbs if b not in COLLAPSE]),
                          ('excl_collapse_and_b0',
                           [b for b in bbs if b not in COLLAPSE + ('efficientnet_b0',)])):
        rhos = [spearman([cells[(b, c1)] for b in subset],
                         [cells[(b, c2)] for b in subset])
                for c1, c2 in itertools.combinations(cats, 2)]
        rhos = [r for r in rhos if not math.isnan(r)]
        m, sd = statistics.mean(rhos), statistics.stdev(rhos)
        t = m / (sd / math.sqrt(len(rhos)))
        out[label] = {'n_backbones': len(subset), 'n_pairs': len(rhos),
                      'mean_rho': m, 'sd': sd,
                      'frac_reversed': sum(1 for r in rhos if r < 0) / len(rhos),
                      't_stat': t}
        print(f'{label:26s} n_bb={len(subset):2d} pairs={len(rhos):3d} '
              f'rho={m:+.4f} sd={sd:.3f} rev={out[label]["frac_reversed"]:.3f} t={t:.2f}')

    with open(os.path.join(HERE, 'rho_wellbehaved_control.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print('wrote rho_wellbehaved_control.json')


if __name__ == '__main__':
    main()
