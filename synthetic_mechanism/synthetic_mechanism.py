# -*- coding: utf-8 -*-
"""Minimal synthetic reproduction of the mechanism behind claim 2B.

The claim, stated generally and with no reference to any particular codebase:

    If a scoring function can silently return the optimum of its own range when
    an evaluation fails, and the selection rule minimises a single sample per
    candidate, then the search does not select the best candidate -- it selects
    the candidate whose evaluation is most likely to fail.

This script demonstrates that on a synthetic objective, so the result does not
depend on the authors' pipeline. It then shows that the usual remedy for noisy
search -- more trials, more candidates -- does not help, and that filtering or
repeating the evaluation does.

Nothing here imports project code. numpy only.

Outputs
-------
  synthetic_mechanism.json            all numbers
  fig7_synthetic_mechanism.png/.pdf   the two-panel figure

Usage: python synthetic_mechanism.py
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

import io, json, os, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT = _os.path.join(_KGD_ROOT, 'synthetic_mechanism')
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(OUT, 'synthetic_mechanism.json')

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
# A candidate c has a true quality q(c) that we care about (here: AUROC, higher
# is better). An evaluation returns a loss-like score s(c), lower is better:
#
#     s(c) = (1 - q(c)) * W + noise
#
# A candidate in the "collapse-prone" set C returns the degenerate value 0.0
# with probability p -- modelling an evaluation that failed and returned the
# optimum of the score's own range. 0.0 is below every well-behaved score, so
# under minimisation a collapsed evaluation wins outright.
#
# The search (matching the protocol under study) samples K candidates with
# replacement and returns the argmin of a single sample per candidate.

W = 10.0            # score scale; keeps well-behaved scores well above 0.0


def one_search(q, collapse_p, K, rng, sigma=0.02, repeats=1, rule='single'):
    """Run one search. Returns (index selected, was_collapsed)."""
    n = len(q)
    idx = rng.choice(n, size=K, replace=True)
    if repeats == 1:
        s = (1.0 - q[idx]) * W + rng.normal(0, sigma, K)
        collapsed = rng.random(K) < collapse_p[idx]
        s = np.where(collapsed, 0.0, s)
        j = int(np.argmin(s))
        return idx[j], bool(collapsed[j])

    # repeated evaluation: `repeats` samples per candidate
    S = np.empty((repeats, K))
    C = np.zeros((repeats, K), dtype=bool)
    for r in range(repeats):
        s = (1.0 - q[idx]) * W + rng.normal(0, sigma, K)
        c = rng.random(K) < collapse_p[idx]
        S[r] = np.where(c, 0.0, s)
        C[r] = c
    if rule == 'mean':
        agg = S.mean(axis=0)
    elif rule == 'median':
        agg = np.median(S, axis=0)
    elif rule == 'filter':
        # discard collapsed evaluations; if all repeats collapsed, fall back to
        # a large penalty rather than accepting the degenerate value
        masked = np.where(C, np.nan, S)
        with np.errstate(invalid='ignore'):
            agg = np.nanmean(masked, axis=0)
        allbad = C.all(axis=0)
        agg[allbad] = 1e6
    else:
        raise ValueError(rule)
    j = int(np.argmin(agg))
    return idx[j], bool(C[:, j].any())


def evaluate(q, collapse_p, K, n_rep=4000, seed=0, n_draws=1,
             broken_frac=None, broken_p=None, **kw):
    """Repeat the search many times; report outcome statistics.

    With n_draws > 1 the collapse-prone set is re-drawn each time, so the
    reported gap to random is not an artefact of which candidates happened to be
    designated broken (with only 2 of 42 broken, one fixed draw has an sd of
    ~1.1 pp, which is larger than the effect being measured).
    """
    rng = np.random.default_rng(seed)
    sel, col = [], []
    per = max(1, n_rep // n_draws)
    for _ in range(n_draws):
        cp = collapse_p
        if n_draws > 1:
            cp = np.zeros(len(q), dtype=float)
            b = int(round(broken_frac * len(q)))
            if b:
                cp[rng.choice(len(q), size=b, replace=False)] = broken_p
        for i in range(per):
            a, c = one_search(q, cp, K, rng, **kw)
            sel.append(a); col.append(c)
    sel = np.asarray(sel); col = np.asarray(col, dtype=bool)
    reached = q[sel]
    return {
        'p_collapse_selected': float(col.mean()),
        'mean_true_quality': float(reached.mean()),
        'mean_rank': float(np.mean([1 + int((q > v).sum()) for v in reached])),
        'gap_to_random_pp': float(100 * (reached.mean() - q.mean())),
        'gap_to_oracle_pp': float(100 * (reached.mean() - q.max())),
        'sd_of_gap_pp': float(100 * reached.std(ddof=0) / np.sqrt(len(reached))),
        'n_searches': int(len(reached)),
        'n_broken_draws': int(n_draws),
    }

    


def main():
    out = {}

    # ----------------------------------------------------------------- reference
    n = 42
    # a generic quality vector: spread comparable to the setting under study
    rng0 = np.random.default_rng(7)
    q_gen = np.sort(rng0.normal(0.8165, 0.015, n))[::-1].copy()
    q_gen = np.clip(q_gen, 0.775, 0.835)

    def summarise(tag, **ev):
        ev = dict(ev)
        out[tag] = ev
        print('  %-42s collapse=%.2f  mean=%.4f%%  vs random %+6.2f pp'
              % (tag, ev['p_collapse_selected'], 100 * ev['mean_true_quality'],
                 ev['gap_to_random_pp']))
        return ev

    # ------------------------------------------------------------------ Part 1
    print('[1] No collapse-prone candidates: minimisation is unbiased')
    cp0 = np.zeros(n)
    summarise('no_collapse__K100', **evaluate(q_gen, cp0, 100))
    print('    random expectation %.4f%%   oracle %.4f%%'
          % (100 * q_gen.mean(), 100 * q_gen.max()))

    # ------------------------------------------------------------------ Part 2
    print()
    print('[2] Sweep: fraction of collapse-prone candidates x collapse probability')
    print('    K = 100 trials, as in the protocol under study')
    print('    %-10s %-8s %-10s %-12s %-10s' %
          ('broken', 'p', 'P(sel brk)', 'mean q', 'vs random'))
    sweep = {}
    for frac in (0.0, 0.05, 0.10, 0.20, 0.30):
        b = int(round(frac * n))
        for p in (0.05, 0.25, 0.50, 0.75, 1.00):
            cp = np.zeros(n)
            if b:
                cp[rng0.choice(n, size=b, replace=False)] = p
            ev = evaluate(q_gen, None, 100, seed=11, n_draws=200,
                          broken_frac=frac, broken_p=p)
            sweep['frac%.2f_p%.2f' % (frac, p)] = ev
            print('    %-10.0f%% %-8.2f %-10.2f %-12.4f %+9.2f pp'
                  % (100 * frac, p, ev['p_collapse_selected'],
                     100 * ev['mean_true_quality'], ev['gap_to_random_pp']))
    out['sweep_broken_vs_p'] = sweep

    # ------------------------------------------------------------------ Part 3
    print()
    print('[3] More trials do NOT remove the bias (b=2/42, p=0.73)')
    cp = np.zeros(n)
    cp[rng0.choice(n, size=2, replace=False)] = 0.73
    more = {}
    print('    %-10s %-12s %-12s %-10s' % ('K trials', 'P(sel brk)', 'mean q', 'vs random'))
    for K in (10, 30, 100, 300, 1000, 5000):
        ev = evaluate(q_gen, None, K, seed=23, n_draws=200,
                      broken_frac=2 / 42, broken_p=0.73)
        more['K%d' % K] = ev
        print('    %-10d %-12.2f %-12.4f %+9.2f pp'
              % (K, ev['p_collapse_selected'], 100 * ev['mean_true_quality'],
                 ev['gap_to_random_pp']))
    out['more_trials'] = more

    # ------------------------------------------------------------------ Part 4
    print()
    print('[4] The remedies (b=2/42, p=0.73, K=100)')
    fix = {}
    print('    %-28s %-12s %-12s %-10s' % ('rule', 'P(sel brk)', 'mean q', 'vs random'))
    ev = evaluate(q_gen, None, 100, seed=41, n_draws=200,
                  broken_frac=2 / 42, broken_p=0.73)
    fix['single_sample'] = ev
    print('    %-28s %-12.2f %-12.4f %+9.2f pp'
          % ('single sample (current)', ev['p_collapse_selected'],
             100 * ev['mean_true_quality'], ev['gap_to_random_pp']))
    for rule in ('mean', 'median', 'filter'):
        for R in (3, 10):
            ev = evaluate(q_gen, None, 100, seed=41, n_draws=200,
                          broken_frac=2 / 42, broken_p=0.73,
                          repeats=R, rule=rule)
            fix['%s_R%d' % (rule, R)] = ev
            print('    %-28s %-12.2f %-12.4f %+9.2f pp'
                  % ('%s of R=%d' % (rule, R), ev['p_collapse_selected'],
                     100 * ev['mean_true_quality'], ev['gap_to_random_pp']))
    out['remedies'] = fix

    # ------------------------------------------------------------------ Part 5
    print()
    print('[5] Calibration against the observed setting')
    print('    14 backbones, true AUROC from the measured matrix, two collapse-prone')
    try:
        ft = json.load(open(_os.path.join(_KGD_ROOT, 'proxy_diag', 'final_truth_analysis.json'),
                            encoding='utf-8'))
        q_real = np.array([v for _, v in ft['ranking']])
        names = [b for b, _ in ft['ranking']]
        cp_real = np.zeros(len(q_real))
        cp_real[names.index('efficientnet_b1')] = 0.73
        cp_real[names.index('mobilenetv3_small_100')] = 0.13
        ev = evaluate(q_real, cp_real, 100, n_rep=2000, seed=5)
        out['calibration'] = dict(ev)
        out['calibration']['n_backbones'] = int(len(q_real))
        out['calibration']['random_mean'] = float(q_real.mean())
        out['calibration']['oracle'] = float(q_real.max())
        print('    model    : P(broken selected)=%.2f  mean=%.4f%%  vs random %+.2f pp'
              % (ev['p_collapse_selected'], 100 * ev['mean_true_quality'],
                 ev['gap_to_random_pp']))
        print('    observed :                      mean= 80.9058%  vs random -0.75 pp'
              '  (12/12 runs selected a collapse-prone backbone)')
    except Exception as e:
        print('    (calibration skipped: %s)' % e)

    base = out.get('no_collapse__K100', {}).get('gap_to_random_pp')
    sing = out.get('remedies', {}).get('single_sample', {}).get('gap_to_random_pp')
    filt = out.get('remedies', {}).get('filter_R10', {}).get('gap_to_random_pp')
    if base and sing is not None:
        out['advantage_retained'] = {
            'well_behaved_objective_pp': base,
            'with_collapse_prone_present_pp': sing,
            'fraction_of_advantage_lost': float(1 - sing / base) if base else None,
            'after_filtering_degenerate_evaluations_pp': filt,
        }
        print()
        print('=== HEADLINE ===')
        print('  advantage over random, well-behaved objective : %+.2f pp' % base)
        print('  advantage over random, collapse-prone present : %+.2f pp' % sing)
        print('  => fraction of the search advantage lost      : %.0f%%'
              % (100 * (1 - sing / base)))
        print('  advantage after filtering degenerate results  : %+.2f pp' % filt)
    json.dump(out, open(RES, 'w', encoding='utf-8'), indent=2)
    print()
    print('saved -> %s' % RES)

    # ------------------------------------------------------------------ figure
    try:
        make_figure(q_gen, rng0, out)
    except Exception as e:
        print('figure skipped: %s' % e)


def make_figure(q_gen, rng0, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    BLUE, ORANGE, GREEN, VERM, SKY = OI[5], OI[1], OI[3], OI[6], OI[2]
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 8, 'axes.linewidth': 0.6,
                         'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 6.5,
                         'legend.frameon': False, 'savefig.dpi': 1200,
                         'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.4),
                                  gridspec_kw={'width_ratios': [1.25, 1]})

    # (a) the sweep: gap to random vs collapse probability, per broken fraction
    fracs = [0.05, 0.10, 0.20, 0.30]
    ps = [0.05, 0.25, 0.50, 0.75, 1.00]
    cols = [SKY, BLUE, ORANGE, VERM]
    for f, c in zip(fracs, cols):
        ys = [out['sweep_broken_vs_p']['frac%.2f_p%.2f' % (f, p)]['gap_to_random_pp']
              for p in ps]
        ax.plot(ps, ys, marker='o', ms=3, color=c, label='%d%% collapse-prone' % (100 * f))
    ax.axhline(0, color='black', lw=0.8)
    ax.text(0.03, 0.12, 'no better than random', fontsize=6.2, style='italic')
    ax.set_xlabel('collapse probability $p$ of a broken evaluation')
    ax.set_ylabel('selected quality vs random (pp)')
    ax.legend(loc='lower left')
    ax.set_title('(a) the advantage over random collapses', fontsize=7.5, pad=3)

    # (b) remedies
    labs, vals, cols2 = [], [], []
    order = [('single_sample', 'single\nsample', VERM),
             ('mean_R3', 'mean of\n3', ORANGE), ('mean_R10', 'mean of\n10', ORANGE),
             ('median_R3', 'median\nof 3', BLUE), ('median_R10', 'median\nof 10', BLUE),
             ('filter_R3', 'filter\nR3', GREEN), ('filter_R10', 'filter\nR10', GREEN)]
    for k, lab, c in order:
        if k in out['remedies']:
            labs.append(lab); vals.append(out['remedies'][k]['gap_to_random_pp']); cols2.append(c)
    ax2.bar(range(len(vals)), vals, color=cols2, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(range(len(labs)))
    ax2.set_xticklabels(labs, fontsize=5.6, rotation=30, ha='right')
    ax2.set_ylabel('selected quality vs random (pp)')
    ax2.set_title('(b) averaging does not help; filtering does', fontsize=7.5, pad=3)
    for a in (ax, ax2):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, 'fig7_synthetic_mechanism.%s' % ext))
    plt.close(fig)
    print('figure -> %s' % os.path.join(OUT, 'fig7_synthetic_mechanism.png'))


if __name__ == '__main__':
    main()
