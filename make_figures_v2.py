# -*- coding: utf-8 -*-
"""Generate the six figures for the rewritten manuscript.

Style: Times New Roman, 1200 dpi (Springer line-art requirement), Okabe-Ito
colour-blind-safe palette, single-column width unless a figure needs the full
text width.

Data sources (all frozen):
  proxy_diag/phaseA_proxy.json          -> Fig 2  proxy repeatability
  proxy_diag/final_truth_analysis.json  -> Fig 3  variance decomposition
  search_repeats/search_repeats_summary.json -> Fig 4  search vs random
  fusion_ksweep/fusion_ksweep_summary.json   -> Fig 5  top-k sensitivity
  pseudo_select/pseudo_summary.json     -> Fig 6  label-free selection

Usage: python make_figures_v2.py
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

OUT = _os.path.join(_KGD_PAPER, 'figures_v2')
D = _KGD_ROOT
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW, PINK = OI[5], OI[1], OI[3], OI[6], OI[7], OI[2], OI[4], OI[0]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'legend.frameon': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.2,
    'ytick.major.size': 2.2,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.0,
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'figure.dpi': 200,
})
COL = 3.35     # single column width (in)
DBL = 6.90     # full text width (in)


def jload(p):
    return json.load(open(p, encoding='utf-8'))


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)))
    plt.close(fig)
    print('  wrote %s.{png,pdf}' % name)


# ==================================================================== Fig 2
def fig2_proxy():
    d = jload(os.path.join(D, 'proxy_diag', 'phaseA_proxy.json'))
    rows = []
    for k, v in d.items():
        s = np.array(v['scores'], dtype=float)
        if len(s) < 10:
            continue
        rows.append((v['backbone'], v['dim'], s, 100.0 * (s == 0).mean()))
    rows.sort(key=lambda r: -r[3])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(DBL, 2.35),
                                  gridspec_kw={'width_ratios': [2.15, 1]})

    # panel a: every configuration's repeats, on a log scale
    bbs = []
    for bb, dim, s, z in rows:
        if bb not in bbs:
            bbs.append(bb)
    FLOOR = 0.05           # exact zeros are plotted on the axis floor and marked
    for i, bb in enumerate(bbs):
        for bb2, dim, s, z in rows:
            if bb2 != bb:
                continue
            xs = np.full(len(s), i) + np.linspace(-0.26, 0.26, len(s))
            ys = np.where(s == 0.0, FLOOR, s)
            isz = (s == 0.0)
            ax.plot(xs[~isz], ys[~isz], ls='none', marker='o', ms=1.6,
                    mfc=BLUE, mec='none', alpha=0.85)
            ax.plot(xs[isz], ys[isz], ls='none', marker='o', ms=2.2,
                    mfc=VERM, mec='none', alpha=0.9)
    ax.set_yscale('log')
    ax.set_ylim(FLOOR * 0.55, 1e4)
    ax.set_xticks(range(len(bbs)))
    ax.set_xticklabels(bbs, rotation=90, fontsize=5.4)
    ax.set_ylabel('proxy score $s(a)$  (log)')
    ax.axhline(FLOOR, color=VERM, lw=0.5, ls=':')
    ax.set_title('(a) ten repeats per configuration', fontsize=7.5, pad=3)
    ax.text(0.015, 0.06,
            'red = exactly $0.0$ (plotted on the floor)',
            transform=ax.transAxes, fontsize=5.8, color=VERM)

    # panel b: median proxy per backbone, showing the 5.7% span
    # pooled median over the backbone's three dimensions and ten repeats --
    # this is the statistic quoted in Section 4.2 (range 7.538-7.971, span 5.7%)
    med = {}
    for bb, dim, s, z in rows:
        med.setdefault(bb, []).extend(list(s))
    healthy = {k: float(np.median(v)) for k, v in med.items() if min(v) > 1.0}
    order = sorted(healthy, key=lambda k: healthy[k])
    vals = [healthy[k] for k in order]
    ax2.plot(vals, range(len(order)), ls='none', marker='s', ms=2.6, color=GREEN)
    lo, hi = min(vals), max(vals)
    ax2.axvspan(lo, hi, color=GREEN, alpha=0.10)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([k[:16] for k in order], fontsize=5.2)
    ax2.set_xlabel('median proxy')
    ax2.set_title('(b) the twelve non-collapsing backbones\nspan only %.1f%%'
                  % (100 * (hi - lo) / lo), fontsize=7.0, pad=3)
    ax2.xaxis.set_major_locator(MultipleLocator(0.2))
    for a in (ax, ax2):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    save(fig, 'fig2_proxy_repeatability')


# ==================================================================== Fig 3
def fig3_variance():
    d = jload(os.path.join(D, 'proxy_diag', 'final_truth_analysis.json'))
    v = d['variance']
    lab = ['backbone\n(architecture)', 'category', 'backbone\n$\\times$ category']
    val = [v['backbone'], v['category'], v['interaction']]

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    bars = ax.bar(range(3), val, width=0.62,
                  color=[ORANGE, BLUE, '#BBBBBB'], edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(lab, fontsize=6.6)
    ax.set_ylabel('share of variance in AUROC (%)')
    ax.set_ylim(0, 100)
    for b, x in zip(bars, val):
        ax.text(b.get_x() + b.get_width() / 2, x + 1.8, '%.2f%%' % x,
                ha='center', fontsize=6.8)
    ax.annotate('', xy=(0, 6), xytext=(1, 6),
                arrowprops=dict(arrowstyle='<->', lw=0.6, color='black'))
    ax.text(0.5, 9.5, '$176\\times$', ha='center', fontsize=7)
    ax.text(0.02, 0.90, '$F(13,182)=1.844$, $p=0.039$', transform=ax.transAxes,
            fontsize=6.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, 'fig3_variance_decomposition')


# ==================================================================== Fig 4
def fig4_search_vs_random():
    """Reads the authoritative analysis, which uses the single-seed (42) truth
    matrix -- the same matrix as the ranking table and the variance decomposition.
    Recomputed here only for the histogram; all quoted numbers come from the file."""
    a = jload(os.path.join(D, 'search_repeats', 'search_vs_random_final.json'))
    tv = dict(jload(os.path.join(D, 'proxy_diag', 'final_truth_analysis.json'))['ranking'])
    sel = a['selected_backbones']
    n = a['n_runs']
    pool = np.array([tv[b] for b in tv])
    rng = np.random.default_rng(a['bootstrap_seed'])
    boot = np.array([rng.choice(pool, size=n, replace=True).mean()
                     for _ in range(a['bootstrap_draws'])])
    search_mean = a['search_mean_true_auroc']
    oracle = a['oracle_true_auroc']
    rnd_mean = a['random_mean_true_auroc']
    pct = a['percentile_of_search_in_random']

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(DBL, 2.3),
                                  gridspec_kw={'width_ratios': [1.35, 1]})

    ax.hist(100 * boot, bins=60, color=SKY, edgecolor='none', alpha=0.85,
            label='uniform random selection')
    ax.axvline(100 * search_mean, color=VERM, lw=1.2)
    ax.axvline(100 * rnd_mean, color=BLUE, lw=1.0, ls='--')
    ax.axvline(100 * oracle, color=GREEN, lw=1.0, ls=':')
    ax.set_xlabel('true mean AUROC of the selected backbone (%)')
    ax.set_ylabel('bootstrap count')
    ax.set_xlim(100 * pool.min() - 1.6, 100 * oracle + 0.9)
    ax.legend(loc='upper right')
    ax.set_title('(a) the search lands at the %.2fth percentile' % pct,
                 fontsize=7.5, pad=3)
    ymax = ax.get_ylim()[1]
    ax.annotate('search %.2f%%' % (100 * search_mean),
                xy=(100 * search_mean, ymax * 0.62),
                xytext=(100 * search_mean - 2.2, ymax * 0.80),
                fontsize=6.0, color=VERM,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=VERM))
    ax.annotate('oracle %.2f%%' % (100 * oracle),
                xy=(100 * oracle, ymax * 0.42),
                xytext=(100 * oracle - 3.4, ymax * 0.60),
                fontsize=6.0, color=GREEN,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=GREEN))

    rank = {b: i for i, b in enumerate(sorted(tv, key=lambda x: -tv[x]), 1)}
    ys = [rank[b] for b in sel]
    ax2.plot(range(1, n + 1), ys, ls='none', marker='o', ms=3.2,
             mfc=VERM, mec='black', mew=0.3)
    ax2.axhspan(0.4, 4.6, color=GREEN, alpha=0.12)
    ax2.text(0.7, 2.5, 'four strongest backbones', fontsize=5.8, color=GREEN)
    ax2.axhline(a['random_expected_rank'], color=BLUE, lw=0.8, ls='--')
    ax2.text(0.7, 8.0, 'random expectation', fontsize=5.8, color=BLUE)
    ax2.set_xlabel('search run')
    ax2.set_ylabel('true rank of the selected backbone')
    ax2.set_ylim(14.6, 0.4)
    ax2.set_title('(b) no run ever selected one of the four best',
                  fontsize=7.0, pad=3)
    for a_ in (ax, ax2):
        a_.spines['top'].set_visible(False)
        a_.spines['right'].set_visible(False)
    save(fig, 'fig4_search_vs_random')


# ==================================================================== Fig 5
def fig5_ksweep():
    d = jload(os.path.join(D, 'fusion_ksweep', 'fusion_ksweep_summary.json'))
    ks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    mean = [100 * d['policies']['uniform|topk_%g' % k]['mean'] for k in ks]
    worst = [100 * d['policies']['uniform|topk_%g' % k]['worst'] for k in ks]
    base_m = 100 * d['policies']['uniform|mean']['mean']
    base_w = 100 * d['policies']['uniform|mean']['worst']
    x = np.array(ks) * 100

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.semilogx(x, mean, marker='o', color=BLUE, label='mean AUROC')
    ax.semilogx(x, worst, marker='s', color=ORANGE, label='worst category')
    ax.axhline(base_m, color=BLUE, lw=0.7, ls='--')
    ax.axhline(base_w, color=ORANGE, lw=0.7, ls=':')
    ax.text(x[0] * 0.85, base_m + 0.5, 'mean pooling (default)\n%.2f%%' % base_m,
            fontsize=5.8, color=BLUE)
    ax.text(x[0] * 0.85, base_w - 3.4, 'worst category under\nmean pooling, %.2f%%' % base_w,
            fontsize=5.8, color=ORANGE)
    ax.axvspan(0.5, 20, color=GREEN, alpha=0.09)
    ax.axvline(1.0, color='black', lw=0.6, ls='-.')
    ax.text(1.05, 72.0, '$k=1\\%$', fontsize=6.2)
    ax.set_xlabel('top-$k$ fraction of map pixels (%)')
    ax.set_ylabel('AUROC (%)')
    ax.set_ylim(40, 94)
    ax.legend(loc='lower right')
    for a in (ax,):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    save(fig, 'fig5_topk_sensitivity')


# ==================================================================== Fig 6
def fig6_pseudo():
    d = jload(os.path.join(D, 'pseudo_select', 'pseudo_summary.json'))
    pc = d['per_category']
    cats = sorted(pc)
    base = np.array([100 * pc[c]['default'] for c in cats])
    pick = np.array([100 * pc[c]['pseudo_selected_auroc'] for c in cats])
    orac = np.array([100 * pc[c]['oracle_auroc'] for c in cats])
    o = np.argsort(base)
    cats = [cats[i] for i in o]
    base, pick, orac = base[o], pick[o], orac[o]

    fig, ax = plt.subplots(figsize=(COL, 2.6))
    y = np.arange(len(cats))
    for i in range(len(cats)):
        ax.plot([pick[i], orac[i]], [i, i], color='#CCCCCC', lw=0.9, zorder=1)
    ax.plot(base, y, ls='none', marker='o', ms=2.8, mfc='white', mec=BLUE, mew=0.6,
            label='default policy', zorder=3)
    ax.plot(pick, y, ls='none', marker='o', ms=3.0, color=VERM,
            label='label-free selection', zorder=4)
    ax.plot(orac, y, ls='none', marker='^', ms=3.0, mfc='none', mec=GREEN, mew=0.7,
            label='oracle (needs labels)', zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=6.0)
    ax.set_xlabel('AUROC (%)')
    ax.set_xlim(35, 102)
    ax.legend(loc='lower left', ncol=1)
    ax.set_title('mean: %.2f $\\rightarrow$ %.2f (label-free), oracle %.2f'
                 % (base.mean(), pick.mean(), orac.mean()), fontsize=6.8, pad=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, 'fig6_label_free_selection')


if __name__ == '__main__':
    print('generating figures ->', OUT)
    fig2_proxy()
    fig3_variance()
    fig4_search_vs_random()
    fig5_ksweep()
    fig6_pseudo()
    print('done')
