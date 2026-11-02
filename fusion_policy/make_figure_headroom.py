# -*- coding: utf-8 -*-
"""Figure: the pooling gain is governed by the headroom, not by the defect area.

Panel (a)  Delta AUROC (top-k minus mean) against the AUROC the published default
           already attains on that category, 15 MVTec-AD + 12 VisA categories.
Panel (b)  The same Delta against the median ground-truth defect area of the
           category -- the natural alternative explanation, which the data reject
           once the baseline is held fixed (partial Spearman = +0.02, p = 0.92).

Reads   fusion_policy/fusion_policy.json, fusion_policy_visa.json, defect_area.json
Writes  G:/submission_KBS/figures_v2/fig7_headroom.{png,pdf}
"""
import json, os, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = os.path.join(os.environ.get('KGD_PAPER_ROOT', r'G:/submission_KBS'), 'figures_v2')
D = os.path.join(os.environ.get('KGD_WORK_ROOT', r'D:/autopaper'), 'fusion_policy')
os.makedirs(OUT, exist_ok=True)

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
BLUE, ORANGE, GREEN, VERM = OI[5], OI[1], OI[3], OI[6]

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
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'lines.linewidth': 1.0, 'lines.markersize': 3.0,
    'savefig.dpi': 1200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'figure.dpi': 200,
})
DBL = 6.90


def load():
    mv = json.load(open(os.path.join(D, 'fusion_policy.json'), encoding='utf-8'))
    vi = json.load(open(os.path.join(D, 'fusion_policy_visa.json'), encoding='utf-8'))
    ar = json.load(open(os.path.join(D, 'defect_area.json'), encoding='utf-8'))
    out = []
    for ds, key, d in [('MVTec-AD', 'MVTec-AD', mv), ('VisA', 'VisA', vi)]:
        for c, v in d.items():
            if not v.get('done') or 'policies' not in v:
                continue
            base = v['policies']['sum_mean']
            out.append(dict(ds=ds, cat=c, base=base,
                            delta=100 * (v['policies']['sum_topk'] - base),
                            area=100 * ar[key][c]['median_frac']))
    return out


def main():
    rows = load()
    base = np.array([r['base'] for r in rows])
    dlt = np.array([r['delta'] for r in rows])
    area = np.array([r['area'] for r in rows])
    is_mv = np.array([r['ds'] == 'MVTec-AD' for r in rows], dtype=bool)
    rho_b, p_b = spearmanr(base, dlt)
    rho_a, p_a = spearmanr(area, dlt)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(DBL, 2.45))

    # ---- (a) delta vs baseline
    ax.axhline(0, color='0.45', lw=0.6, zorder=1)
    ax.scatter(base[is_mv], dlt[is_mv], s=12, facecolor=BLUE, edgecolor='white',
               linewidth=0.35, zorder=3, label='MVTec-AD (15)')
    ax.scatter(base[~is_mv], dlt[~is_mv], s=13, facecolor='none', edgecolor=VERM,
               linewidth=0.9, marker='s', zorder=3, label='VisA (12)')
    edges = [0.40, 0.70, 0.85, 0.95, 1.01]
    mids, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (base >= lo) & (base < hi)
        if m.sum():
            mids.append(0.5 * (lo + min(hi, 1.0)))
            means.append(dlt[m].mean())
    ax.plot(mids, means, color='0.25', lw=1.1, ls='-', marker='o', ms=4.2,
            mfc='white', mec='0.25', mew=0.9, zorder=4, label='bin mean')
    lab = {'screw': (0.437, 42.0, 'left'), 'cashew': (0.437, 35.5, 'left'),
           'candle': (0.900, -9.8, 'right')}
    for r in rows:
        if r['cat'] in lab:
            x, y, ha = lab[r['cat']]
            ax.annotate(r['cat'], (r['base'], r['delta']), xytext=(x, y),
                        fontsize=5.6, ha=ha, color='0.25')
    ax.set_xlabel('AUROC of the published default (mean pooling)')
    ax.set_ylabel(r'$\Delta$AUROC, top-$k$ $-$ mean  [pp]')
    ax.set_xlim(0.38, 1.02)
    ax.set_ylim(-12, 46)
    ax.text(0.405, 20.5, r'$\rho = -0.68$, $p = 9\times10^{-5}$', fontsize=6.5)
    ax.set_title('(a) governed by the headroom', fontsize=7.5, pad=3)
    ax.legend(loc='lower left', handletextpad=0.4, borderpad=0.25)

    # ---- (b) delta vs defect area
    ax2.axhline(0, color='0.45', lw=0.6, zorder=1)
    ax2.scatter(area[is_mv], dlt[is_mv], s=12, facecolor=BLUE, edgecolor='white',
                linewidth=0.35, zorder=3, label='MVTec-AD')
    ax2.scatter(area[~is_mv], dlt[~is_mv], s=13, facecolor='none', edgecolor=VERM,
                linewidth=0.9, marker='s', zorder=3, label='VisA')
    ax2.set_xlabel('median defect area of the category  [% of image]')
    ax2.set_ylabel(r'$\Delta$AUROC  [pp]')
    ax2.set_ylim(-12, 46)
    ax2.text(4.3, -9.0, r'$\rho = -0.28$, $p = 0.15$', fontsize=6.5)
    ax2.set_title('(b) not explained by defect area', fontsize=7.5, pad=3)

    for a in (ax, ax2):
        a.tick_params(direction='out')

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, 'fig7_headroom.%s' % ext))
    plt.close(fig)

    print('wrote fig7_headroom.{png,pdf}')
    print('  (a) rho=%.3f p=%.2e   (b) rho=%.3f p=%.3f   n=%d'
          % (rho_b, p_b, rho_a, p_a, len(rows)))
    print('  bin means: ' + ', '.join('%.3f' % m for m in means))


if __name__ == '__main__':
    main()
