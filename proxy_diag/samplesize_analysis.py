# -*- coding: utf-8 -*-
"""
Sample-size / resolvability analysis for the 210-detector truth matrix.

Purpose
-------
Anchor the study's noise floor to an externally reported effect size, so that
the paper states *how small a difference this design can resolve* rather than
only reporting its own spread.

Inputs
------
  proxy_diag/phaseB_truth.json   210 (category, backbone) cells; 90 of them
                                 replicated over 3 training seeds.
Outputs
-------
  proxy_diag/samplesize_analysis.json

Definitions
-----------
  sigma_seed  : pooled within-cell SD over training seeds (the run-to-run noise
                of a single detector).
  MS_resid    : two-way ANOVA residual mean square on the single-seed matrix;
                this is the working error for a comparison *between* backbones,
                because it also contains the backbone x category interaction.
  sigma_pair  : empirical SD of the per-category difference between two
                backbones; what a paired comparison across categories sees.
"""
import itertools
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = os.path.join(HERE, 'phaseB_truth.json')
OUT = os.path.join(HERE, 'samplesize_analysis.json')

# Two-sided alpha = 0.05, power = 0.80  ->  z_sum = 2.8016
Z_SUM = 1.959963985 + 0.841621234


def load():
    d = json.load(open(TRUTH, encoding='utf-8'))
    cells = {}
    for v in d.values():
        cells.setdefault((v['category'], v['backbone']), {})[v['seed']] = v['auroc']
    return cells


def pooled_seed_sd(cells):
    ss = df = 0.0
    per_cell = []
    for se in cells.values():
        if len(se) > 1:
            vals = list(se.values())
            m = st.mean(vals)
            ss += sum((x - m) ** 2 for x in vals)
            df += len(vals) - 1
            per_cell.append(st.stdev(vals))
    return (math.sqrt(ss / df),
            df,
            len(per_cell),
            st.mean(per_cell),
            st.median(per_cell),
            max(per_cell))


def anova_two_way(M):
    """Balanced two-way ANOVA with interaction, rows=backbones, cols=categories."""
    backbones = sorted(M)
    cats = sorted({c for b in M for c in M[b]})
    a, b = len(backbones), len(cats)
    y = [[M[bb][c] for c in cats] for bb in backbones]
    n = a * b
    grand = sum(sum(r) for r in y) / n
    row_mean = [sum(r) / b for r in y]
    col_mean = [sum(y[i][j] for i in range(a)) / a for j in range(b)]
    ss_rows = b * sum((m - grand) ** 2 for m in row_mean)
    ss_cols = a * sum((m - grand) ** 2 for m in col_mean)
    ss_tot = sum((y[i][j] - grand) ** 2 for i in range(a) for j in range(b))
    ss_res = ss_tot - ss_rows - ss_cols
    ms_rows = ss_rows / (a - 1)
    ms_cols = ss_cols / (b - 1)
    ms_res = ss_res / ((a - 1) * (b - 1))
    return {
        'a': a, 'b': b,
        'ss_rows': ss_rows, 'ss_cols': ss_cols, 'ss_res': ss_res, 'ss_tot': ss_tot,
        'ms_rows': ms_rows, 'ms_cols': ms_cols, 'ms_res': ms_res,
        'F_rows': ms_rows / ms_res, 'F_cols': ms_cols / ms_res,
        'share_rows': ss_rows / ss_tot, 'share_cols': ss_cols / ss_tot,
        'share_res': ss_res / ss_tot,
    }


def paired_diff_sd(M):
    backbones = sorted(M)
    cats = sorted({c for b in M for c in M[b]})
    sds = []
    for b1, b2 in itertools.combinations(backbones, 2):
        sds.append(st.stdev([M[b1][c] - M[b2][c] for c in cats]))
    return {
        'n_pairs': len(sds),
        'mean': st.mean(sds), 'median': st.median(sds),
        'min': min(sds), 'max': max(sds),
    }


def n_unpaired(sigma, delta, z=Z_SUM):
    """Independent two-sample, equal n per arm, two-sided alpha .05, power .80."""
    return 2.0 * (z ** 2) * (sigma ** 2) / (delta ** 2)


def mdd_unpaired(sigma, n, z=Z_SUM):
    """Smallest detectable difference for the unpaired test (n per arm)."""
    return z * math.sqrt(2.0 * sigma ** 2 / n)


def n_paired(sigma_d, delta, z=Z_SUM):
    """Paired design: sigma_d is the SD of one paired observation."""
    return (z ** 2) * (sigma_d ** 2) / (delta ** 2)


def mdd_paired(sigma_d, n, z=Z_SUM):
    """Paired design over n pairs."""
    return z * sigma_d / math.sqrt(n)


def main():
    cells = load()
    sd_seed, df_seed, n_cells, m_cell_sd, med_cell_sd, max_cell_sd = pooled_seed_sd(cells)

    M = {}
    for (c, b), se in cells.items():
        if 42 in se:
            M.setdefault(b, {})[c] = se[42]
    an = anova_two_way(M)
    pd = paired_diff_sd(M)

    sigma_res = math.sqrt(an['ms_res'])          # working error for backbone comparisons
    sigma_pair = pd['mean']

    # External anchors (reported by the original authors, not re-run here):
    #   PatchCore 99.1 -> SimpleNet 99.6 / Dinomaly 99.6  = +0.5 pp
    #   the 2026 survey's "not evidence of progress" threshold = +0.3 pp
    anchors = [0.5, 0.3]

    res = {
        'source': 'proxy_diag/phaseB_truth.json',
        'n_cells': len(cells),
        'n_multiseed_cells': n_cells,
        'seed_noise_pp': {
            'pooled_sd': 100 * sd_seed, 'df': df_seed,
            'mean_of_cell_sd': 100 * m_cell_sd,
            'median_of_cell_sd': 100 * med_cell_sd,
            'max_of_cell_sd': 100 * max_cell_sd,
            'note': ('The mean of per-cell SDs (1.036 pp) is the figure quoted '
                     'in earlier internal notes; it is a ddof=0 mean over 90 '
                     'three-sample cells and understates the noise because the '
                     'per-cell SD distribution is heavily right-skewed '
                     '(median %.3f pp, max %.3f pp). The pooled SD over all '
                     '180 residual df is the honest single number.'
                     % (100 * med_cell_sd, 100 * max_cell_sd)),
        },
        'anova_seed42_matrix': an,
        'paired_diff_sd_pp': {k: (100 * v if isinstance(v, float) else v)
                              for k, v in pd.items()},
        'working_noise_pp': {
            'sigma_anova_residual': 100 * sigma_res,
            'sigma_paired_diff': 100 * sigma_pair,
        },
        'resolvability': {},
    }

    for delta in anchors:
        entry = {'delta_pp': delta}
        # (a) one category, two backbones, independent fresh training runs.
        #     On a single fixed category the only random component is the seed,
        #     so this is an unpaired two-sample comparison with sigma_seed.
        entry['n_per_arm_single_category_unpaired'] = n_unpaired(sd_seed, delta / 100.0)
        # (b) paired across the 15 categories we actually have, one seed each.
        #     sigma_pair is the SD of ONE per-category difference, so the paired
        #     formula applies.
        entry['mdd_with_15_categories_paired'] = 100 * mdd_paired(sigma_pair, 15)
        entry['categories_needed_paired'] = n_paired(sigma_pair, delta / 100.0)
        # (c) our full design: 15 categories x 3 seeds. Split the ANOVA residual
        #     into its seed and interaction parts; the seed part shrinks by sqrt(3).
        inter_sd2 = max(sigma_res ** 2 - sd_seed ** 2, 0.0)
        per_cell_sd = math.sqrt(inter_sd2 + sd_seed ** 2 / 3.0)
        entry['sigma_per_cell_15cat_x3seed_pp'] = 100 * per_cell_sd
        entry['mdd_with_15cat_x3seed_paired'] = 100 * mdd_paired(
            per_cell_sd * math.sqrt(2.0), 15)
        res['resolvability']['delta_%gpp' % delta] = entry

    res['headline'] = {
        'pooled_seed_sd_pp': round(100 * sd_seed, 3),
        'backbone_spread_pp': 100 * (max(st.mean(list(M[b].values())) for b in M)
                                     - min(st.mean(list(M[b].values())) for b in M)),
        'resolve_0.5pp_needs_detectors_per_arm_single_category':
            round(n_unpaired(sd_seed, 0.005)),
        'mdd_single_category_one_shot_pp': round(100 * sd_seed, 3),
        'mdd_paired_15cat_pp': round(100 * mdd_paired(sigma_pair, 15), 3),
        'categories_needed_for_0.5pp': round(n_paired(sigma_pair, 0.005)),
    }

    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print('pooled within-cell seed SD      : %.3f pp (df=%d, %d cells, %d seeds/cell)'
          % (100 * sd_seed, df_seed, n_cells, 3))
    print('  mean of per-cell SD           : %.3f pp   <-- the earlier "1.04"'
          % (100 * m_cell_sd))
    print('  median / max of per-cell SD   : %.3f / %.3f pp'
          % (100 * med_cell_sd, 100 * max_cell_sd))
    print('ANOVA (14 x 15, seed 42)')
    print('  backbone share                : %.2f %%   F=%.3f' % (100 * an['share_rows'], an['F_rows']))
    print('  category share                : %.2f %%   F=%.1f' % (100 * an['share_cols'], an['F_cols']))
    print('  residual share                : %.2f %%' % (100 * an['share_res']))
    print('  sigma_residual                : %.3f pp' % (100 * sigma_res))
    print('paired-difference SD            : mean %.3f pp (median %.3f, max %.3f)'
          % (100 * pd['mean'], 100 * pd['median'], 100 * pd['max']))
    print('backbone spread (mean AUROC)    : %.2f pp' % res['headline']['backbone_spread_pp'])
    print()
    for delta in anchors:
        e = res['resolvability']['delta_%gpp' % delta]
        print('to resolve +%.1f pp (external anchor):' % delta)
        print('  single category, fresh runs   : %.0f detectors per arm (unpaired)'
              % e['n_per_arm_single_category_unpaired'])
        print('  paired over the 15 categories : MDD = %.2f pp  -> needs %.0f categories'
              % (e['mdd_with_15_categories_paired'], e['categories_needed_paired']))
        print('  paired, 15 cat x 3 seeds      : MDD = %.2f pp (per-cell sd %.2f pp)'
              % (e['mdd_with_15cat_x3seed_paired'], e['sigma_per_cell_15cat_x3seed_pp']))
    print('\nwritten -> %s' % OUT)


if __name__ == '__main__':
    main()
