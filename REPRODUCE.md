# Reproducing every number in the paper

Two levels of reproduction are possible from this repository.

* **Level 1 — from the frozen results.** Every table, figure and quoted statistic
  in the paper is computed from JSON records that ship with the repository. These
  commands need **no GPU and no dataset**; they take seconds to a few minutes.
* **Level 2 — from scratch.** Re-running the experiments themselves needs one GPU,
  the MVTec-AD dataset, and hours to days (the 210-detector matrix alone is 210
  training runs of 500 epochs). Level 2 is documented at the end.

Level 1 was verified end-to-end on the released tree: all five figures regenerate
**byte for byte**, and every script below reproduces the numbers printed in the
paper.

---

## Level 1 — artifacts, by paper location

### §4.1 Two thirds of the search space is deployment-inert

| Paper statement | Where it comes from |
|---|---|
| 42 configurations contain 14 distinct models; 28 are deployment-equivalent | **Source inspection**, not a computation: `unsupervised_nas/methods/rd_detector.py` line 109 (`self._init_bottleneck(f.shape[1], 256, f.device)`) against `nas_engine.py`, which constructs the decoder with the requested width and never passes `bottleneck_dim`. |
| 4 recorded searches: same backbone, different dimension | `unsupervised_nas/experiments/stage2*/` — `search_results.json` + `best_architecture.yaml` in each of the four run directories. |
| 19 recorded searches, 834 trials | `search_forensics/search_forensics.json` |

```bash
python search_forensics/analyze.py          # prints runs=19, trials=834
```

### Table 2 (`tab:proxy_repeats`) and Figure 2 (`fig:proxy_repeatability`)

Ten independent evaluations of each of the 42 configurations on MVTec `wood`
(420 calls to the search's own `evaluate_architecture`).

| Number | Value |
|---|---|
| `efficientnet_b1` zero-score rate | 73% |
| `mobilenetv3_small_100` zero-score rate | 13% |
| degenerate fraction over all 420 calls | 6.2% |
| median proxy span across the 12 well-behaved backbones | 5.7% (7.538 – 7.971) |

```bash
python make_figures_v2.py                   # writes paper/figures_v2/fig2_*.{png,pdf}
```
Frozen record: `proxy_diag/phaseA_proxy.json`
Producer (Level 2): `python proxy_diag/proxy_diag_v2.py --phase A --repeats 10`

### Table 3 (`tab:degeneracy`)

Where degenerate zero-score trials occur in the 19 recorded runs, and whether the
plain `argmin` lands on one: **67/834 = 8.0%** of trials, **14 of 19 runs**.

```bash
python search_forensics/analyze.py
```
Frozen record: `search_forensics/search_forensics.json`

### Table 4 (`tab:capacity_null`)

Five parameter-capacity bands; the selected backbone is retrained on all 15
categories within each band. Budget ×9.2 → **+1.06 points**; full spread **1.70
points**; smallest paired *t*-test *p* = **0.058**.

```bash
python capacity_experiment/code/final_analysis.py
```
Frozen records: `capacity_experiment/results/budget_*/`

### Table 5 (`tab:variance`) and Figure 3 (`fig:variance_decomposition`)

Two-way decomposition of the 210-detector truth matrix.

| Source | Share | Test |
|---|---|---|
| backbone | **0.54%** | `F(13,182) = 1.844`, `p = 0.039` |
| category | **95.32%** | |
| interaction (error) | 4.13% | |

```bash
python proxy_diag/samplesize_analysis.py    # prints the shares, F and p
```
Frozen record: `proxy_diag/phaseB_truth.json` (210 cells; 90 of them replicated over
three training seeds)

### §4.4 Resolvability — the paper's headline measurement

| Quantity | Value | Meaning |
|---|---|---|
| σ_seed | **1.96 pp** | pooled within-cell SD over training seeds (180 residual df) |
| σ_res | **3.30 pp** | two-way ANOVA residual, single-seed matrix |
| σ_Δ | **4.12 pp** | SD of the per-category difference between two backbones (91 pairs) |
| runs per backbone to resolve +0.5 pp on one category | **240** | 80% power, two-sided α = 0.05 |
| minimum detectable difference, paired over 15 categories | **2.98 pp** | against a 3.98 pp backbone spread and a 0.5 pp contested margin |
| categories needed to resolve +0.5 pp | **532** | |
| same calculation for a PatchCore-style baseline (σ = 0.51 pp) | **17 runs** | shows the requirement is method-specific |

```bash
python proxy_diag/samplesize_analysis.py
# -> also writes proxy_diag/samplesize_analysis.json
```

### Figure 4 (`fig:search_vs_random`) and §4.5

Twelve independent runs of the search (100 trials each) versus uniform-random
selection.

| Number | Value |
|---|---|
| runs returning a collapse-prone backbone | **12 / 12** (8 × `efficientnet_b1`, 4 × `mobilenetv3_small_100`) |
| probability under uniform sampling | 7.2 × 10⁻¹¹ |
| search mean AUROC | **80.91%** |
| uniform-random mean AUROC | **81.65%** (bootstrap s.d. 0.32) |
| best backbone (`resnet18`) | **83.34%** |
| percentile of the search's output | **1.2th** |

```bash
python search_repeats/authoritative_analysis.py
```
Frozen records: `search_repeats/search_repeats.json`,
`search_repeats/search_repeats_summary.json`

### Table 6 (`tab:fusion`) and §5.2 — the scoring-policy axis

82 policies evaluated on one frozen detector per category.

| Policy | Mean AUROC | Worst category |
|---|---|---|
| `mean`, uniform (published default) | 82.80 | 42.22 |
| `max`, uniform | 86.84 | — |
| top-*k*, uniform | 90.21 | 78.37 |
| best single policy (`w0.3_0.6_0.1`, top-*k*) | **90.60** | — |
| per-category oracle | **93.72** | 84.67 |

Three-seed aggregate (seeds 42/43/44): **82.36 → 90.28, +7.92 points.**

```bash
# the frozen per-seed summaries
python -c "import json;print(json.load(open('fusion_policy/fusion_summary.json'))['default_mean'])"
```
Frozen records: `fusion_policy/fusion_summary.json`,
`fusion_multiseed/fusion_multiseed_summary.json`

### Table 7 (`tab:ksweep`) and Figure 5 (`fig:ksweep`)

Sensitivity of the pooling gain to the top-*k* fraction: a broad plateau, maximum
**91.08%** at *k* = 5%, every *k* in 0.5–20% recovering 7–8.4 points.

```bash
python make_figures_v2.py
```
Frozen record: `fusion_ksweep/fusion_ksweep_summary.json`

> **Note the 0.13-point discrepancy.** The k-sweep reads the default policy as
> 82.67% while Table 6 reads 82.80%, both on seed 42 — two independent evaluation
> scripts on the same trained models. The paper reports this rather than averaging
> it away; see `docs/notes_zh/11_声明与引用审计_zh.md` §C.1.

### Table 8 (`tab:pseudo`), Figure 6 (`fig:label_free`) and §5.4

Label-free selection from synthetic pseudo-anomalies, over three independently
trained students.

| Seed | default | label-free | oracle | gain | recovery |
|---|---|---|---|---|---|
| 42 | 81.62 | 89.99 | 93.19 | +8.37 | 72.4% |
| 43 | 81.90 | 89.96 | 93.36 | +8.07 | 70.4% |
| 44 | 80.98 | 85.81 | 92.90 | +4.83 | 40.5% |
| **mean** | **81.50** | **88.59** | **93.15** | **+7.09** | **60.9%** |

```bash
python pseudo_select/aggregate_multiseed.py
```
Frozen records: `pseudo_select/pseudo_summary{,_s43,_s44}.json`

### §4.5 The synthetic reproduction of the mechanism

A code-independent simulation of the degenerate-objective × `argmin` interaction:
**95%** of the search's advantage over random is lost when collapse-prone
candidates are present, and restored to +2.38 pp by filtering degenerate results.

```bash
python synthetic_mechanism/synthetic_mechanism.py
```
Frozen record: `synthetic_mechanism/synthetic_mechanism.json`

### Table 1 and Table 9 (`tab:controlled`)

* Table 1 (published AUROC values) reports numbers **as published by the original
  authors**; they were not re-run here and no script produces them.
* Table 9 is the 14-backbone controlled comparison: `proxy_diag/phaseB_truth.json`,
  aggregated by `make_figures_v2.py` (Figure 3 path) and by
  `proxy_diag/samplesize_analysis.py`.

---

## Release tooling

```bash
# citation audit: resolve every DOI through Crossref, fall back to DataCite
python adbench/audit_citations.py                     # writes adbench/citation_audit.json
python adbench/audit_citations.py path/to/other.bib

# abstract / keyword / title statistics for the manuscript
python adbench/abstract_stats.py

# number-diff: guard against silently losing a number while editing the manuscript
python adbench/numsnap.py save paper/sections/_draft_v2 before.json
python adbench/numsnap.py diff before.json paper/sections/_draft_v2

# drop bibliography entries that no longer receive a \cite
python adbench/trim_bib.py
```

The manuscript compiles standalone from `paper/`:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main_v2.tex   # 22 pages, 0 undefined
```

---

## Level 2 — re-running the experiments

Prepare the data first ([`docs/DATA.md`](docs/DATA.md)), then:

```bash
cd unsupervised_nas

# Stage 1 — teacher pre-training (contrastive, 100 epochs, MVTec `wood`)
python scripts/stage1_train_teacher.py

# Stage 2 — the search itself, 100 trials from the 42-cell space
python scripts/stage2_nas_search.py

# Stage 3 — train the selected architecture as an RD detector on all categories
python scripts/stage3_final_training.py
```

Then the diagnostic experiments (each is a standalone script; run from the
repository root):

| Experiment | Command | Cost |
|---|---|---|
| Proxy repeatability (420 calls) | `python proxy_diag/proxy_diag_v2.py --phase A --repeats 10` | ~1 GPU-hour |
| 210-detector truth matrix | `python proxy_diag/proxy_diag_v2.py --phase B --epochs 500` | ~210 × 500 epochs — the dominant cost |
| Analysis of the matrix | `python proxy_diag/proxy_diag_v2.py --phase C` | minutes |
| 12 search re-runs | `python search_repeats/search_repeats.py --runs 12 --trials 100` | 12 × 100 × 3 quick epochs |
| Scoring-policy sweep | `python fusion_policy/fusion_policy.py` | 15 trainings, then 82 policies each |
| top-*k* sweep | `python fusion_ksweep/fusion_ksweep.py --seed 42` | reuses the trained detectors |
| Three-seed replication | `python fusion_multiseed/fusion_multiseed.py --seeds 43 44` | 2 × 15 trainings |
| Label-free selection | `python pseudo_select/pseudo_select.py --train-seed 43 --tag s43` | 15 trainings per seed |
| Synthetic mechanism | `python synthetic_mechanism/synthetic_mechanism.py` | CPU, minutes |
| Three-dataset harness | `python adbench/run_matrix.py --dataset mvtec --backbones resnet50.a1_in1k --seeds 0 1 2` | minutes per cell (feature-only) |

**Not currently runnable from this repository:** the manuscript's own history of
four *recorded* searches predates the current revision of the search engine, and
the paper says so. The records are shipped
(`unsupervised_nas/experiments/stage2*/`) and analysed by
`search_forensics/analyze.py`; they are not reproduced by re-running.

---

## Environments actually used

| Component | Version |
|---|---|
| Python | 3.9.13 (Anaconda) |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.4, single RTX 3090 (24 GB) |
| numpy / scikit-learn / matplotlib | 1.26.4 / 1.0.2 / 3.9.4 |
| TeX Live | 2026 (`sn-jnl` class, two-column) |

Seed noise figures are tied to this stack; expect small differences elsewhere.
