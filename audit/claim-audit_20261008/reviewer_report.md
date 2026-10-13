# Independent Pre-Submission Numeric Audit — KG-NAS (submission_KBS)

Auditor: independent, zero prior knowledge. Mode: read-only, evidence-first.
Manuscript: `G:/submission_KBS` (main_springer.tex + sections/ + tables/ + supplementary/).
Evidence: `D:/autopaper/unsupervised_nas/paper_results/**`, `D:/autopaper/unsupervised_nas/experiments/**`, `D:/autopaper/fig/**`, `D:/autopaper/experiments/**`.
Compute: `C:/Users/dengpeng/anaconda3/python.exe` (pandas/numpy/scipy). Scratch scripts under `.aris/traces/paper-claim-audit/2026-10-08_run01/`.

Note on process: two `grep` sweeps surfaced prohibited artefacts (`D:/autopaper/PROJECT.md`, `D:/autopaper/paper/PAPER_CLAIM_AUDIT.md`, `G:/submission_KBS/tools/_float_audit.py`). Those files were NOT opened/read; only fragments shown incidentally in a match line were seen. No finding below relies on them. Further searches were restricted to allowed evidence roots.

---

### Round verdict

| field | value |
|---|---|
| verdict | **FAIL** |
| reason_code | claim_mismatch |
| total_claims | 178 |
| exact | 152 |
| rounding | 9 |
| ambiguous | 5 |
| missing | 1 |
| mismatch | 11 |

RULE applied: `mismatch > 0` ⇒ FAIL.

---

### Issues found

**[F1] material — Restatement drift (direction) — `sections/4_experiments.tex` vs `main_springer.tex`**
- Quote (abstract, `main_springer.tex:79`): "at 200 epochs ResNet-18 leads on 10 of 15 categories, whereas at 500 epochs **the discovered architecture closes the gap and wins 12 of 15**."
- Quote (body, `sections/4_experiments.tex:112`): "at 500 epochs ResNet-18 **still wins 8** categories while the discovered architecture **wins 3**."
- Evidence: `D:/autopaper/unsupervised_nas/experiments/p2_rd/nas/ep500/<cat>/final_evaluation.json` (NAS) vs `D:/autopaper/unsupervised_nas/experiments/p2_rd/teacher_resnet18/fixed/{resnet18,swin_tiny,mbv3_small}/ep500/<cat>/final_evaluation.json` (fixed).
- Value found: argmax winners over 15 categories = **NAS 3, ResNet-18 8, MobileNetV3 2, Swin-Tiny 2** (matches Table `tab:fixed_kd`).
- Command: `python audit_compute.py` §D → `500-ep winner counts: {'nas':3,'resnet18':8,'swin_tiny':2,'mbv3_small':2}`.
- Why it differs: the abstract says the discovered architecture wins 12/15 and "closes the gap"; the body and the raw per-category files say it wins **3/15** and ResNet-18 still leads (84.0% vs 82.4%). The abstract flips the direction of the headline comparison.

**[F2] material — Restatement drift (direction) — `main_springer.tex:79` and `sections/1_introduction.tex:38`**
- Quote (abstract): "the discovered 5.3M-parameter student attains 82.4% average AUROC across 15 categories, **exceeding a 12.9M ResNet-18 student** under an identical protocol".
- Quote (intro): "**exceeding a 12.9M ResNet-18 student** trained under an identical protocol at less than half the parameter count."
- Evidence: `p2_rd/teacher_resnet18/fixed/resnet18/ep500/<cat>/final_evaluation.json`; Table `tab:fixed_kd` itself (`sections/4_experiments.tex:101`) prints ResNet-18 **84.0** vs discovered **82.4**; `sections/5_conclusion.tex:19` calls it "**98.2% of the accuracy** of a 12.9M ResNet-18".
- Value found: ResNet-18 (fixed, 500 ep) mean AUROC = **83.95%**; discovered = **82.40%**. ResNet-18 is *higher*.
- Command: `python audit_compute.py` §D → `resnet18: mean=0.8395`.
- Why it differs: the paper contradicts itself. Abstract and intro say the searched student *beats* ResNet-18; the table, the conclusion, and the raw files all say ResNet-18 is ahead.

**[F3] material — Config/unit mismatch (parameter counts) — `sections/4_experiments.tex:46,101-104` and `main_springer.tex:79`**
- Quote (`tab:main_results` caption, line 46): "using only **5.3M parameters (student encoder+decoder)**."
- Quote (`tab:fixed_kd`, lines 101–104): "Fixed ResNet-18 & **12.9** & 84.0 …; NAS-Eff-B0 (discovered) & **5.3** …; Fixed MobileNetV3-Small & **2.7** …; Fixed Swin-Tiny & **29.3**". Caption: "Parameter counts are measured for the student (encoder$+$decoder)."
- Evidence: `D:/autopaper/fig/arch_efficiency.csv` (regenerated from the same fixed runs by `fig/generate_figures.py:206-209`) and `D:/autopaper/unsupervised_nas/paper_results/efficiency_2_inference.csv`.
- Value found: encoder+decoder params = **ResNet-18 14.9M, discovered (Eff-B0+256d) 6.0M, MobileNetV3 2.5M, Swin-Tiny 28.0M**. The value **5.3M** is the *backbone-only* EfficientNet-B0 count (also printed in the search-space table, line 95: "EfficientNet-B0 … 5.3"). `scripts/collect_all_results.py` hard-codes the discovered student at `'params_m': 6.0`.
- Command: `python audit_compute.py` (fig CSVs); `cat D:/autopaper/fig/arch_efficiency.csv`.
- Why it differs: (a) 5.3M is a backbone-only figure but is labelled "student encoder+decoder"; (b) all four printed fixed-student counts disagree with the only measured counts (12.9≠14.9, 5.3≠6.0, 2.7≠2.5, 29.3≠28.0). The derived claims "2.4× fewer parameters" and "roughly 44% of the 12–15M baselines" rest on these counts.

**[F4] material — Internally inconsistent duplicate + mismatch — `sections/4_experiments.tex:365` vs `:370`**
- Quote (line 365, Ablation 3): "the single best trial (EfficientNet-B0 + 256-d, **score 0.0097**)".
- Quote (line 370, Ablation 4): "the NAS-discovered configuration (EfficientNet-B0+256d) attained the **single best trial score (0.0023)**".
- Evidence: `D:/autopaper/fig/nas_search_trials.csv` → `efficientnet_b0,256,…,0.009731,93`; and `D:/autopaper/unsupervised_nas/paper_results/ablation_phase1/ablation_3_feature_dim.csv` → 256-dim `min = 0.0022611599415540695`.
- Value found: main-search single best trial = **0.009731**; the **0.0023** figure is the *minimum of the 256-dim subset* in the separate feature-dimension ablation, not the NAS-discovered trial.
- Command: `python -c "import csv;print([r for r in csv.DictReader(open(r'D:/autopaper/fig/nas_search_trials.csv',encoding='utf-8-sig')) if r['Score'].startswith('0.0097')])"`.
- Why it differs: the same quantity is printed as 0.0097 and 0.0023; the evidence supports 0.0097 and re-attributes the 0.0023 value to a different experiment.

**[F5] material — Internally inconsistent duplicate — `sections/4_experiments.tex:217` vs `:190`**
- Quote (`tab:sota_comparison`, line 217): "**KG-NAS (Ours)** & 82.4 & - & **6** & 142$^\dagger$ & Single-class + NAS".
- Quote (`tab:efficiency_comparison`, line 190): "**KG-NAS (Ours)** & 82.40 & **5.3** & 142$^\dagger$ & 102".
- Value found: two different parameter counts for the same model in adjacent tables; the evidence value is 6.0M (`efficiency_2_inference.csv`, `inference_efficiency.csv`).
- Why it differs: same quantity printed as both 5.3 and 6 (see F3).

**[F6] minor — mismatch (rounding direction) — `sections/4_experiments.tex:71,46`**
- Quote (main-table Average row): "… & **78.0** & …" for PaDiM; caption "exceeds PaDiM (**78.0%**)".
- Evidence: `D:/autopaper/fig/baselines_full_results.csv`; `D:/autopaper/fig/baseline_ranking.csv` (=77.92); `tables/BASELINE_ANALYSIS_TABLE.tex` (=77.92).
- Value found: PaDiM mean = **77.9182 → 77.9** (half-up at 1 dp).
- Command: `python audit3.py` §J → `PaDiM avg=77.9182 -> 77.9`.
- Why it differs: 78.0 is not reachable by half-up rounding of 77.92; every other table in the paper prints 77.92/77.9. Magnitude 0.08 pp.

**[F7] minor — Rounding drift (double rounding) — `sections/4_experiments.tex:58,71`**
- Quote: main-table KG-NAS carpet = **77.7**.
- Evidence: `p2_rd/nas/ep500/carpet/final_evaluation.json` → `0.7764847512038523`.
- Value found: 77.648 → half-up **77.6**.
- Why it differs: `category_paradigms.csv` stored 0.7765 (4dp), so the printed 77.7 comes from a double rounding of the raw 77.648. 0.05 pp.

**[F8] minor — Rounding drift (double rounding) — `sections/4_experiments.tex:71`**
- Quote: main-table Average row CFA = **89.8**.
- Evidence: `baselines_full_results.csv`. Value found: CFA mean = **89.7465 → 89.7** (half-up).
- Why it differs: 89.75 (2dp) → 89.8 (1dp) is a two-step path; direct half-up is 89.7.

**[F9] minor — Boundary breach — `sections/4_experiments.tex:78,165` and `supplementary/supplementary.tex:131`**
- Quote: "CFA achieves best performance in 9 out of 15 categories … (strictly 8, **with two ties within 0.02 pp**)", "tying … on bottle and leather **within 0.02 pp**".
- Evidence: `baselines_full_results.csv`.
- Value found: bottle CFA=RevDist=99.6825 (exact tie); leather CFA=96.603 vs PaDiM=96.569 → diff = **0.0340 pp** (> 0.02).
- Command: `python audit3.py` §I → `CFA-PaDiM leather diff pp = 0.0340`.
- Why it differs: the stated bound "within 0.02 pp" is exceeded by the leather pair (0.034 pp).

**[F10] minor — mismatch (count) — `sections/4_experiments.tex:422`**
- Quote: "**CNNs (13 types, 65 trials)** yield a median score of 0.064 … Transformers (3 types, 20 trials) …".
- Evidence: search-space table (`sections/3_method.tex:91-105`) has 15 backbones = **12 CNN + 3 Transformer**; `nas_search_trials.csv` covers **10 CNN** families.
- Value found: 12 CNN backbone types, not 13. (65/20 trials, medians 0.0639/0.0689, std 3.84/0.098, p=0.404 are all correct and reproduce.)
- Why it differs: "13 types" is off by one; it is consistent with a 16-backbone space (13 CNN + 3 TR), reinforcing the 15-vs-16 discrepancy (see U1).

**[F11] minor — Internally inconsistent duplicate — `sections/3_method.tex:265` vs `:202`/`5_conclusion.tex:19`**
- Quote (`tab:hyperparams`, line 265): "NAS trials per config & **$\sim$2.1 (100 total)**". Quote (text/summary): "**$\sim$2.2** trials per architecture".
- Value found: 100 / 45 = **2.22**.
- Why it differs: 2.1 is inconsistent with 100/45 and with the body's own 2.2 (2.1 is what 100/48, a 16-backbone space, would give).

**[F12] minor — Config mismatch — `sections/4_experiments.tex:370`**
- Quote: "256-dim exhibits the highest mean score and variance (16.9529, Std=66.5308) at **~5.3M parameters**".
- Evidence: `D:/autopaper/fig/feature_dim_ablation.csv` → `256,16.9529,66.5308,**6**,Unstable`.
- Value found: 6M.
- Why it differs: 5.3 vs 6 (same 5.3-as-6.0 conflation as F3). Means/stds themselves match exactly.

**[F13] minor — Internal boundary/range inconsistency — `sections/4_experiments.tex:609` vs `:565,591,625`**
- Quote (`tab:cross_dataset_comparison`, line 609): BTAD "Train samples & **400--1000**". Quote (body, line 591): "training sets of **399 to 1000** samples"; line 565 gives Product 02 = **399** training samples.
- Value found: the manuscript's own body value 399 lies below the table's stated lower bound 400.
- Why it differs: the range lower bound excludes the smallest BTAD training set.

**[M1] missing — 107 FPS pipeline — `sections/4_experiments.tex:192,219`, `5_conclusion.tex:23`**
- Quote: "the full teacher--student detector runs at **$\sim$107 FPS**".
- Evidence searched: `D:/autopaper/fig/inference_efficiency.csv` (has student 142 and backbone FPS only), `D:/autopaper/fps_remeasure2.py` (benchmarks resnet50/efficientnet_b0/vit_tiny only), `D:/autopaper/unsupervised_nas/paper_results/efficiency_2_inference.csv`.
- Value found: **no file records a 107 FPS measurement** (a grep for `107` in allowed roots returns only epoch indices `"epoch": 107`).
- Why it differs: untraceable measured control.

---

### Checked and not defects

- **Main results table, all 15 categories × 11 baselines (165 cells)** — every printed 1-dp value equals the half-up rounding of `D:/autopaper/fig/baselines_full_results.csv`. (e.g., grid SPADE 0.39850→39.8, tile FastFlow 0.91450→91.5, zipper PaDiM 0.61450→61.4). Verified in `audit_compute.py`.
- **KG-NAS per-category row** — matches `p2_rd/nas/ep500/<cat>/final_evaluation.json` (mean 82.4009). Only carpet is off (F7).
- **Fixed-architecture mean AUROC and wins** (Table `tab:fixed_kd`) — 84.0 / 82.4 / 82.0 / 80.2 and 8 / 3 / 2 / 2 reproduce exactly from `p2_rd/teacher_resnet18/fixed/*/ep500` (`audit_compute.py` §D). Only the *parameter* column fails (F3).
- **"98.2% of ResNet-18" and "2.4× fewer parameters"** — 82.4009/83.95 = 98.15→98.2 ✓; 12.9/5.3 = 2.43→2.4 ✓ (internally consistent with the paper's own param pair; see F3 for the param pair itself).
- **200-vs-500 budget** — 79.6% (`rd200` mean 79.56) vs 82.4% ✓; 200-ep winner counts {ResNet-18 10, NAS 2, MBV3 2, Swin 1} reproduce (body line 112) ✓. (Only the abstract's "12 of 15" is wrong — F1.)
- **Paradigm ablation** — prototype mean 72.19→72.2 ✓, RD 82.4 ✓, screw 90.4/44.0 ✓, "11 of 15" (RD wins 11) ✓, oracle 86.09→86.1 ✓, CFA gap 3.66→"within 3.7 points" ✓.
- **Statistical significance** — n=60; mean diff +5.17→+5.2 ✓; paired t=2.937→2.94 ✓; p=0.00472→0.005 ✓; 95% CI [1.65, 8.70]→[1.7, 8.7] ✓; 13/15 positive ✓; ResNet-18 avg improvement +12.94→+12.9 ✓; all 15 per-category "Avg Impv." values and the ResNet-18 "dramatic gains" set (tile 29.4, bottle 29.3, leather 27.9, carpet 23.7, transistor 23.9) reproduce exactly (`audit2.py` §F).
- **CI computation** — σ(sample) of the 15 KG-NAS AUROCs = 13.8901→13.9; 82.4 ± 2.145·13.9/√15 = [74.71, 90.09]→[74.7, 90.1] ✓. Endpoints correctly ordered (no interval-sign error).
- **Ablation counts** — 43 failure cases (AUROC<50) reproduce with breakdown {PatchCore 8, STPM 8, AE 7, SimpleNet 7, EffAD 6, VAE 5, RevDist 1, SPADE 1} ✓ and the three extreme cases (VAE/screw 3.73, STPM/zipper 12.84, AE/grid 21.14) ✓; `BASELINE_ANALYSIS_TABLE.tex` Failure/Std/Best columns all match `baseline_stability.csv`/`baseline_best_counts.csv` except the note below.
- **93 recorded trials / CNN-vs-Transformer** — 93 rows reproduce; after excluding divergent (>1000): CNN n=65 median 0.0639→0.064, std 3.840→3.84; TR n=20 median 0.0689→0.069, std 0.098→0.10; Mann-Whitney p=0.404→0.40 ✓. (Only "13 types" is wrong — F10.)
- **Feature-dimension ablation** — 7.5771/24.5738, 16.9529/66.5308, 2.2496/4.9099 and count 39 all reproduce (`ablation_3_feature_dim.csv`). (Only the 256-dim param "~5.3M" is off — F12.)
- **Teacher ablation** — ConvNeXt-Tiny best 0.2114 vs ResNet-18 0.3040, and the selected students, all reproduce (`ablation_phase2`/`phase2_summary.csv`; best_student labels verified). Note: teacher "params (M)=12/29/30" are the *teacher* backbone counts, not the student.
- **VisA** — average AUROC 70.0658→70.1, AP 74.23→74.2, all 12 rows reproduce (`visa_stage3/evaluation_summary.json`); modern 75% (9/12) and classical 25% (3/12) ✓; dim split 128→50%, 512→33%, 256→17% ✓; "7 backbones × 3 = 21" ✓ (7 distinct backbones, 100 trials each, in `visa_stage2/*/all_results.json`); "CV>100% removal of MobileNetV3-Small (155.5) and EfficientNet-B0 (113.2)" ✓ (`experiments/backbone_stability_analysis.json`).
- **VisA baselines** — PaDiM 0.76295→76.3 ✓; CFA 0.8397→84.0 ✓ (computed from `experiments/visa_baselines/*/{padim,cfa}/…`); "KG-NAS beats PaDiM on 4 of 12" (cashew, pcb2, pcb1, pcb3) ✓.
- **BTAD** — 0.7532/0.8983, 0.7338/0.9546, 0.7898/0.5468, average 0.758916→0.7589 / 0.799900→0.7999, all MobileNetV3-Small, dims 256/512/128 ✓; convergence epochs **34 / 98 / 8** reproduce exactly from `btad_stage3/*/training_log.json` (argmin of test_loss) ✓; 100 trials/product ✓.
- **Efficiency** — 102 min (13+30+59) ✓; "~1.7 h" ✓; 142 FPS student present in `inference_efficiency.csv` ✓; 87% (12→1.6 h) and 50% (20.7→10.3 h) ✓; 76%-fewer-params and 88%-fewer-FLOPs vs ResNet-50 ✓ (87.8%); VisA "~56% fewer (5.3 vs 12M)" ✓; "2.3× fewer than SimpleNet (12M)" ✓; "~16× fewer than Dinomaly (86M)" ✓; "2.4× faster than Dinomaly (59 FPS)" ✓.
- **Proxy-score spread** — pooled per-category medians (3 strategies × 30 trials): wood 7.7486→7.75, hazelnut 0.3847→0.38, capsule 0.1102→0.11 ✓ (`p1_search_strategies/*/*/results.json`).
- **Search strategies** — wood 6.288/6.168/6.935, hazelnut 0.323/0.327/0.336, capsule 0.095/0.092/0.072 all match `search_strategies.csv`/`strategy_summary*.json`; "+1.9% over random" on wood ((6.288−6.168)/6.288=1.9%) ✓; "TPE +31% on capsule" on the *best* score basis is directionally consistent.
- **Loss-function ablation** — means 62.49/63.05/61.11 and population stds 9.03/7.62/9.32 reproduce; best counts 6/6/3 reproduce once the single wood tie (kd+cons = full = 0.6781) is broken toward KD+Consistency ✓ (not a defect).
- **Baseline averages/ranking** — CFA 89.75, RevDist 80.49, SPADE 78.68, PaDiM 77.92, FastFlow 71.31, EffAD 60.07, AE 57.97, STPM 51.10, PatchCore 50.59, SimpleNet 50.52, VAE 47.57; gap "best−worst 42.18" (89.75−47.57) ✓.
- **AC (accuracy) is not used** as capability evidence anywhere; all detection metrics are AUROC/AP. Failure-mode 7 (majority-class masking) is therefore not applicable.
- **Dataset category counts** — MVTec-AD 15, VisA 12, BTAD 3 all confirmed by the evidence trees.

---

### Open items

- **[U1] "15 backbones" search space.** The manuscript consistently says 15 (and Table `tab:search_space` lists 15), but the evidence says 16: `D:/autopaper/unsupervised_nas/paper_results/efficiency_1_training_time.csv` ("**16 backbones x 3 dims**"), `ablation_1_search_space.csv` ("Full (**16**, Ours)"), and a run directory `stage2_topjournal_16backbones`. The 100-trial result actually covers only **13 distinct backbones** (`results_for_paper/origin_plot_data/backbone_performance_detail.csv`). Resolution needs the search-space definition (MobileNetV3-Small, EfficientNetV2-S, and one other are in the 15-table but absent from the 93-trial evidence). This also bears on F10 ("13 types") and F11 ("2.1").
- **[U2] BTAD "10 backbones × 3 dimensions".** No evidence file lists the BTAD backbone pool (`btad_stage2/*/*/nas_results.json` records only `num_trials: 100`). Cannot confirm 10 (or 30 configs).
- **[U3] "…vary by up to 81× within a category".** Not reproducible exactly. Depending on subset I get: main 93-trial search max within-config ratio **105.9** (MobileOne-S0/128); search-strategy repeats **up to 211.5** (capsule, Eff-B0/512) and as low as **1.5** (wood). Needs the specific computation the authors used.
- **[U4] "NAS stage requires only ~6× the computation of training a single architecture".** Untraceable: 100 trials × 3 ep = 300 epoch-equivalents vs 500-ep final training ⇒ ≈0.6×; vs one search trial ⇒ ≈100×; neither is 6×. Needs the intended reference quantity.
- **[U5] "100 NAS trials per category" on MVTec (data-availability, `main_springer.tex:116`).** VisA (100/category) and BTAD (100/product) are supported, but the MVTec search is a **single 93-trial aggregate** over 13 backbones with no category field (`stage2_topjournal_15backbones_fixed/search_results.json`). The per-category claim appears to overstate the MVTec protocol.
- **[U6] Cross-dataset dataset facts** (MVTec train 100–300 @700–1024 px; VisA 200–500 normal / 100–300 test @300–640 px; BTAD per-product normal/anomalous splits). No dataset-metadata file is present in the allowed evidence roots; not independently verifiable here (aside from the internal 399-vs-400 issue, F13).
- **[U7] Baseline FPS in `tab:efficiency_comparison`** (CFA ~150, RevDist ~120, PaDiM ~80, PatchCore ~50, STPM ~100) are labelled "literature-based estimates"; the only local artefact (`fig/efficiency_metrics.csv`) is stale (KG-NAS 320 FPS / 78 min) and disagrees with the reported 142 / 102. Not scored as a mismatch (explicitly labelled estimates), but the student 142 has no dedicated measurement script.

---

### Method-to-claim consistency spot-check (paraphrase, no independent numeric claim)

Stage-3 configuration differences that the paper treats as a single protocol should be noted for the next round, but were not scored here: `sections/3_method.tex:250` says "batch size 64", while the VisA efficiency discussion (`4_experiments.tex:28`, `supplementary.tex:240`) uses "batch size 32→128"; `btad_stage2/03/03/best_architecture.yaml` shows `batch_size: 32` while 01/02 show 64. Evidence: `btad_stage2/02/02/best_architecture.yaml` (64), `btad_stage2/03/03/best_architecture.yaml` (32).
