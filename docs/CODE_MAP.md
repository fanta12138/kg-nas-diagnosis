# Code map

Where to look for what, in the order a reader usually needs it.

Line numbers refer to the **released tree** and are the anchors for the statements
the paper makes about the code.

---

## 1. The pipeline under study — `unsupervised_nas/`

Three stages, matching Sections 3 and 4 of the paper.

```
unsupervised_nas/
├── scripts/
│   ├── stage1_train_teacher.py          # Stage 1: contrastive teacher pre-training
│   ├── stage2_nas_search.py             # Stage 2: the search itself
│   ├── stage3_final_training.py         # Stage 3: train the selected RD detector
│   ├── p2_rd_train.py                   # the training entry point the paper cites
│   ├── p1_search_strategies.py          # search-strategy variants
│   └── ... (30 scripts total, incl. per-dataset variants for VisA/BTAD)
├── methods/
│   ├── nas_engine.py                    # *** NASSearchEngine: the object of the diagnosis
│   └── rd_detector.py                   # *** RDDetector: the deployed detector
├── models/
│   ├── backbone.py                      # BackboneFactory: 17 supported backbones
│   └── contrastive_model.py             # student projector (takes `projection_dim`)
├── baselines/                           # 11 published detectors reimplemented
│   ├── patchcore.py, spade.py, padim.py, fastflow.py, cfa.py,
│   └── simplenet.py, efficientad.py, reverse_distillation.py, stpm.py, ...
├── datasets/                            # mvtec_loader.py, btad_loader.py
├── configs/                             # stage2_nas_search*.yaml — the 14-backbone space
├── experiments/                         # every recorded run (JSON/YAML only; no weights)
└── paper_results/                       # ablation CSVs and their figures
```

The **search space** is the 14 backbones listed in
`configs/stage2_nas_search.yaml`; `models/backbone.py` supports 17, three of which
(`efficientnetv2_s`, `mobileone_s3`, `mobileone_s4`) the search space does not use.
An earlier revision of the project searched 16 backbones (48 cells); the paper
reports the 14-backbone/42-cell space, which is what this configuration encodes.

### The two code facts the paper's diagnosis rests on

**`methods/rd_detector.py`, line 109** — the student encoder hard-codes the
bottleneck width:

```python
self._init_bottleneck(f.shape[1], 256, f.device)
```

while `RDDetector(student_backbone=..., teacher_arch=...)` constructs the decoder
with the *requested* width and the training entry point (`p2_rd_train.py`) never
passes `bottleneck_dim`. Every configuration therefore deploys a 256-wide
bottleneck, so the three searched values of `feature_dim` are
**deployment-equivalent**. This is the §4.1 claim, checkable by reading three
files.

**`methods/nas_engine.py`, lines 192 and 198** — the degenerate scoring path:

```python
if torch.isfinite(score) and score_val < 1000:   # line 192: a rejected batch is skipped
    ...
avg_score = total_score / max(count, 1)          # line 198: zero batches -> exactly 0.0
```

and the search returns `min(self.results, key=score)`. Since 0.0 is the minimum of
the score's own range, a trial that failed to produce a usable number is
indistinguishable from a trial that found the best possible number. This is the
§4.2 claim, and the mechanism behind §4.5.

> These behaviours are **the object of study, not bugs fixed here.** The release is
> frozen as it was used, so that the released code is the code the paper's numbers
> came from.

---

## 2. The diagnostic experiments

| Directory | Question it answers | Key entry point |
|---|---|---|
| `proxy_diag/` | Does the searched quantity reach the model? Is the proxy stable? Does the axis matter? How small a difference can the benchmark resolve? | `proxy_diag_v2.py --phase {A,B,C}`, `samplesize_analysis.py` |
| `search_forensics/` | What happened in the 19 recorded searches? | `analyze.py` |
| `search_repeats/` | What does the search return when re-run, against chance? | `search_repeats.py`, `authoritative_analysis.py` |
| `capacity_experiment/` | Is the gap to state of the art a parameter-budget effect? | `code/final_analysis.py` |

### `proxy_diag/`

| File | Role |
|---|---|
| `phaseA_proxy.json` | 42 configurations × 10 repeats = 420 calls to `evaluate_architecture` (Table 2, Figure 2) |
| `phaseB_truth.json` | the **210-detector truth matrix**: 14 backbones × 15 categories; 90 cells replicated over 3 seeds (Table 5, Figure 3) |
| `phaseC_analysis.json` | decomposition and correlation analyses of Phase B |
| `final_truth_analysis.json` | the aggregated truth summary used by the figures |
| `samplesize_analysis.py` | **the resolvability analysis** (§4.4): σ_seed, σ_res, σ_Δ, 240 runs, 2.98 pp, 532 categories |
| `train_faithful.py` | protocol-faithful re-training used to check the recorded numbers |
| `true/s42,s43,s44/` | per-seed truth matrices |

### `search_repeats/`

`search_repeats.json` holds the 12 runs; `search_repeats_summary.json` the summary;
`search_vs_random_final.json` the bootstrap comparison. `--analyse-only` re-derives
everything from the shipped run records without touching a GPU.

### `capacity_experiment/`

`code/` (9 scripts: budget-band configs, tier selection, training workers,
analysis), `configs/`, `results/budget_{3,6,8,13,30}M/`, `logs/`.

---

## 3. The scoring-policy axis

| Directory | Role |
|---|---|
| `fusion_policy/` | 82 policies on one frozen detector per category → Table 6 |
| `fusion_ksweep/` | top-*k* fraction sweep → Table 7, Figure 5 |
| `fusion_multiseed/` | seeds 43 and 44 → the three-seed aggregate (82.36 → 90.28) |
| `policy_select/` | the distributional criteria that **fail**, on the same models |
| `paradigm_select/` | paradigm-level comparison (supporting, earlier stage) |
| `pseudo_select/` | the label-free procedure and its three-seed aggregate → Table 8, Figure 6 |

`pseudo_select/pseudo_select.py` trains a student on 70% of `train/good`, perturbs
the held-out 30% with three families of synthetic anomaly (smoothed blobs, thin
oriented scratches, rectangular cuts; five seeds each = 15 variants), and picks the
policy that maximises the AUC between perturbed and clean. It never touches a real
defect. `aggregate_multiseed.py` pools seeds 42/43/44.

---

## 4. The synthetic reproduction

`synthetic_mechanism/synthetic_mechanism.py` rebuilds the
degenerate-objective × `argmin` interaction **without any part of the pipeline**:
a population of candidate objectives, a subset that can return the minimum of
their own range, and a minimising selection rule. It reports what fraction of the
search's advantage over random survives — 95% of it is lost, and filtering
degenerate evaluations restores it.

---

## 5. The auxiliary benchmark harness — `adbench/`

| File | Role |
|---|---|
| `common.py` | data loading (MVTec-AD, VisA, MVTec-LOCO), `Backbone` (CNN + ViT), all metrics |
| `methods.py` | `PatchCore` with a locality radius; `KNNGlobal` |
| `run_matrix.py` | the matrix driver (`--dataset`, `--backbones`, `--radii`, `--seeds`, ...) |
| `analysis_*.py` | power, cross-architecture, cross-dataset, three-dataset, per-category-gain |
| `results/` | all matrices, per dataset and configuration |
| `audit_citations.py`, `numsnap.py`, `trim_bib.py`, `abstract_stats.py` | the release tooling (see [`../REPRODUCE.md`](../REPRODUCE.md)) |

Metrics implemented: image AUROC, pixel AUROC, PRO, **sPRO** (`pro` with
`fpr_limit=0.05`), and the deployment-faithful **bound-FPR** pair R@1%FPR and
AP@1%FPR.

**This harness is not reported in the paper.** Its MVTec-AD/VisA/LOCO matrices were
used to cross-check that "category dominates configuration" holds outside MVTec-AD,
and one of its measurements (the 0.51-point seed noise of a PatchCore-style
baseline) supplies the method-specific counter-example quoted in §4.4.

---

## 6. The audit trail — `audit/`

`claim-audit_20261008/` is the working trace of the manuscript's claim audit:
per-claim check definitions (`checks*.json`), the verification scripts, the
reviewer reports (`reviewer_report.md`, `reviewer_prompt.md`), and the derived
artefacts. `PAPER_CLAIM_AUDIT.json` is the machine-readable ledger with input
hashes.

Some of its scripts read data belonging to an **earlier revision** of the
manuscript that is not shipped (`experiments/`, `fig/`). They are archived for
provenance — how the claims were checked — not as part of the reproduction path.

---

## 7. Paths

Every script resolves its roots through [`../kgd_paths.py`](../kgd_paths.py). The
original absolute paths are gone; `patch_paths.py → REPO_ROOT/DATASETS_ROOT/
PAPER_ROOT` is the only indirection. Run `python kgd_paths.py` to print what
resolves where.
