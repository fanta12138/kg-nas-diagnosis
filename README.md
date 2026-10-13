# The Architecture Axis Is Not the Axis

**Code, frozen results and audit trail for the paper**

> *The Architecture Axis Is Not the Axis: A Diagnostic Study of Neural Architecture
> Search and Score Fusion in Unsupervised Industrial Anomaly Detection*
> Peng Deng, Hong Li, Yuanwu Zhou — Guangxi University of Science and Technology

This repository accompanies a **benchmark and evaluation study**. It is not a new
detector. It contains the pipeline that was diagnosed, every experiment that
produced a number in the paper, the frozen outputs those experiments produced, and
the scripts that regenerate every table and figure.

If you are here for one number, it is this: **on MVTec-AD the standard
single-category comparison resolves 2.98 AUROC points, while the margin the field
contests between successive state-of-the-art generations is 0.5 points.** Most
reported backbone comparisons on this benchmark are therefore not measurements.

---

## Contents

| Path | What it is |
|---|---|
| `unsupervised_nas/` | **The pipeline under study** (KG-NAS). Three stages: teacher pre-training, knowledge-distillation architecture search, reverse-distillation detection. 11 reimplemented baseline detectors, 30 scripts, and the search space configured in `configs/stage2_nas_search.yaml` (14 backbones × 3 feature dimensions = 42 cells; the backbone factory supports 17, three of which the search space does not use). Shipment includes every search record and result JSON; model checkpoints are omitted (13 GB). |
| `proxy_diag/` | The search proxy, instrumented. Repeated evaluation of all 42 configurations (420 calls), the 14 × 15 **210-detector truth matrix**, and the **resolvability analysis** (sample sizes and minimum detectable differences). |
| `search_forensics/` | Forensics over the 19 recorded searches (834 trials): where degenerate zero-score evaluations occur and whether the `argmin` lands on one. |
| `search_repeats/` | The search re-run with 12 independent seeds, against uniform-random selection. |
| `capacity_experiment/` | Five parameter-capacity bands, to test whether the gap to state of the art is a parameter-budget effect. |
| `fusion_policy/`, `fusion_ksweep/`, `fusion_multiseed/` | The **scoring-policy** axis: 82 policies on frozen models, the top-*k* sensitivity sweep, and the three-seed replication. |
| `policy_select/`, `paradigm_select/` | Criteria for choosing a policy without labels (the distributional attempt that fails, and the paradigm comparison). |
| `pseudo_select/` | The **label-free selection** procedure and its three-seed aggregate. |
| `synthetic_mechanism/` | A code-independent synthetic reproduction of the degenerate-objective × `argmin` mechanism. |
| `adbench/` | An auxiliary three-dataset benchmark harness (MVTec-AD, VisA, MVTec-LOCO) plus the release tooling: citation audit, number-diff, bib trimming. |
| `anomalib_baselines/` | External-baseline runs through the `anomalib` library. |
| `audit/` | The claim-audit trace: per-claim checks, reviewer reports, and the scripts that verified them. |
| `logs/` | Raw run logs, in chronological order. |
| `paper/` | Manuscript source (`main_v2.tex`), bibliography, figure generator output, and the compiled PDF. |
| `docs/` | Data setup, method notes, and the original Chinese working notes under `docs/notes_zh/`. |
| `kgd_paths.py` | Filesystem-root resolver. **Read this first if you intend to run anything.** |

---

## What the study found

Four measurements, all reproducible from this repository.

**1. The benchmark cannot resolve the differences it is used to rank.**
Two-way decomposition of the 210-detector matrix leaves a residual standard
deviation of 3.30 points; the run-to-run standard deviation of one detector at a
fixed (backbone, category) is 1.96 points over 180 residual degrees of freedom.
Comparing two backbones across the 15 categories is a paired test whose minimum
detectable difference is **2.98 points** — larger than the entire 3.98-point spread
of the 14 backbones, and six times the 0.5-point margin the field argues about. To
resolve 0.5 points on a single category takes **240 independent training runs per
backbone**; to resolve it by pairing over categories would take 532 categories.

**2. Two thirds of the search space cannot reach the deployed model.**
The searched `feature_dim` enters the search through the student projector but
never reaches the detector: `StudentEncoder.forward` hard-codes the bottleneck
width (`unsupervised_nas/methods/rd_detector.py`, line 109) while the decoder
honours the requested one, and the training entry point never passes the argument.
All three values of the searched dimension therefore give the *identical* deployed
model, so 42 configurations contain only 14 distinct detectors.

**3. The search signal does not discriminate, and collapses for two backbones.**
Across the 12 well-behaved backbones the median proxy spans 5.7% of itself — the
same order as its own run-to-run noise. Two backbones return **exactly 0.0** in 73%
and 13% of evaluations, because the student diverges, every validation batch trips
a hard threshold, and the score is then silently computed as an average over zero
samples (`nas_engine.py`, line 198: `total_score / max(count, 1)`). Since the search
minimises, 0.0 is the optimum of the search's own range. Running the search 12 times,
**every run returned one of those two backbones**, and the detectors they produce are
**worse than uniform random selection** (80.91% against 81.65%, at the 1.2th
percentile; the best backbone reaches 83.34%).

**4. The leverage is in the reduction nobody documents.**
Replacing the unweighted mean over the discrepancy map with top-*k* pooling raises
mean AUROC from 82.36% to 90.28% across three training seeds — **+7.92 points,
without retraining anything** — and the policy can be chosen with no anomaly label
at all, by separating synthetic pseudo-anomalies from clean images, recovering
**61% of the oracle gap** (7.09 of 11.65 points, three seeds).

We do **not** claim state of the art: the pipeline reaches 82.47% mean AUROC on
MVTec-AD against published values of 95.6–99.6%. That gap is reported in the paper
rather than obscured, and the code that produces it is here.

---

## Install

```bash
git clone <this-repo> && cd kg-nas-diagnosis

# Python 3.9+ (developed on 3.9.13 / torch 2.5.1+cu121)
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists the full environment. A GPU is needed only for training
the detectors; every analysis script in this repository runs on CPU.

## Data

The datasets are **not** included — download them yourself (see
[`docs/DATA.md`](docs/DATA.md)). By default they are expected under the repository
root; point `KGD_DATASETS_ROOT` elsewhere if you keep them on another drive:

```bash
# expected layout
$KGD_DATASETS_ROOT/datasets/mvtec_anomaly_detection/...     # MVTec-AD  (paper)
$KGD_DATASETS_ROOT/datasets-2/...                           # VisA      (adbench)
$KGD_DATASETS_ROOT/mvtec_loco_anomaly_detection/...         # MVTec-LOCO (adbench)
```

## Run

**Nothing needs to be trained to reproduce the paper's numbers.** Every figure and
every statistic is computed from frozen JSONs that ship with the repository, by
scripts that need no GPU and no dataset:

```bash
python proxy_diag/samplesize_analysis.py      # resolvability: 1.96 / 3.30 / 4.12, 240 runs, 2.98 pp
python pseudo_select/aggregate_multiseed.py   # label-free selection over three seeds: +7.09, 61%
python search_repeats/authoritative_analysis.py  # 12 runs vs chance: 80.91 / 81.65 / 83.34
python search_forensics/analyze.py            # 19 runs, 834 trials, 8.0% degenerate, 14/19 argmin
python synthetic_mechanism/synthetic_mechanism.py # 95% of the search advantage lost
python make_figures_v2.py                     # regenerate all five figures
```

`make_figures_v2.py` reproduces `paper/figures_v2/*.png` **byte for byte** from the
shipped JSONs. [`REPRODUCE.md`](REPRODUCE.md) maps every table, figure and quoted
number in the paper to the command that produces it.

## Paths

The scripts were written while the project lived at fixed absolute paths. Those
roots are now redirected through [`kgd_paths.py`](kgd_paths.py), which resolves the
repository root by walking up from each file, so everything runs from any location
and any drive:

| Original root | Now resolves to | Override |
|---|---|---|
| `D:/autopaper` | repository root | `KGD_WORK_ROOT` |
| `D:/autopaper/datasets*` | `<repo>/datasets*` | `KGD_DATASETS_ROOT` |
| `G:/submission_KBS` | `<repo>/paper` | `KGD_PAPER_ROOT` |

```bash
python kgd_paths.py    # print the resolved roots
```

## Verifying the release

`MANIFEST.sha256` lists the SHA-256 of every file in the release, so you can check
that your copy is the one the paper's numbers came from:

```bash
sha256sum -c MANIFEST.sha256        # macOS: shasum -a 256 -c MANIFEST.sha256
```

Two strong checks that the release is intact and self-consistent:

1. `python make_figures_v2.py` regenerates `paper/figures_v2/*.png`
   **byte for byte** from the shipped JSONs.
2. The manuscript compiles standalone: `cd paper && latexmk -pdf main_v2.tex`
   → 22 pages, 0 undefined references.

## Scope and caveats

* **Checkpoints are not shipped.** `unsupervised_nas/experiments/` originally held
  13 GB of `.pth` weights; only the JSON/YAML/CSV/log records are kept. Re-running
  the training studies requires a GPU and the datasets.
* **The scripts are research code, frozen as it was used.** They are preserved
  rather than refactored so that the released code is the code the paper's numbers
  came from. Some behaviours are deliberately untouched — the degenerate scoring
  path in `nas_engine.py` is the object of study, not a bug to be fixed here.
* **`audit/claim-audit_20261008/` is an archive**, kept for provenance. A few of
  its scripts read data from an earlier revision of the manuscript that is not
  shipped; they document how claims were checked, not how to re-run the pipeline.
* **Measured numbers are host- and version-specific.** The seed noise figures were
  obtained on one RTX 3090 with the versions pinned in `requirements.txt`.
* The manuscript's own internal inconsistencies are documented, not hidden, in
  `docs/notes_zh/11_声明与引用审计_zh.md` (§C) — including a 0.13-point
  disagreement between two evaluation scripts, which is reported in the paper as
  evidence about the measurement floor.

## Citing

See [`CITATION.cff`](CITATION.cff). The funding acknowledged in the paper is the
Guangxi Natural Science Foundation (Grant No. `2025GXNSFHA069076`) and the Guangxi
University of Science and Technology Doctoral Fund (Grant Nos. `26Z15`, `26Z25`).

## Licence

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Peng Deng, Hong Li, Yuanwu Zhou.

## Contact

Corresponding author: **Peng Deng** — dengpeng@gxust.edu.cn
School of Civil and Architectural Engineering, Guangxi University of Science and
Technology, Liuzhou 545006, China
