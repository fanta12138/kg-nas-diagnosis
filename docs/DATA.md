# Datasets

Datasets are **not** redistributed here — all of them are third-party and carry
their own licences. Download them yourself and place them as shown below.

Set `KGD_DATASETS_ROOT` if you keep them outside the repository:

```bash
export KGD_DATASETS_ROOT=/data/benchmarks     # Windows: set KGD_DATASETS_ROOT=D:\benchmarks
```

## Expected layout

```
$KGD_DATASETS_ROOT/
├── datasets/
│   └── mvtec_anomaly_detection/          # MVTec-AD   — used by the paper
│       ├── bottle/{train,test,ground_truth}/...
│       ├── cable/...
│       └── ... (15 categories)
├── datasets-2/                           # VisA       — used by adbench/
│   ├── 1cls.csv                          # official split
│   └── visa/...
├── datasets-3/                           # BTAD       — used by the BTAD scripts only
│   └── BTech_Dataset_transformed/...
└── mvtec_loco_anomaly_detection/         # MVTec-LOCO — used by adbench/
    └── ...
```

If `KGD_DATASETS_ROOT` is not set it defaults to the repository root, so a fresh
clone expects exactly the four directories above at the top level. They are
`gitignore`d.

## What each dataset is used for

| Dataset | Categories | Used by | Notes |
|---|---|---|---|
| **MVTec-AD** | 15 | The entire paper: the search proxy, the 210-detector truth matrix, the policy sweeps, the label-free selection | The one dataset the paper's claims rest on. |
| **VisA** | 12 | `adbench/` only | Auxiliary benchmark harness. **Not** reported in the paper. |
| **MVTec-LOCO** | 5 | `adbench/` only | Auxiliary benchmark harness; used to cross-check that "category dominates configuration" holds outside MVTec-AD. **Not** reported in the paper. |
| **BTAD** | 3 | `unsupervised_nas/scripts/*_btad_*.py` only | Left from an earlier stage of the project; **not** used in the paper. |

## Sources and licences

**MVTec-AD** — Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for
Unsupervised Anomaly Detection*, CVPR 2019.
<https://www.mvtec.com/company/research/datasets/mvtec-ad>
Licence: CC BY-NC-SA 4.0. The paper cites the IJCV version (DOI
`10.1007/s11263-020-01400-4`).

**VisA** — Zou et al., *SPot-the-Difference Self-Supervised Pre-training for
Anomaly Detection and Segmentation*, ECCV 2022.
<https://github.com/amazon-science/spot-diff>
Licence: CC BY-NC-SA 4.0. The harness reads the official `1cls.csv` split.

**MVTec-LOCO** — *Logical Anomaly Detection* extension of MVTec-AD.
<https://www.mvtec.com/company/research/datasets/mvtec-loco>
Licence: CC BY-NC-SA 4.0.
Note: LOCO stores one **directory** of per-instance PNGs per anomalous test image.
`adbench/common.py` (`_load_mask_dir`) takes their union.

**BTAD** — Mishra et al., *VT-ADL: A Vision Transformer Network for Image Anomaly
Detection and Localization*, ISIE 2021.

## Splits

All studies in the paper use the **official MVTec-AD split**: `train/good` for the
normal-only training set and the shipped `test/` for evaluation. Two studies
additionally carve a hold-out out of `train/good`:

| Study | Split of the normal training set |
|---|---|
| Everything except §5.4 | full `train/good` |
| Label-free selection (§5.4, Table 8) | 70% fit / 30% hold-out for selection |

This is why the default policy reads 82.80% in Table 6 and 81.62% in Table 8 on the
same seed: the second set of students saw 70% of the normal training images. The
gain in Table 8 is a within-model comparison, so it is not inflated by the split;
absolute values should be read with the split in mind.

## Sanity check

Verify the layout without loading any framework:

```bash
python -c "
import os, kgd_paths as kp
root = os.path.join(kp.DATASETS_ROOT, 'datasets', 'mvtec_anomaly_detection')
cats = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
print(len(cats), 'categories:', cats)
print('all train/good present:',
      all(os.path.isdir(os.path.join(root, c, 'train', 'good')) for c in cats))
print('all test present:',
      all(os.path.isdir(os.path.join(root, c, 'test')) for c in cats))
"
# expect: 15 categories: ['bottle', 'cable', 'capsule', ... 'zipper']
```

The pipeline's own loaders are `unsupervised_nas/datasets/mvtec_loader.py`
(`MVTecWoodDataset` for the search category, `MVTecTestDataset` for evaluation)
and `unsupervised_nas/datasets/btad_loader.py` (`BTADDataset`).

The feature-only harness in `adbench/` needs no GPU and is the quickest way to
confirm the data paths end to end:

```bash
python adbench/run_matrix.py --dataset mvtec --backbones resnet50.a1_in1k \
                             --radii None --seeds 0 --categories bottle
```
