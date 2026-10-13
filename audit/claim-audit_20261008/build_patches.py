"""Build the patch spec for the audit fixes (round 1).

Strings are authored here in Python so the LaTeX escapes are readable and
written to JSON only once.
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

import json, os

OUT = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')

T46_CAP_OLD = (r"\caption{Comparison with state-of-the-art methods on MVTec-AD dataset (15 categories). "
               r"AUROC (\%) is reported. Best results are highlighted in \textbf{bold}, second best are "
               r"\underline{underlined}. KG-NAS (82.4\%) exceeds PaDiM (78.0\%) and Reverse Distillation (80.5\%) "
               r"while remaining below CFA (89.8\%), using only 5.3M parameters (student encoder$+$decoder).}")
T46_CAP_NEW = (r"\caption{Comparison with state-of-the-art methods on MVTec-AD dataset (15 categories). "
               r"AUROC (\%) is reported. Best results are highlighted in \textbf{bold}, second best are "
               r"\underline{underlined}. KG-NAS (82.4\%) exceeds PaDiM (77.9\%) and Reverse Distillation (80.5\%) "
               r"while remaining below CFA (89.8\%), using only 5.3M parameters (student encoder$+$decoder).}")

T71_OLD = (r"\textbf{Average} & 58.0 & 47.6 & 78.0 & 50.6 & 51.1 & 80.5 & \textbf{89.8} & 78.7 & 71.3 & 60.1 & 50.5 & \underline{82.4} \\")
T71_NEW = (r"\textbf{Average} & 58.0 & 47.6 & 77.9 & 50.6 & 51.1 & 80.5 & \textbf{89.8} & 78.7 & 71.3 & 60.1 & 50.5 & \underline{82.4} \\")

T78_OLD = r"above PaDiM (78.0\%) and Reverse Distillation (80.5\%) and approaching CFA (89.8\%)"
T78_NEW = r"above PaDiM (77.9\%) and Reverse Distillation (80.5\%) and approaching CFA (89.8\%)"

# F9 wording, two places
T78B_OLD = r"taking best place in 9 of 15 categories (60\%; strictly 8, tying with itself on bottle and leather within 0.02\,pp)"
T78B_NEW = r"taking best place in 9 of 15 categories (60\%; strictly 8, sharing the ninth with Reverse Distillation on bottle and holding leather by 0.03\,pp)"

T165_OLD = r"CFA achieves best performance in 9 out of 15 categories (60\%; strictly 8, with two ties within 0.02\,pp), while other methods excel"
T165_NEW = r"CFA achieves best performance in 9 out of 15 categories (60\%; strictly 8, sharing the ninth with Reverse Distillation on bottle and holding leather by 0.03\,pp), while other methods excel"

SUP131_OLD = (r"\caption{Best method distribution across MVTec-AD categories. CFA dominates (best on 9/15 categories, "
              r"including two ties within 0.02 pp), but 5 other methods achieve best results in at least one category, "
              r"demonstrating method complementarity.}")
SUP131_NEW = (r"\caption{Best method distribution across MVTec-AD categories. CFA dominates (best on 9/15 categories: "
              r"8 outright plus a shared best with Reverse Distillation on bottle), but 5 other methods achieve best "
              r"results in at least one category, demonstrating method complementarity.}")

EFF_FOOT_OLD = (r"\textbf{KG-NAS (Ours)} & 82.40 & \textbf{5.3} & 142$^{\dagger}$ & 102 & \textbf{Resource-constrained} \\" "\n"
                r"\bottomrule" "\n"
                r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$Student throughput, measured on an idle RTX 3090 (batch size 1, 224$\times$224, fp32); the full teacher--student detector runs at $\sim$107 FPS.} \\")
EFF_FOOT_NEW = (r"\textbf{KG-NAS (Ours)} & 82.40 & \textbf{5.3} & 142$^{\dagger}$ & 102 & \textbf{Resource-constrained} \\" "\n"
                r"\bottomrule" "\n"
                r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$Student throughput, measured on an idle RTX 3090 (batch size 1, 224$\times$224, fp32).} \\")

SOTA_FOOT_OLD = (r"\textbf{KG-NAS (Ours)} & 82.4 & - & \textbf{6} & 142$^{\dagger}$ & Single-class + NAS \\" "\n"
                 r"\bottomrule" "\n"
                 r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$Student throughput, measured on an idle RTX 3090 (batch size 1, 224$\times$224, fp32); the full teacher--student detector runs at $\sim$107 FPS.} \\")
SOTA_FOOT_NEW = (r"\textbf{KG-NAS (Ours)} & 82.4 & - & \textbf{5.3} & 142$^{\dagger}$ & Single-class + NAS \\" "\n"
                 r"\bottomrule" "\n"
                 r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$Student throughput, measured on an idle RTX 3090 (batch size 1, 224$\times$224, fp32).} \\")

patch = {
    "tag": "round01_claim_fixes",
    "paper_dir": _KGD_PAPER,
    "generated_from": ".aris/traces/paper-claim-audit/2026-10-08_run01/reviewer_report.md",
    "patches": [
        # ---- F1: abstract flips the 500-epoch winner direction ----
        {"file": "main_springer.tex", "id": "F1",
         "find": r"whereas at 500 epochs the discovered architecture closes the gap and wins 12 of 15.",
         "replace": r"whereas at 500 epochs ResNet-18 still leads on 8 of 15 and the discovered architecture wins 3, the two separated by 1.6 points of average AUROC.",
         "note": "F1: verified 500-ep winners = resnet18 8 / nas 3 (p2_rd/*/ep500). Abstract said 12/15."},

        # ---- F2: abstract claims to exceed ResNet-18 ----
        {"file": "main_springer.tex", "id": "F2a",
         "find": r"attains 82.4\% average AUROC across 15 categories, exceeding a 12.9M ResNet-18 student under an identical protocol while using 2.4$\times$ fewer parameters.",
         "replace": r"attains 82.4\% average AUROC across 15 categories, matching 98.2\% of the average AUROC of a 12.9M ResNet-18 student under an identical protocol while using 2.4$\times$ fewer parameters.",
         "note": "F2: ResNet-18 fixed = 83.95% > discovered 82.40%; ratio 82.4009/83.95 = 98.2%."},

        # ---- F2 in the introduction ----
        {"file": "sections/1_introduction.tex", "id": "F2b",
         "find": r"attains 82.4\% AUROC on MVTec-AD, exceeding a 12.9M ResNet-18 student trained under an identical protocol at less than half the parameter count.",
         "replace": r"attains 82.4\% AUROC on MVTec-AD, matching 98.2\% of the average AUROC of a 12.9M ResNet-18 student trained under an identical protocol at less than half the parameter count.",
         "note": "F2: same direction error as the abstract."},

        # ---- F6: PaDiM average should be 77.9, not 78.0 ----
        {"file": "sections/4_experiments.tex", "id": "F6a",
         "find": T46_CAP_OLD, "replace": T46_CAP_NEW,
         "note": "F6: PaDiM mean = 77.9182 -> 77.9 under both direct and two-step rounding."},
        {"file": "sections/4_experiments.tex", "id": "F6b",
         "find": T71_OLD, "replace": T71_NEW,
         "note": "F6: main-table Average row."},
        {"file": "sections/4_experiments.tex", "id": "F6c",
         "find": T78_OLD, "replace": T78_NEW,
         "note": "F6: key-observations sentence."},

        # ---- F9: tie wording ----
        {"file": "sections/4_experiments.tex", "id": "F9a",
         "find": T78B_OLD, "replace": T78B_NEW,
         "note": "F9: only ONE tie involves CFA (bottle, exact); leather margin is 0.034 pp, not within 0.02."},
        {"file": "sections/4_experiments.tex", "id": "F9b",
         "find": T165_OLD, "replace": T165_NEW,
         "note": "F9: same wording in the baseline-distribution paragraph."},
        {"file": "supplementary/supplementary.tex", "id": "F9c",
         "find": SUP131_OLD, "replace": SUP131_NEW,
         "note": "F9: supplementary figure caption carried the same inaccurate clause."},

        # ---- F10: CNN type count ----
        {"file": "sections/4_experiments.tex", "id": "F10",
         "find": r"CNNs (13 types, 65 trials) yield a median score of 0.064 with Std 3.84, while Transformers (3 types, 20 trials)",
         "replace": r"CNNs (10 types, 65 trials) yield a median score of 0.064 with Std 3.84, while Transformers (3 types, 20 trials)",
         "note": "F10: backbone_distribution.csv covers 13 backbones total = 10 CNN + 3 Transformer."},

        # ---- F11: trials per configuration ----
        {"file": "sections/3_method.tex", "id": "F11",
         "find": r"NAS trials per config & $\sim$2.1 (100 total) \\",
         "replace": r"NAS trials per config & $\sim$2.2 (100 total) \\",
         "note": "F11: 100/45 = 2.22; body text already says 2.2."},

        # ---- F13: BTAD training-sample lower bound ----
        {"file": "sections/4_experiments.tex", "id": "F13a",
         "find": r"Train samples & 100--300 & 200--500 & 400--1000 \\",
         "replace": r"Train samples & 100--300 & 200--500 & 399--1000 \\",
         "note": "F13: body states Product 02 has 399 training samples."},
        {"file": "sections/4_experiments.tex", "id": "F13b",
         "find": r"and 400--1000 (BTAD) training samples",
         "replace": r"and 399--1000 (BTAD) training samples",
         "note": "F13: consistency with the table and the body."},

        # ---- F4: 0.0023 is scoped to the feature-dimension subset ----
        {"file": "sections/4_experiments.tex", "id": "F4",
         "find": r"Note that the NAS-discovered configuration (EfficientNet-B0+256d) attained the single best trial score (0.0023), so the final choice reflects per-trial evidence rather than the dimension-level average.",
         "replace": r"Within this subset the 256-dim configuration holds the single lowest-scoring trial (0.0023, against 0.0032 for 128-dim and 0.0166 for 512-dim), so the final choice reflects per-trial evidence rather than the dimension-level average.",
         "note": "F4: 0.0023 is the 256-dim minimum of the 39-trial subset, not the 93-trial NAS optimum (0.0097, Ablation 3)."},

        # ---- F5 + M1: sota table params and the untraceable 107 FPS ----
        {"file": "sections/4_experiments.tex", "id": "F5+M1a",
         "find": SOTA_FOOT_OLD, "replace": SOTA_FOOT_NEW,
         "note": "F5: 6 -> 5.3 to match the measured 5.33M and the rest of the paper. M1: 107 FPS has no artefact."},
        {"file": "sections/4_experiments.tex", "id": "M1b",
         "find": EFF_FOOT_OLD, "replace": EFF_FOOT_NEW,
         "note": "M1: no measurement of the full-pipeline throughput exists in the evidence tree."},
        {"file": "sections/5_conclusion.tex", "id": "M1c",
         "find": r"$\sim$142 FPS student throughput and $\sim$107 FPS for the full detection pipeline",
         "replace": r"$\sim$142 FPS student throughput",
         "note": "M1: same untraceable claim in the conclusion."},
    ],
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(patch, f, ensure_ascii=False, indent=2)
print('wrote %s with %d patches' % (OUT, len(patch['patches'])))
for p in patch['patches']:
    print('   %-8s %s' % (p['id'], p['file']))
