# -*- coding: utf-8 -*-
"""Round 09 — narrative revision driven by the capacity experiment.

Findings that force the revision:
  * capacity 2.81M -> 25.77M changes MVTec-AD AUROC by only 1.70 pp, and all ten
    pairwise comparisons are non-significant (paired t over 15 categories,
    smallest p = 0.058).  See docs/参数量扫描实验报告.md
  * at matched capacity the framework trails EfficientAD (8.0M -> 97.50%) by
    ~16 pp and CFA (12M -> 89.75%) by ~6.5 pp, so the gap is methodological.

The paper must therefore stop selling "efficiency" as a differentiator and
present automation + an honest, measured capacity envelope instead.
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

import json

OUT = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')

# ---------------------------------------------------------------- new section
CAPACITY = r"""\subsection{Parameter-Capacity Analysis}
\label{subsec:capacity}

The discovered student is deliberately small. A natural question is whether the remaining gap to recent detectors (Table~\ref{tab:sota_comparison}) is a consequence of that budget: would KG-NAS close it if it were allowed more parameters? We answer this by running the complete pipeline under five mutually exclusive parameter bands, each trained with the identical protocol (500 epochs, batch size 64, seed 42, ResNet-18 teacher, reverse distillation). For each band the search is restricted to backbones whose deployed size (encoder$+$decoder) falls inside the band, and the best-scoring backbone is trained on all 15 MVTec-AD categories.

\begin{table}[t]
\centering
\caption{Parameter-capacity study. Each band constrains the search to backbones of a given deployed size; the selected backbone is then trained on all 15 MVTec-AD categories under the standard protocol. Std is the per-category standard deviation.}
\label{tab:capacity}
\footnotesize
\begin{tabular}{llccc}
\toprule
Band & Backbone & Params (M) & Ave AUROC (\%) & Std \\
\midrule
0--3M   & MobileNetV3-Small & 2.81  & 82.01 & 15.97 \\
3--6M   & EfficientNet-B0   & 5.41  & 83.14 & 14.49 \\
6--9M   & DeiT-Tiny         & 7.31  & 81.55 & 15.40 \\
9--15M  & ResNet-18         & 13.04 & \textbf{83.25} & 14.71 \\
15--31M & ResNet-50         & 25.77 & 83.08 & 15.05 \\
\bottomrule
\end{tabular}
\end{table}

Two observations follow. First, \emph{capacity is not the binding constraint.} Across a $9.2\times$ range in parameter count the average AUROC varies by only 1.7 points (81.55--83.25\%), and none of the ten pairwise comparisons is significant (paired $t$-test over the 15 categories, smallest $p=0.058$); the trend is not even monotone, as the 25.77M ResNet-50 is no better than the 2.81M MobileNetV3-Small. Second, the gap to modern detectors is therefore \emph{methodological} rather than budgetary: EfficientAD reaches 97.5\% with 8.0M parameters while the comparably sized DeiT-Tiny band attains 81.55\%, and CFA reaches 89.75\% at 12M against 83.25\% for the 13.04M band.

This result retroactively justifies the search's preference for small models (Section~\ref{subsubsec:ablation_search_space}): on MVTec-AD's limited training data, additional capacity buys no measurable accuracy, so preferring EfficientNet-B0 to a 28M Swin-Tiny is not a failure of the search but a correct response to the data. It also bounds what the present framework can claim---improving on the models above will require a different detection or search objective, not a larger budget.

"""

# ------------------------------------------------------------------- abstract
ABS_OLD = (r"A paradigm study shows reverse distillation and prototype-based scoring are "
           r"complementary; combining them per category raises the average to 86.1\% AUROC, "
           r"within four points of the strongest baseline. Code and search logs will be released "
           r"to support deployment on new production lines.")
ABS_NEW = (r"A paradigm study shows reverse distillation and prototype-based scoring are "
           r"complementary; combining them per category raises the average to 86.1\% AUROC, "
           r"within four points of the strongest baseline. The discovered architectures exceed "
           r"the classical embedding-based and teacher--student baselines on both accuracy and "
           r"size (PaDiM 77.9\%, Reverse Distillation 80.5\%, PatchCore 50.6\%, STPM 51.1\%), but "
           r"trail recent high-capacity detectors (CFA 89.8\%, EfficientAD 97.5\%). A capacity "
           r"study spanning 2.8M to 25.8M parameters finds no statistically significant change in "
           r"accuracy, indicating that this remaining gap is methodological rather than an effect "
           r"of the parameter budget. Code and search logs will be released to support deployment "
           r"on new production lines.")

# ------------------------------------------------------------- intro bullet 4
INTRO_OLD = (r"    \item \textbf{Efficiency-Focused Design}: Unlike existing methods that prioritize "
             r"absolute accuracy at the cost of computational resources, KG-NAS adopts an "
             r"efficiency-first design philosophy targeting resource-constrained deployment scenarios. "
             r"The discovered 5.3M-parameter student attains 82.4\% AUROC on MVTec-AD, matching 98.2\% "
             r"of the average AUROC of a 12.9M ResNet-18 student trained under an identical protocol "
             r"at less than half the parameter count. We further characterize how this comparison "
             r"depends on the training budget (Section~\ref{subsec:main_results}). Cross-dataset "
             r"evaluation across three datasets (MVTec-AD, VisA, BTAD) reveals dataset-dependent "
             r"architectural preferences: CNNs dominate for texture-based defects (MVTec-AD, BTAD), "
             r"while Transformers excel for complex structures (VisA 75\%). This automatic adaptation "
             r"eliminates manual architecture selection, enabling rapid deployment across diverse "
             r"industrial scenarios without requiring domain expertise.")
INTRO_NEW = (r"    \item \textbf{Deployment-Oriented Discovery with a Measured Capacity Envelope}: "
             r"KG-NAS targets deployment scenarios in which architecture engineering is the bottleneck "
             r"rather than absolute accuracy. Without any manual architecture design the discovered "
             r"5.3M-parameter student attains 82.4\% AUROC on MVTec-AD, exceeding the classical "
             r"embedding-based and teacher--student baselines (PaDiM 77.9\%, Reverse Distillation 80.5\%, "
             r"PatchCore 50.6\%, STPM 51.1\%) on both accuracy and size. We also delimit the framework "
             r"honestly: a five-band capacity study (Section~\ref{subsec:capacity}) shows that varying "
             r"the parameter budget over a $9.2\times$ range leaves accuracy statistically unchanged, so "
             r"the remaining gap to recent high-capacity detectors (CFA 89.8\%, EfficientAD 97.5\%) is "
             r"methodological rather than budgetary. Cross-dataset evaluation across three datasets "
             r"(MVTec-AD, VisA, BTAD) reveals dataset-dependent architectural preferences: CNNs dominate "
             r"for texture-based defects (MVTec-AD, BTAD), while Transformers excel for complex "
             r"structures (VisA 75\%). This automatic adaptation eliminates manual architecture selection, "
             r"enabling rapid deployment across diverse industrial scenarios without requiring domain "
             r"expertise.")

INTRO_CLOSE_OLD = (r"Extensive ablations validate each component: dynamic alignment contributes +5.2\% "
                   r"AUROC, the searched student attains 82.4\% AUROC on MVTec-AD (above PaDiM and "
                   r"Reverse Distillation), and no fixed architecture dominates all categories. "
                   r"Cross-dataset evaluation confirms KG-NAS's adaptability to diverse data "
                   r"characteristics. These findings establish KG-NAS as a practical solution for "
                   r"automated architecture discovery in resource-constrained industrial anomaly detection.")
INTRO_CLOSE_NEW = (r"Extensive ablations validate each component: dynamic alignment contributes +5.2\% "
                   r"AUROC, the searched student attains 82.4\% AUROC on MVTec-AD (above PaDiM and "
                   r"Reverse Distillation), and no fixed architecture dominates all categories. "
                   r"Cross-dataset evaluation confirms KG-NAS's adaptability to diverse data "
                   r"characteristics, and a capacity study bounds what the framework can currently "
                   r"deliver. These findings establish KG-NAS as a practical solution for automated "
                   r"architecture discovery in industrial anomaly detection, while identifying the "
                   r"accuracy gap to high-capacity detectors as the open problem that motivates future work.")

# -------------------------------------------------------- efficiency table row
EFF_ROW_OLD = (r"STPM & 51.10 & $\sim$15 & $\sim$100 & $\sim$150 & Teacher-student \\" "\n"
               r"\midrule")
EFF_ROW_NEW = (r"STPM & 51.10 & $\sim$15 & $\sim$100 & $\sim$150 & Teacher-student \\" "\n"
               r"EfficientAD~\cite{batzner2024efficientad} & 97.50 & 8 & 1000+ & $\sim$60 & Real-time edge \\" "\n"
               r"SimpleNet~\cite{liu2023simplenet} & 98.00 & 12 & - & $\sim$120 & High-accuracy \\" "\n"
               r"\midrule")

EFF_CAP_OLD = (r"\caption{Efficiency comparison across methods. FPS: baseline values are literature-based "
               r"estimates; KG-NAS is measured on a single RTX 3090 (batch size 1, 224$\times$224 input). "
               r"KG-NAS targets resource-constrained scenarios with minimal parameters.}")
EFF_CAP_NEW = (r"\caption{Efficiency comparison across methods. FPS: baseline values are literature-based "
               r"estimates; KG-NAS is measured on a single RTX 3090 (batch size 1, 224$\times$224 input). "
               r"KG-NAS has the smallest deployed model of the methods compared, but EfficientAD reaches a "
               r"far higher AUROC at 8M parameters, so parameter count alone does not separate the "
               r"methods---see the capacity study in Section~\ref{subsec:capacity}.}")

# ---------------------------------------------------------------- conclusion
CONC_OLD = (r"Beyond accuracy, KG-NAS offers clear practical advantages for resource-constrained deployment: "
            r"the complete search and training process takes roughly 102 minutes on MVTec-AD on a single GPU, "
            r"making the framework suitable for rapid prototyping and deployment in industrial settings where "
            r"computational resources are limited and quick iteration is essential.")
CONC_NEW = (r"Beyond accuracy, KG-NAS keeps the full search-and-train cycle within roughly 102 minutes on "
            r"MVTec-AD on a single GPU, which suits rapid prototyping where architecture engineering, not "
            r"compute, is the bottleneck. We also report the limits of the approach. A five-band capacity "
            r"study shows that varying the parameter budget over a $9.2\times$ range leaves accuracy "
            r"statistically unchanged (1.7 points spread; all pairwise $p>0.05$), so the gap to recent "
            r"high-capacity detectors such as CFA and EfficientAD is a property of the search and detection "
            r"objective rather than of model size; closing it is the central open problem this work leaves.")

patches = [
    {"file": "sections/4_experiments.tex", "id": "N1-capacity-section",
     "find": "\\subsection{Discussion}\n\\label{subsec:discussion}",
     "replace": CAPACITY + "\\subsection{Discussion}\n\\label{subsec:discussion}",
     "note": "New capacity study, placed between Efficiency Analysis and Discussion."},
    {"file": "main_springer.tex", "id": "N2-abstract", "find": ABS_OLD, "replace": ABS_NEW,
     "note": "Abstract: honest benchmark positioning + capacity finding."},
    {"file": "main.tex", "id": "N2b-abstract-elsevier",
     "find": r"Finally, a paradigm study shows that reverse distillation and prototype-based scoring are complementary; combining them per category raises the average to 86.1\% AUROC, within four points of the strongest baseline. Code will be made publicly available.",
     "replace": r"Finally, a paradigm study shows that reverse distillation and prototype-based scoring are complementary; combining them per category raises the average to 86.1\% AUROC, within four points of the strongest baseline. The discovered architectures exceed the classical embedding-based and teacher--student baselines on both accuracy and size (PaDiM 77.9\%, Reverse Distillation 80.5\%, PatchCore 50.6\%, STPM 51.1\%), but trail recent high-capacity detectors (CFA 89.8\%, EfficientAD 97.5\%). A capacity study spanning 2.8M to 25.8M parameters finds no statistically significant change in accuracy, indicating that this remaining gap is methodological rather than an effect of the parameter budget. Code will be made publicly available.",
     "note": "Same abstract change in the Elsevier fallback."},
    {"file": "sections/1_introduction.tex", "id": "N3-intro-bullet", "find": INTRO_OLD, "replace": INTRO_NEW,
     "note": "Contribution 4: efficiency-first -> deployment-oriented discovery with a measured capacity envelope."},
    {"file": "sections/1_introduction.tex", "id": "N4-intro-close", "find": INTRO_CLOSE_OLD, "replace": INTRO_CLOSE_NEW,
     "note": "Closing paragraph: name the accuracy gap as the open problem."},
    {"file": "sections/4_experiments.tex", "id": "N5-eff-row", "find": EFF_ROW_OLD, "replace": EFF_ROW_NEW,
     "note": "Efficiency table: add EfficientAD and SimpleNet so the comparison is not selective."},
    {"file": "sections/4_experiments.tex", "id": "N6-eff-caption", "find": EFF_CAP_OLD, "replace": EFF_CAP_NEW,
     "note": "Efficiency caption: state explicitly that parameter count does not separate the methods."},
    {"file": "sections/5_conclusion.tex", "id": "N7-conclusion", "find": CONC_OLD, "replace": CONC_NEW,
     "note": "Conclusion: replace the deployment-friendly framing with the measured capacity bound."},
]

spec = {"tag": "round09_narrative", "paper_dir": _KGD_PAPER, "patches": patches}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(spec, f, ensure_ascii=False, indent=2)
print('wrote %d patches' % len(patches))
for p in patches:
    print('   %-22s %s' % (p['id'], p['file']))
