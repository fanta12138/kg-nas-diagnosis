# -*- coding: utf-8 -*-
"""Round 10 — honesty pass on cherry-picked wording + the capacity limitation."""
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

CONC_SENT_OLD = (r"On MVTec-AD, KG-NAS attains 82.4\% average AUROC across 15 categories with a "
                 r"5.3M-parameter student, exceeding PaDiM (78.0\%) and Reverse Distillation (80.5\%) "
                 r"and reaching near-SOTA accuracy on several categories (zipper 96.9\%, bottle 96.3\%, "
                 r"hazelnut 96.3\%).")
CONC_SENT_NEW = (r"On MVTec-AD, KG-NAS attains 82.4\% average AUROC across 15 categories with a "
                 r"5.3M-parameter student, exceeding PaDiM (77.9\%) and Reverse Distillation (80.5\%) "
                 r"while remaining 7.4 points below the strongest baseline (CFA, 89.8\%); its strongest "
                 r"categories are zipper (96.9\%), bottle (96.3\%) and hazelnut (96.3\%).")

OBS_OLD = (r"Performance is near-SOTA on several categories (zipper 96.9\%, bottle 96.3\%, hazelnut "
           r"96.3\%, wood 94.6\%, tile 93.3\%) and weaker on the challenging ones (screw 44.0\%, "
           r"grid 67.8\%, capsule 74.8\%);")
OBS_NEW = (r"Performance is strongest on zipper (96.9\%), bottle (96.3\%), hazelnut (96.3\%), wood "
           r"(94.6\%) and tile (93.3\%), and weakest on screw (44.0\%), grid (67.8\%) and capsule "
           r"(74.8\%);")

LIM_OLD = (r"larger budgets could find better architectures at the cost of search time. "
           r"\emph{The proxy metric is the binding constraint.}")
LIM_NEW = (r"larger budgets could find better architectures at the cost of search time. "
           r"\emph{Capacity is not the constraint.} Across five mutually exclusive parameter bands the "
           r"average AUROC moves by only 1.7 points and not one pairwise comparison is significant "
           r"(Section~\ref{subsec:capacity}), so the framework has a fixed accuracy envelope near 83\% on "
           r"MVTec-AD. The gap to CFA (89.8\%) and EfficientAD (97.5\%) therefore cannot be closed by "
           r"allocating more parameters; it requires a different detection or search objective, which we "
           r"regard as the central open problem for this line of work. "
           r"\emph{The proxy metric is the binding constraint.}")

patches = [
    {"file": "sections/5_conclusion.tex", "id": "H1-conc-padim-nearsota",
     "find": CONC_SENT_OLD, "replace": CONC_SENT_NEW,
     "note": "Conclusion: PaDiM 78.0 -> 77.9 (missed in round 01, inconsistent with the body) and drop the near-SOTA claim."},
    {"file": "sections/4_experiments.tex", "id": "H2-nearsota",
     "find": OBS_OLD, "replace": OBS_NEW,
     "note": "Main-results observations: 'near-SOTA' is cherry-picked; KG-NAS only beats CFA on 3 of 15 categories."},
    {"file": "sections/5_conclusion.tex", "id": "H3-capacity-limitation",
     "find": LIM_OLD, "replace": LIM_NEW,
     "note": "Limitations: add the measured capacity envelope and name the accuracy gap as the open problem."},
]

spec = {"tag": "round10_honesty", "paper_dir": _KGD_PAPER, "patches": patches}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(spec, f, ensure_ascii=False, indent=2)
print('wrote %d patches' % len(patches))
