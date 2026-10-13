"""Build patch round 03 — search-space corrections proven from the experiment tree.

Evidence for U1 (<PAPER_ROOT>/.aris/traces/.../resolve_open_items.md):
  <REPO_ROOT>/unsupervised_nas/configs/stage2_nas_search.yaml:20
      # - 'efficientnetv2_s'    # 无预训练权重，暂时移除
  -> the MVTec search space is 14 backbones x 3 dims = 42 configurations,
     not 15 x 3 = 45. EfficientNetV2-S must leave the table.

Evidence for U3: max within-configuration score ratio over 44 repeated groups
  is 105.9x (mobileone_s0/128, n=4 in the 93-trial run); no group measures 81x.
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

P = [
    # ---- U1: drop the backbone that was removed from the search space ----
    {"file": "sections/3_method.tex", "id": "U1-a",
     "find": "EfficientNetV2-S & CNN & 2021 & 21.5 & 8.4 \\\\\nMobileOne-S0 & CNN & 2022 & 5.3 & 1.1 \\\\",
     "replace": "MobileOne-S0 & CNN & 2022 & 5.3 & 1.1 \\\\",
     "note": "U1: EfficientNetV2-S was commented out of stage2_nas_search.yaml (no pretrained weights)."},

    {"file": "sections/3_method.tex", "id": "U1-b",
     "find": "Our search space includes 15 diverse backbone architectures spanning different design paradigms and eras",
     "replace": "Our search space includes 14 diverse backbone architectures spanning different design paradigms and eras",
     "note": "U1"},

    {"file": "sections/3_method.tex", "id": "U1-c",
     "find": "We include 15 diverse backbones spanning CNN and Transformer architectures from 2015 to 2023",
     "replace": "We include 14 diverse backbones spanning CNN and Transformer architectures from 2015 to 2023",
     "note": "U1"},

    {"file": "sections/3_method.tex", "id": "U1-d",
     "find": "This creates a search space of $15 \\times 3 = 45$ candidate architectures",
     "replace": "This creates a search space of $14 \\times 3 = 42$ candidate architectures",
     "note": "U1"},

    {"file": "sections/3_method.tex", "id": "U1-e",
     "find": "($K \\approx 2.2$ trials per architecture on average given 45 candidates)",
     "replace": "($K \\approx 2.4$ trials per architecture on average given 42 candidates)",
     "note": "U1: 100/42 = 2.38."},

    {"file": "sections/3_method.tex", "id": "U1-f",
     "find": "NAS trials per config & $\\sim$2.2 (100 total) \\\\",
     "replace": "NAS trials per config & $\\sim$2.4 (100 total) \\\\",
     "note": "U1"},

    {"file": "sections/3_method.tex", "id": "U1-g",
     "find": "With our configuration ($|\\mathcal{A}|=45$, $K \\approx 2.2$, $E_S=3$)",
     "replace": "With our configuration ($|\\mathcal{A}|=42$, $K \\approx 2.4$, $E_S=3$)",
     "note": "U1"},

    {"file": "sections/3_method.tex", "id": "U1-h",
     "find": "We search over a diverse space of 15 backbone architectures $\\times$ 3 feature dimensions (45 configurations) using knowledge distillation.",
     "replace": "We search over a diverse space of 14 backbone architectures $\\times$ 3 feature dimensions (42 configurations) using knowledge distillation.",
     "note": "U1: Stage-2 paragraph."},

    {"file": "sections/1_introduction.tex", "id": "U1-h2",
     "find": "We search over a diverse space of 15 backbone architectures $\\times$ 3 feature dimensions (45 configurations) using knowledge distillation.",
     "replace": "We search over a diverse space of 14 backbone architectures $\\times$ 3 feature dimensions (42 configurations) using knowledge distillation.",
     "note": "U1: introduction Stage-2 paragraph (same wording as the method section)."},

    {"file": "sections/3_method.tex", "id": "U1-i",
     "find": "Stage 2 searches over 15 backbones $\\times$ 3 feature dimensions using a knowledge-distillation score",
     "replace": "Stage 2 searches over 14 backbones $\\times$ 3 feature dimensions using a knowledge-distillation score",
     "note": "U1: overview caption."},

    {"file": "sections/3_method.tex", "id": "U1-j",
     "find": "and a search space of 15 backbones spanning 2015--2023 that covers both the CNN and Transformer paradigms",
     "replace": "and a search space of 14 backbones spanning 2015--2023 that covers both the CNN and Transformer paradigms",
     "note": "U1"},

    {"file": "sections/1_introduction.tex", "id": "U1-k",
     "find": "We design a diverse search space covering 15 backbones spanning CNN and Transformer paradigms from 2015 to 2023.",
     "replace": "We design a diverse search space covering 14 backbones spanning CNN and Transformer paradigms from 2015 to 2023.",
     "note": "U1"},

    {"file": "sections/4_experiments.tex", "id": "U1-l",
     "find": "100 trials over 15 backbones $\\times$ 3 feature dimensions, 3 epochs per trial.",
     "replace": "100 trials over 14 backbones $\\times$ 3 feature dimensions, 3 epochs per trial.",
     "note": "U1"},

    {"file": "sections/4_experiments.tex", "id": "U1-m",
     "find": "KG-NAS's diverse search space (15 backbones) enables discovery",
     "replace": "KG-NAS's diverse search space (14 backbones) enables discovery",
     "note": "U1"},

    {"file": "sections/4_experiments.tex", "id": "U1-n",
     "find": "we examine the search-score landscape across the full 15-backbone space (93 recorded trials)",
     "replace": "we examine the search-score landscape across the full 14-backbone space (93 recorded trials)",
     "note": "U1"},

    {"file": "sections/4_experiments.tex", "id": "U1-o",
     "find": "Including 15 backbones spanning 2015-2023 ensures exploration",
     "replace": "Including 14 backbones spanning 2015-2023 ensures exploration",
     "note": "U1"},

    {"file": "sections/5_conclusion.tex", "id": "U1-p",
     "find": "the search averages 2.2 trials per configuration (100 trials over 45 configurations)",
     "replace": "the search averages 2.4 trials per configuration (100 trials over 42 configurations)",
     "note": "U1"},

    {"file": "supplementary/supplementary.tex", "id": "U1-q",
     "find": "random search is a strong baseline on the compact 45-configuration space",
     "replace": "random search is a strong baseline on the compact 42-configuration space",
     "note": "U1"},

    # ---- U3: the within-configuration spread is 106x, not 81x ----
    {"file": "sections/5_conclusion.tex", "id": "U3",
     "find": "the same architecture re-evaluated under identical settings can vary by up to 81$\\times$ within a category",
     "replace": "the same architecture re-evaluated under identical settings can vary by up to 106$\\times$ within a category",
     "note": "U3: max ratio over 44 repeated (category,backbone,dim) groups is 105.9x."},
]

# the abstract lives in two files
for f in ["main_springer.tex", "main.tex"]:
    P.append({"file": f, "id": "U1-abs-%s" % f,
              "find": "guides the search over 15 CNN and Transformer backbones",
              "replace": "guides the search over 14 CNN and Transformer backbones",
              "note": "U1: abstract."})

spec = {"tag": "round03_search_space", "paper_dir": _KGD_PAPER, "patches": P}
with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(spec, fh, ensure_ascii=False, indent=2)
print('wrote %d patches' % len(P))
