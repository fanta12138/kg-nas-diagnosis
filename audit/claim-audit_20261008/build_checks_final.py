"""Final three-way checks covering round 01 + round 03/04/05 fixes."""
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

B = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')

MAIN = [
    # ---- round 01 (already verified, re-asserted) ----
    {"name": "R1_F1_abstract_winner", "must_appear": ["ResNet-18 still leads on 8 of 15"],
     "must_absent": ["closes the gap and wins 12 of 15"], "sources": True},
    {"name": "R1_F2_abstract_exceeds",
     "must_appear": ["matching 98.2% of the average AUROC of a 12.9M ResNet-18 student under an identical protocol"],
     "must_absent": ["exceeding a 12.9M ResNet-18 student under an identical protocol"], "sources": True},
    {"name": "R1_F6_padim", "must_appear": ["above PaDiM (77.9%)"],
     "must_absent": ["above PaDiM (78.0%)"], "sources": True},
    {"name": "R1_F10_cnn_types", "must_appear": ["CNNs (10 types, 65 trials)"],
     "must_absent": ["CNNs (13 types, 65 trials)"], "sources": True},
    {"name": "R1_F13_btad_range", "must_appear": ["399--1000 (BTAD) training samples"],
     "must_absent": ["400--1000 (BTAD) training samples"], "sources": True},
    {"name": "R1_F4_best_trial", "must_appear": ["0.0023, against 0.0032 for 128-dim"],
     "must_absent": ["attained the single best trial score (0.0023)"], "sources": True},

    # ---- round 03: U1 search space ----
    {"name": "U1_no_45_config", "must_appear": [],
     "must_absent": ["45 configurations", "45 candidate architectures", "45 candidates"],
     "sources": True},
    {"name": "U1_no_15_backbones", "must_appear": [],
     "must_absent": ["15 backbones", "15 diverse backbone", "15 CNN and Transformer",
                     "15-backbone"], "sources": True},
    {"name": "U1_42_present", "must_appear": ["14 backbones", "14 backbone architectures"],
     "must_absent": [], "sources": True},
    {"name": "U1_effv2s_row_gone", "must_appear": [],
     "must_appear_in_sources": ["MobileOne-S0 & CNN & 2022 & 5.3 & 1.1"],
     "must_absent": ["EfficientNetV2-S & CNN & 2021 & 21.5 & 8.4"], "sources": True},
    {"name": "U1_trials_per_config_2_4", "must_appear": [],
     "must_appear_in_sources": ["NAS trials per config & $\\sim$2.4 (100 total)"],
     "must_absent": ["NAS trials per config & $\\sim$2.2 (100 total)"], "sources": True},

    # ---- round 03: U3 ----
    {"name": "U3_spread_106", "must_appear": ["vary by up to 106"],
     "must_absent": ["vary by up to 81"], "sources": True},

    # ---- round 04: M1 reinstated with measurement ----
    {"name": "M1_pipeline_fps_measured",
     "must_appear": ["the full teacher--student detector measures 111 FPS"],
     "must_absent": ["detector runs at $\\sim$107 FPS"], "sources": True},

    # ---- round 05: U5 data availability ----
    {"name": "U5_data_availability",
     "must_appear": ["a single 100-trial search for MVTec-AD run on one category"],
     "must_absent": ["100 NAS trials per category on each of the three datasets"], "sources": True},
]

SUPP = [
    {"name": "U1_supp_42_config",
     "must_appear": [], "must_appear_in_sources": ["the compact 42-configuration space"],
     "must_absent": ["the compact 45-configuration space"], "sources": True},
]

json.dump(MAIN, open(B + 'checks_final_main.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
json.dump(SUPP, open(B + 'checks_final_supp.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('main checks: %d   supp checks: %d' % (len(MAIN), len(SUPP)))
