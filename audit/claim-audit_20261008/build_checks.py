"""Build the three-way re-read checks from the applied patch set."""
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

CHECKS = [
    {"name": "F1_abstract_winner_direction",
     "must_appear": ["ResNet-18 still leads on 8 of 15"],
     "must_absent": ["closes the gap and wins 12 of 15"],
     "sources": True},

    {"name": "F2a_abstract_exceeds",
     "must_appear": ["matching 98.2% of the average AUROC of a 12.9M ResNet-18 student under an identical protocol"],
     "must_absent": ["exceeding a 12.9M ResNet-18 student under an identical protocol"],
     "sources": True},

    {"name": "F2b_intro_exceeds",
     "must_appear": ["matching 98.2% of the average AUROC of a 12.9M ResNet-18 student trained under an identical protocol"],
     "must_absent": ["exceeding a 12.9M ResNet-18 student trained under an identical protocol"],
     "sources": True},

    {"name": "F6a_caption_padim",
     "must_appear": ["exceeds PaDiM (77.9%)"],
     "must_absent": ["exceeds PaDiM (78.0%)"],
     "sources": True},

    {"name": "F6b_table_average_row",
     "must_appear": [],
     "must_appear_in_sources": ["& 58.0 & 47.6 & 77.9 & 50.6 &"],
     "must_absent": ["& 58.0 & 47.6 & 78.0 & 50.6 &"],
     "sources": True},

    {"name": "F6c_prose_padim",
     "must_appear": ["above PaDiM (77.9%) and Reverse Distillation (80.5%)"],
     "must_absent": ["above PaDiM (78.0%) and Reverse Distillation (80.5%)"],
     "sources": True},

    {"name": "F9_tie_wording_body",
     "must_appear": ["sharing the ninth with Reverse Distillation on bottle and holding leather by 0.03"],
     "must_absent": ["two ties within 0.02"],
     "sources": True},

    {"name": "F9c_tie_wording_supplement",
     "must_appear": ["8 outright plus a shared best with Reverse Distillation on bottle"],
     "must_absent": ["including two ties within 0.02 pp"],
     "sources": True},

    {"name": "F10_cnn_type_count",
     "must_appear": ["CNNs (10 types, 65 trials)"],
     "must_absent": ["CNNs (13 types, 65 trials)"],
     "sources": True},

    {"name": "F11_trials_per_config",
     "must_appear": [],
     "must_appear_in_sources": ["~2.2 (100 total)"],
     "must_absent": ["~2.1 (100 total)"],
     "sources": True},

    {"name": "F13a_btad_range_table",
     "must_appear": [],
     "must_appear_in_sources": ["& 200--500 & 399--1000"],
     "must_absent": ["& 200--500 & 400--1000"],
     "sources": True},

    {"name": "F13b_btad_range_prose",
     "must_appear": ["399--1000 (BTAD) training samples"],
     "must_absent": ["400--1000 (BTAD) training samples"],
     "sources": True},

    {"name": "F4_best_trial_scope",
     "must_appear": ["0.0023, against 0.0032 for 128-dim and 0.0166 for 512-dim"],
     "must_absent": ["attained the single best trial score (0.0023)"],
     "sources": True},

    {"name": "F5_sota_params",
     "must_appear": [],
     "must_appear_in_sources": ["KG-NAS (Ours)} & 82.4 & - & \\textbf{5.3} &"],
     "must_absent": ["KG-NAS (Ours)} & 82.4 & - & \\textbf{6} &"],
     "sources": True},

    {"name": "M1_107fps_removed",
     "must_appear": ["142 FPS student throughput"],
     "must_absent": ["107 FPS"],
     "sources": True},
]

with open(_os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008'),
          'w', encoding='utf-8') as f:
    json.dump(CHECKS, f, ensure_ascii=False, indent=2)
print('wrote %d checks' % len(CHECKS))
