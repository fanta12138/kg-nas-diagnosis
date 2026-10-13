"""Emit the audit artefacts (PAPER_CLAIM_AUDIT.json/.md, REVIEW_STATE.json).

Hash convention follows the skill's contract: paper files use paths relative to
paper_dir, results use absolute paths, every value carries the 'sha256:' prefix.
Run LAST — after every recompilation — so the hashes describe the shipped state.
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

import hashlib, json, os, glob
from datetime import datetime

PAPER = _KGD_PAPER
TRACE = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')

PAPER_FILES = ['main_springer.tex', 'main.tex', 'math_commands.tex',
               'references.bib',
               'sections/1_introduction.tex', 'sections/2_related_work.tex',
               'sections/3_method.tex', 'sections/4_experiments.tex',
               'sections/5_conclusion.tex',
               'tables/BASELINE_ANALYSIS_TABLE.tex',
               'supplementary/supplementary.tex',
               'main_springer.pdf', 'main.pdf',
               'supplementary/supplementary.pdf']

RESULT_GLOBS = [
    _os.path.join(_KGD_ROOT, 'fig', '*.csv'),
    _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'paper_results', '*.csv'),
    _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'paper_results', 'ablation_phase1', '*.csv'),
    _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', '*.csv'),
    _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', '*.json'),
    _os.path.join(_KGD_ROOT, 'unsupervised_nas', 'experiments', 'ablation_*', '**', '*.csv'),
    _os.path.join(_KGD_ROOT, 'experiments', 'backbone_stability_analysis.json'),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return 'sha256:' + h.hexdigest()


hashes = {}
for rel in PAPER_FILES:
    p = os.path.join(PAPER, rel)
    if os.path.exists(p):
        hashes[rel] = sha(p)

for g in RESULT_GLOBS:
    for p in sorted(glob.glob(g, recursive=True)):
        if os.path.isfile(p):
            hashes[os.path.abspath(p).replace('\\', '/')] = sha(p)

audit = {
    "paper": "KG-NAS: Knowledge-Guided Neural Architecture Search for Unsupervised Anomaly Detection",
    "target_journal": "Machine Vision and Applications (Springer)",
    "round": "2026-10-08_run01",
    "generated": datetime.now().isoformat(timespec='seconds'),
    "review_independence": "same-family",
    "acceptance_status": "provisional",
    "independence_note": ("The zero-context reviewer was a fresh sub-agent of the same model family "
                          "(no cross-family channel available in this environment). Its verdict is "
                          "an input, not an acceptance: the executor re-derived every finding from "
                          "raw evidence before acting."),
    "verdict": "PASS_AFTER_FIX",
    "reason_code": "claim_mismatch_remediated",
    "rounds": [
        {
            "round": "reviewer",
            "verdict": "FAIL",
            "reason_code": "claim_mismatch",
            "total_claims": 178, "exact": 152, "rounding": 9,
            "ambiguous": 5, "missing": 1, "mismatch": 11,
            "artefacts": ["reviewer_report.md", "findings.json"]
        },
        {
            "round": "executor_verification",
            "confirmed": 11, "false_positives": 4, "open_items": 7,
            "artefacts": ["executor_verification.json", "verify_F1F2.log",
                          "verify_rest.log", "verify_focus.log", "verify_round.log",
                          "verify_params2.log"]
        },
        {
            "round": "post_fix_reread",
            "main_checks": 14, "main_failed": 0,
            "supplement_checks": 1, "supplement_failed": 0,
            "extra_fix_from_reread": ("the re-read caught that main.tex (Elsevier fallback) "
                                      "carries its own copy of the abstract and still had both "
                                      "direction errors; patched in round 02"),
            "artefacts": ["checks_main.json", "checks_supp.json", "threeway.py"]
        }
    ],
    "fixes_applied": [
        {"id": "F1",  "file": "main_springer.tex", "what": "abstract: 500-epoch winner direction 12/15 -> 3/15 (ResNet-18 leads 8)"},
        {"id": "F1b", "file": "main.tex",          "what": "same fix in the Elsevier fallback abstract"},
        {"id": "F2a", "file": "main_springer.tex", "what": "abstract: 'exceeding a 12.9M ResNet-18' -> 'matching 98.2% of the average AUROC of'"},
        {"id": "F2b", "file": "sections/1_introduction.tex", "what": "same fix in the introduction"},
        {"id": "F2c", "file": "main.tex",          "what": "same fix in the Elsevier fallback abstract"},
        {"id": "F4",  "file": "sections/4_experiments.tex", "what": "Ablation 4: scope the 0.0023 figure to the 39-trial feature-dimension subset instead of calling it the NAS best trial (0.0097)"},
        {"id": "F5",  "file": "sections/4_experiments.tex", "what": "tab:sota_comparison params 6 -> 5.3"},
        {"id": "F6",  "file": "sections/4_experiments.tex", "what": "PaDiM average 78.0 -> 77.9 (caption, table row, prose, conclusion)"},
        {"id": "F6d", "file": "sections/5_conclusion.tex",  "what": "PaDiM 78.0 -> 77.9"},
        {"id": "F9",  "file": "sections/4_experiments.tex + supplementary/supplementary.tex",
         "what": "'two ties within 0.02 pp' -> accurate wording (one shared best on bottle; 0.034 pp leather margin)"},
        {"id": "F10", "file": "sections/4_experiments.tex", "what": "CNN type count 13 -> 10"},
        {"id": "F11", "file": "sections/3_method.tex",      "what": "trials per config 2.1 -> 2.2"},
        {"id": "F13", "file": "sections/4_experiments.tex", "what": "BTAD train-sample lower bound 400 -> 399"},
        {"id": "M1",  "file": "sections/4_experiments.tex + sections/5_conclusion.tex",
         "what": "removed the untraceable '~107 FPS full pipeline' claim (no artefact exists)"},
        {"id": "N1",  "file": _os.path.join(_KGD_ROOT, 'fig', 'generate_figures.py'),
         "what": "figure data params corrected 6.0/14.9/2.5/28.0 -> 5.3/12.9/2.7/29.3 to match the (independently verified) manuscript and the figure's own caption; 19 PNGs regenerated at 1200 dpi and resynced"}
    ],
    "rejected_as_false_positive": [
        {"id": "F3",  "why": "RDDetector (the authors' own model) measures encoder+decoder at 5.41/13.04/2.66/29.25 M = the printed 5.3/12.9/2.7/29.3; ratio 2.41 = 2.4x. The reviewer trusted arch_efficiency.csv, which was itself the wrong artefact."},
        {"id": "F7",  "why": "carpet 77.7 follows the paper's consistent two-step rounding (77.6485 -> 77.65 -> 77.7); applying it to all 11 baseline means reproduces 10 exactly."},
        {"id": "F8",  "why": "CFA 89.8 is the same two-step convention (89.7465 -> 89.75 -> 89.8); the detailed table prints 89.75."},
        {"id": "F12", "why": "same root as F3; 5.3M for EfficientNet-B0+256d is correct."}
    ],
    "open_for_user_decision": [
        {"id": "U1", "issue": "15 vs 16 backbones: evidence contains both counts; the 93-trial run covers 13 distinct backbones."},
        {"id": "U2", "issue": "BTAD '10 backbones x 3 = 30 configurations' has no supporting artefact."},
        {"id": "U3", "issue": "'vary by up to 81x within a category' not reproducible (subsets give 1.5x-211x)."},
        {"id": "U4", "issue": "'~6x the computation of training a single architecture' untraceable."},
        {"id": "U5", "issue": "data-availability says '100 NAS trials per category'; the MVTec search is one 93-trial aggregate."},
        {"id": "U7", "issue": "baseline FPS are literature estimates (labelled); fig/efficiency_metrics.csv is stale."},
        {"id": "U8", "issue": "Stage-3 batch size 64 stated, but BTAD Product 03 ran at 32."}
    ],
    "build": {
        "main_springer.pdf": {"pages": 28, "errors": 0, "undefined_refs": 0},
        "main.pdf":          {"pages": 21, "errors": 0, "undefined_refs": 0},
        "supplementary/supplementary.pdf": {"pages": 14, "errors": 0, "undefined_refs": 0}
    },
    "audited_input_hashes": hashes
}

with open(os.path.join(PAPER, 'PAPER_CLAIM_AUDIT.json'), 'w', encoding='utf-8') as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)

state = {
    "status": "audited_and_fixed",
    "timestamp": audit["generated"],
    "rounds_completed": 3,
    "verdict": "PASS_AFTER_FIX",
    "review_independence": "same-family",
    "acceptance_status": "provisional",
    "open_for_user_decision": [d["id"] for d in audit["open_for_user_decision"]]
}
with open(os.path.join(PAPER, 'REVIEW_STATE.json'), 'w', encoding='utf-8') as f:
    json.dump(state, f, ensure_ascii=False, indent=2)

print('PAPER_CLAIM_AUDIT.json  -> %d hashed inputs' % len(hashes))
print('REVIEW_STATE.json')
