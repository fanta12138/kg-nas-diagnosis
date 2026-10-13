"""Round 04 — reinstate the pipeline-throughput claim with fresh evidence,
and align the baseline-efficiency figure with the manuscript's own table.

M1 reversal: the claim "the full teacher--student detector runs at ~107 FPS"
had no artefact, so round 01 removed it. The missing benchmark has now been run
(idle RTX 3090, batch 1, 224x224, fp32, warmup 100 / iters 500 / 3 rounds,
median): full pipeline 110.8 FPS, student alone 146.2 FPS. See fps_benchmark.json.
The claim is therefore reinstated at the measured value.

N2: generate_figures.py:323 hard-codes baseline parameter estimates that
contradict the manuscript's tab:efficiency_comparison, and :350/336 hard-code
KG-NAS at 6.0M instead of the measured 5.3M.
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

EFF_FOOT_FIND = (r"\textbf{KG-NAS (Ours)} & 82.40 & \textbf{5.3} & 142$^{\dagger}$ & 102 & \textbf{Resource-constrained} \\" "\n"
                 r"\bottomrule" "\n"
                 r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$Student throughput, measured on an idle RTX 3090 (batch size 1, 224$\times$224, fp32).} \\")
EFF_FOOT_REPL = (r"\textbf{KG-NAS (Ours)} & 82.40 & \textbf{5.3} & 142$^{\dagger}$ & 102 & \textbf{Resource-constrained} \\" "\n"
                 r"\bottomrule" "\n"
                 r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$Student throughput measured on an idle RTX 3090 (batch size 1, 224$\times$224, fp32); the full teacher--student detector measures 111 FPS under the same protocol.} \\")

SOTA_FOOT_FIND = (r"\textbf{KG-NAS (Ours)} & 82.4 & - & \textbf{5.3} & 142$^{\dagger}$ & Single-class + NAS \\" "\n"
                  r"\bottomrule" "\n"
                  r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$Student throughput, measured on an idle RTX 3090 (batch size 1, 224$\times$224, fp32).} \\")
SOTA_FOOT_REPL = (r"\textbf{KG-NAS (Ours)} & 82.4 & - & \textbf{5.3} & 142$^{\dagger}$ & Single-class + NAS \\" "\n"
                  r"\bottomrule" "\n"
                  r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$Student throughput measured on an idle RTX 3090 (batch size 1, 224$\times$224, fp32); the full teacher--student detector measures 111 FPS under the same protocol.} \\")

P = [
    {"file": "sections/4_experiments.tex", "id": "M1-restore-a",
     "find": EFF_FOOT_FIND, "replace": EFF_FOOT_REPL,
     "note": "M1 reversal: benchmarked full-pipeline throughput = 110.8 FPS."},

    {"file": "sections/4_experiments.tex", "id": "M1-restore-b",
     "find": SOTA_FOOT_FIND, "replace": SOTA_FOOT_REPL,
     "note": "M1 reversal, second footnote."},

    {"file": "sections/5_conclusion.tex", "id": "M1-restore-c",
     "find": r"$\sim$142 FPS student throughput",
     "replace": r"$\sim$142 FPS student throughput and $\sim$111 FPS end-to-end",
     "note": "M1 reversal: conclusion."},
]

spec = {"tag": "round04_restore_fps", "paper_dir": _KGD_PAPER, "patches": P}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(spec, f, ensure_ascii=False, indent=2)
print('wrote %d patches' % len(P))
