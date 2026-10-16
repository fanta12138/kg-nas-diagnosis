#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-run only the segments that failed in the first pass (S1,S3,S4,S5,S6),
then merge with the good S2 result into the final audit report."""
import json, os, sys, importlib.util, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rca", os.path.join(HERE, "run_claim_audit.py"))
rca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rca)

OUT_DIR = rca.OUT_DIR
PAPER_DIR = rca.PAPER_DIR
REPO = rca.REPO
SECTIONS = rca.SECTIONS

paper_all = {s: rca.read(f"{PAPER_DIR}/sections/{s}.tex") for s in SECTIONS}
main_tex = rca.read(f"{PAPER_DIR}/manuscript.tex")


def evf(*rel):
    return [(r, rca.read(f"{REPO}/{r}")) for r in rel if os.path.exists(f"{REPO}/{r}")]


def truth_compact():
    j = json.load(open(f"{REPO}/proxy_diag/phaseB_truth.json", encoding="utf-8"))
    rows = []
    for cell, d in sorted(j.items()):
        aurocs = d.get("aurocs") or d.get("auroc") or d
        rows.append(f"{cell} -> {json.dumps(aurocs, ensure_ascii=False)}")
    return ("proxy_diag/phaseB_truth.json [COMPACTED: one line per cell, key=category|backbone|seed]\n"
            + "\n".join(rows))


def forensics_compact():
    j = json.load(open(f"{REPO}/search_forensics/search_forensics.json", encoding="utf-8"))

    def scalars(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = {"__nested_items__": len(v)} if len(v) > 6 else scalars(v)
            elif isinstance(v, list):
                out[k] = {"__list_items__": len(v)} if len(v) > 6 else v
            else:
                out[k] = v
        return out
    keep = {k: (scalars(v) if isinstance(v, dict) else v) for k, v in j.items()}
    return ("search_forensics.json [COMPACTED: top-level summaries only]\n"
            + json.dumps(keep, ensure_ascii=False, indent=1))


seg4 = paper_all["4_diagnosis"]
marker = r"\subsection{What This Benchmark Design Can Resolve}"
ci = seg4.find(marker)
seg4a, seg4b = seg4[:ci], seg4[ci:]

SEGS = [
    ("S1 abstract / introduction / related work",
     "## manuscript.tex (title, abstract, keywords, declarations)\n" + main_tex
     + "\n\n## 1_introduction.tex\n" + paper_all["1_introduction"]
     + "\n\n## 2_related_work.tex\n" + paper_all["2_related_work"],
     evf("adbench/results/analysis_p1_p3.json",
         "search_repeats/search_vs_random_final.json",
         "search_repeats/search_repeats_summary.json")),

    ("S3 results — proxy repeatability + variance decomposition",
     seg4a,
     evf("proxy_diag/phaseA_proxy.json",
         "proxy_diag/phaseC_analysis.json",
         "proxy_diag/final_truth_analysis.json")
     + [("proxy_diag/phaseB_truth.json [COMPACTED]", truth_compact())]),

    ("S4 results — resolvability + repeated searches",
     seg4b,
     evf("proxy_diag/samplesize_analysis.json",
         "adbench/results/power_summary.json",
         "search_repeats/search_repeats.json",
         "search_repeats/search_repeats_summary.json",
         "search_repeats/search_vs_random_final.json")
     + [("search_forensics/search_forensics.json [COMPACTED]", forensics_compact())]),

    ("S5 results — inference-time scoring policy",
     paper_all["5_leverage"],
     evf("fusion_multiseed/fusion_multiseed_summary.json",
         "fusion_ksweep/fusion_ksweep_summary.json",
         "pseudo_select/pseudo_multiseed.json",
         "policy_select/policy_select.json")),

    ("S6 discussion + conclusions",
     paper_all["6_positioning"] + "\n\n" + paper_all["7_conclusion"],
     evf("anomalib_baselines/results.json",
         "adbench/results/analysis_p1_p3.json")),
]

results = []
for title, ptxt, ev in SEGS:
    print(f"--- rerun: {title} (paper {len(ptxt)} chars, {len(ev)} evidence files)", flush=True)
    t0 = time.time()
    try:
        res, raw = rca.audit_segment(title, ptxt, ev)
    except Exception as e:
        res, raw = {"claims": [], "segment_verdict": "ERROR", "notes": repr(e)[:500]}, str(e)
    res["_segment"] = title
    res["_raw"] = raw
    results.append(res)
    import re as _re
    slug = _re.sub(r"[^A-Za-z0-9]+", "_", title)[:60]
    with open(f"{OUT_DIR}/seg_{slug}.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n## PAPER TEXT (sent)\n\n{ptxt}\n\n## EVIDENCE FILES SENT\n\n")
        for n, _ in ev:
            f.write(f"- {n}\n")
        f.write("\n## REVIEWER RAW RESPONSE\n\n" + raw + "\n")
    print(f"    verdict={res.get('segment_verdict')} claims={len(res.get('claims',[]))} "
          f"({time.time()-t0:.0f}s)", flush=True)

with open(f"{OUT_DIR}/_rerun.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("DONE ->", f"{OUT_DIR}/_rerun.json")
