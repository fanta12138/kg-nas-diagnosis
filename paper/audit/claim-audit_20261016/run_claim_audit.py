import os
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zero-context paper-to-evidence audit driver.

Feeds the manuscript and raw result files to GLM-5.3-flash in SEGMENTS
(the reviewer model truncates oversized single inputs), collects one audit
verdict per segment, then merges into PAPER_CLAIM_AUDIT.md / .json.

Usage:  python run_claim_audit.py
"""
import json, os, sys, time, hashlib, urllib.request, urllib.error, re

API_KEY = os.environ["GLM_API_KEY"]
BASE    = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL   = "glm-5.3-flash"

PAPER_DIR = "G:/submission_ApplSci"
REPO      = "G:/kg-nas-diagnosis"
OUT_DIR   = "G:/submission_KBS/audit/claim-audit_20261016"

SECTIONS = ["1_introduction", "2_related_work", "3_method",
            "4_diagnosis", "5_leverage", "6_positioning", "7_conclusion"]


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def call(messages, max_tokens=64000, retries=3):
    """Single chat call.

    glm-5.3-flash is a REASONING model: reasoning tokens are billed inside
    max_tokens, and content stays empty until reasoning finishes
    (finish_reason='length' with empty content == budget exhausted mid-thought).
    So the budget must be generous.
    """
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    last = None
    for i in range(retries):
        req = urllib.request.Request(BASE, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        })
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                j = json.loads(r.read().decode())
            ch = j["choices"][0]
            content = ch["message"].get("content") or ""
            if not content.strip():
                last = f"empty content (finish={ch.get('finish_reason')})"
                time.sleep(3)
                continue
            return content, j.get("usage", {})
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:200]}"
            time.sleep(4)
        except Exception as e:
            last = repr(e)[:200]
            time.sleep(4)
    raise RuntimeError("call failed: " + str(last))


def sha(p):
    with open(p, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


SYSTEM = (
    "You are a paper-to-evidence auditor with ZERO prior context about this "
    "research. You receive ONLY (a) paper source text and (b) raw result files. "
    "You verify that every number, comparison and scope statement in the paper "
    "matches the raw evidence EXACTLY. You never assume a number is right; you "
    "trace it. Report only what the given files support.\n\n"
    "IMPORTANT: keep your internal deliberation brief. Do NOT restate the paper "
    "or the evidence files back to the reader. Go straight to the audit table "
    "and then emit the JSON. Brevity matters: an extremely long preamble will "
    "cause you to hit the output limit before the JSON is written."
)

PROTOCOL = """
## Audit protocol

### A. Extract every quantitative claim from the PAPER TEXT below
number, percentage, comparison, or scope statement — with its exact quote.

### B. Trace each claim to the EVIDENCE FILES
Find the exact value. Match status must be one of:
exact_match | rounding_ok | ambiguous_mapping | missing_evidence |
config_mismatch | aggregation_mismatch | number_mismatch |
scope_overclaim | unsupported_claim

Rounding rule: only standard rounding to the displayed precision is allowed
(0.8334 -> 83.3% or 83% is OK; -> 85.3% is NOT).
Use missing_evidence ONLY when the value is genuinely absent from the files
given (e.g. a hyperparameter that no file records) — for method/implementation
constants this is expected and not a defect; still list it.

### C. Flag these failure modes explicitly
1. number inflation      2. best-seed cherry-pick   3. config mismatch
4. aggregation mismatch (claims N seeds, files show fewer)
5. delta arithmetic error   6. caption/table mismatch
7. scope overclaim ("consistently", "all", "demonstrates" beyond evidence)

## OUTPUT — strict format
Return ONE fenced ```json block and nothing after it. No prose before it
beyond 2–3 lines. Format:
{
  "claims": [
    {"id": 1, "location": "...", "paper_text": "...", "paper_value": "...",
     "evidence_file": "...", "evidence_value": "...", "status": "...",
     "details": "..."}
  ],
  "segment_verdict": "PASS" | "WARN" | "FAIL",
  "notes": "anything ambiguous"
}
Keep each "details" under 25 words.
"""


def audit_segment(title, paper_text, evidence_blocks):
    ev = "\n\n".join(
        f"===== EVIDENCE FILE: {name} =====\n{content}"
        for name, content in evidence_blocks
    )
    user = (
        f"# SEGMENT: {title}\n\n"
        f"## PAPER TEXT\n\n{paper_text}\n\n"
        f"## EVIDENCE FILES\n\n{ev}\n\n"
        f"{PROTOCOL}"
    )
    content, usage = call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ])
    print(f"  [{title}] returned {len(content)} chars, usage={usage.get('total_tokens')}")
    # extract json block (tolerate ```json / ``` / no fence)
    m = re.search(r"```json\s*(.*?)```", content, re.S)
    if not m:
        m = re.search(r"```\s*(\{.*?\})\s*```", content, re.S)
    raw = m.group(1) if m else content
    # last resort: grab the outermost {...}
    if not m:
        i, j2 = raw.find("{"), raw.rfind("}")
        if i >= 0 and j2 > i:
            raw = raw[i:j2 + 1]
    try:
        return json.loads(raw), content
    except Exception as e:
        print(f"    !! parse failed: {e}; tail of response: ...{content[-300:]!r}")
        return {"claims": [], "segment_verdict": "PARSE_ERROR",
                "notes": f"parse: {e}; head={content[:400]}"}, content


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paper_all = {s: read(f"{PAPER_DIR}/sections/{s}.tex") for s in SECTIONS}
    main_tex  = read(f"{PAPER_DIR}/manuscript.tex")

    def evf(*rel):
        out = []
        for r in rel:
            p = f"{REPO}/{r}"
            if os.path.exists(p):
                out.append((r, read(p)))
        return out

    def truth_compact():
        """phaseB_truth is ~400 nested cells; compress to one line per cell."""
        j = json.load(open(f"{REPO}/proxy_diag/phaseB_truth.json", encoding="utf-8"))
        rows = []
        for cell, d in sorted(j.items()):
            aurocs = d.get("aurocs") or d.get("auroc") or d
            rows.append(f"{cell} -> {json.dumps(aurocs, ensure_ascii=False)}")
        return ("proxy_diag/phaseB_truth.json [COMPACTED: one line per cell, "
                "cell key = category|backbone|seed]\n" + "\n".join(rows))

    def forensics_compact():
        """search_forensics.json is ~113 KB (per-run and per-trial records).
        Keep only the top-level scalar summaries the paper actually cites."""
        j = json.load(open(f"{REPO}/search_forensics/search_forensics.json", encoding="utf-8"))

        def scalars(d, depth=0):
            """Recursively keep scalars; collapse lists/dicts of length>6 to a size note."""
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    if len(v) > 6:
                        out[k] = {"__nested_items__": len(v)}
                    else:
                        out[k] = scalars(v, depth + 1)
                elif isinstance(v, list):
                    out[k] = {"__list_items__": len(v)} if len(v) > 6 else v
                else:
                    out[k] = v
            return out

        keep = {k: (scalars(v) if isinstance(v, dict) else v) for k, v in j.items()}
        return ("search_forensics.json [COMPACTED: top-level summaries only; "
                "nested tables of size>6 collapsed to a count]\n"
                + json.dumps(keep, ensure_ascii=False, indent=1))

    segments = []

    # S1 — abstract + intro + related work  (no numeric evidence needed beyond framing)
    segments.append((
        "S1 abstract / introduction / related work",
        "## manuscript.tex (title, abstract, keywords, declarations)\n"
        + main_tex + "\n\n## 1_introduction.tex\n" + paper_all["1_introduction"]
        + "\n\n## 2_related_work.tex\n" + paper_all["2_related_work"],
        evf("adbench/results/analysis_p1_p3.json",
            "search_repeats/search_vs_random_final.json",
            "search_repeats/search_repeats_summary.json"),
    ))

    # S2 — method (pipeline, 42-cell space, degenerate objective)
    segments.append((
        "S2 materials and methods",
        paper_all["3_method"],
        evf("proxy_diag/backbone_params.json",
            "proxy_diag/probe_degenerate.json",
            "synthetic_mechanism/synthetic_mechanism.json"),
    ))

    # S3 — diagnosis part A: proxy repeatability + variance decomposition
    seg4 = paper_all["4_diagnosis"]
    # split at the real \subsection{What This Benchmark Design Can Resolve}
    marker = r"\subsection{What This Benchmark Design Can Resolve}"
    ci = seg4.find(marker)
    if ci < 0:
        raise RuntimeError("split marker not found in 4_diagnosis.tex")
    seg4a, seg4b = seg4[:ci], seg4[ci:]
    segments.append((
        "S3 results — proxy repeatability + variance decomposition",
        seg4a,
        evf("proxy_diag/phaseA_proxy.json",
            "proxy_diag/phaseC_analysis.json",
            "proxy_diag/final_truth_analysis.json")
        + [("proxy_diag/phaseB_truth.json [COMPACTED]", truth_compact())],
    ))

    # S4 — diagnosis part B: resolvability + searches worse than random
    segments.append((
        "S4 results — resolvability + repeated searches",
        seg4b,
        evf("proxy_diag/samplesize_analysis.json",
            "adbench/results/power_summary.json",
            "search_repeats/search_repeats.json",
            "search_repeats/search_repeats_summary.json",
            "search_repeats/search_vs_random_final.json")
        + [("search_forensics/search_forensics.json [COMPACTED]", forensics_compact())],
    ))

    # S5 — leverage: top-k pooling + label-free selection
    segments.append((
        "S5 results — inference-time scoring policy",
        paper_all["5_leverage"],
        evf("fusion_multiseed/fusion_multiseed_summary.json",
            "fusion_ksweep/fusion_ksweep_summary.json",
            "pseudo_select/pseudo_multiseed.json",
            "policy_select/policy_select.json"),
    ))

    # S6 — positioning + conclusion
    segments.append((
        "S6 discussion + conclusions",
        paper_all["6_positioning"] + "\n\n" + paper_all["7_conclusion"],
        evf("anomalib_baselines/results.json",
            "adbench/results/analysis_p1_p3.json"),
    ))

    results = []
    for title, ptxt, ev in segments:
        print(f"--- segment: {title} (paper {len(ptxt)} chars, {len(ev)} evidence files)")
        try:
            res, raw = audit_segment(title, ptxt, ev)
        except Exception as e:
            res, raw = {"claims": [], "segment_verdict": "ERROR",
                        "notes": repr(e)[:500]}, str(e)
        res["_segment"] = title
        res["_raw"] = raw
        results.append(res)
        # save raw per-segment transcript (forensic trace)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:60]
        with open(f"{OUT_DIR}/seg_{slug}.md", "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n## PAPER TEXT (sent)\n\n{ptxt}\n\n")
            f.write("## EVIDENCE FILES SENT\n\n")
            for n, _ in ev:
                f.write(f"- {n}\n")
            f.write("\n## REVIEWER RAW RESPONSE\n\n" + raw + "\n")

    with open(f"{OUT_DIR}/_partial.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\nDONE. partial ->", f"{OUT_DIR}/_partial.json")


if __name__ == "__main__":
    main()
