"""Round 2 segmented external review of main_v2 via glm-5.3-flash.

Same protocol as run_external_review.py (segmented calls + one synthesis), but:
  * the shared preamble is updated to the CORRECTED manuscript's claims, so the
    reviewer does not re-litigate stale wording;
  * the segment outputs are written with an r2_ prefix so Round 1 is preserved;
  * the preamble explicitly states which Round-1 blocking items were fixed and how,
    so the reviewer can judge whether the fix is adequate rather than re-reporting.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
API_KEY = os.environ["GLM_API_KEY"]
MODEL = "glm-5.3-flash"
MAX_TOKENS = 64000

PAPER = "G:/submission_KBS"
OUT = "G:/submission_KBS/review-stage"
SRC = {
    "P0_preamble": f"{PAPER}/main_v2.tex",
    "P1_introduction": f"{PAPER}/sections/_draft_v2/1_introduction.tex",
    "P3_method": f"{PAPER}/sections/_draft_v2/3_method.tex",
    "P4_diagnosis": f"{PAPER}/sections/_draft_v2/4_diagnosis.tex",
    "P5_leverage": f"{PAPER}/sections/_draft_v2/5_leverage.tex",
    "P6_positioning": f"{PAPER}/sections/_draft_v2/6_positioning.tex",
    "P7_conclusion": f"{PAPER}/sections/_draft_v2/7_conclusion.tex",
}

SYSTEM = (
    "You are a senior reviewer for a peer-reviewed applied-ML journal "
    "(Applied Sciences, MDPI). This is ROUND 2 of a review loop: you reviewed "
    "Round 1 and raised blocking and major items. The authors have now revised. "
    "Judge whether each round-1 concern is genuinely resolved, and flag any NEW "
    "error the revision introduced. Be specific and terse. Do not restate the "
    "manuscript. Do not pad. Never invent numbers that are not in the text."
)

CONTEXT = """\
## Manuscript under review (Round 2, revised)

**Title**: The Architecture Axis Is Not the Axis: A Diagnostic Study of Neural
Architecture Search and Score Fusion in Unsupervised Industrial Anomaly Detection

**Genre**: benchmark / evaluation study, NOT a new-model paper.

**Setup**: 14 backbones x 15 MVTec-AD categories x up to 3 training seeds; a
reverse-distillation detector per cell; a proxy search signal the pipeline
minimises. Target: Applied Sciences (MDPI), free-format first submission,
abstract <= 200 words (now 199), GenAI disclosed in Materials and Methods.

**Headline claims as now stated**
1. Category explains 95.32% of the AUROC variance, backbone 0.54%, interaction
   4.13%: a two-way fixed-effects ANOVA on the single-seed (seed-42) matrix,
   F(13,182)=1.844, p=0.039.
2. The contested margin is 0.5 points; the standard single-category comparison's
   minimum detectable difference is 2.98 points; 240 runs per backbone on one
   category, or 532 categories if pairing across the 15 available.
3. The search axis is inert: the searched feature dimension never reaches the
   deployed detector, so 2 of 3 values are deployment-equivalent duplicates; the
   proxy separates the 12 degenerate-free backbones by 5.7% of its median and
   returns exactly 0.0 for 2 backbones that the minimising rule then prefers; the
   12 recorded runs returned a collapse-prone backbone every time.
4. The leverage is the inference-time scoring policy. A fixed top-k pooling raises
   mean AUROC by about +7.5 points with no retraining; a label-free criterion that
   selects the policy per category does NOT improve on the fixed top-k
   (its own contribution is -0.40 points on average over three seeds), and the
   paper reports that decomposition explicitly.

**Round-1 blocking items and how they were addressed**
- B1 headline-number drift -> one canonical value per quantity, each defined by
  split/seed/script; the 111-vs-82 policy sets are now separated explicitly.
- B2 the twelve-run vs four-run record -> the two run populations are now stated
  separately, with the sampling-vs-selection distinction made explicit.
- B3 inertness stated inconsistently -> now stated once, as the actual code path.
- B4 the label-free claim was framed against the weakest baseline -> rewritten to
  lead with the fixed top-k baseline; the criterion's own contribution is now the
  reported figure (-0.40), and the paper says the useful part is the pooling.
- B5 teacher scope undeclared -> the method section now distinguishes the
  ImageNet-1k teacher used by the DEPLOYED detector from the contrastively
  pre-trained teacher behind the search proxy, and calls it a deviation.
- M1-M12 -> variance-decomposition scope stated; Eq. (1) rewritten with the skip
  indicator; degenerate-filter control added; "worse than random" rephrased to
  rest on the verifiable selection facts; Table-1 seed scope stated; the
  cross-script per-category discrepancy (up to 2.49 points) disclosed in the table
  caption; transfer correlation given a robustness control; scope of the pooling
  result bounded to the one backbone family tested; GenAI disclosure made
  self-consistent.

**What we want from you**: read the segment below and answer, in under 400 words:
1. Is the round-1 concern for this segment genuinely resolved? If not, what
   remains?
2. Any NEW factual/arithmetic error or internal contradiction introduced?
3. Any remaining overstated claim or missing control?
4. Severity (blocking / major / minor).
If a segment is now clean, say so in one line. Do not speculate about parts you
were not shown.

---

## SEGMENT: {name}
"""


def call(messages, max_tokens=MAX_TOKENS, retries=4):
    payload = {"model": MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": 0.2}
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            API_URL, data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:400]}"
            time.sleep(5 * attempt)
            continue
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(5 * attempt)
            continue
        ch = body["choices"][0]
        content = (ch["message"].get("content") or "").strip()
        if content:
            return content, body
        last = f"empty content (finish_reason={ch.get('finish_reason')})"
        max_tokens = min(int(max_tokens * 1.5), 200000)
        payload["max_tokens"] = max_tokens
        data = json.dumps(payload).encode()
        time.sleep(2)
    raise RuntimeError(f"call failed after {retries} attempts: {last}")


def main():
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] if len(sys.argv) > 1 else list(SRC)
    reports = {}
    for name in only:
        text = open(SRC[name], encoding="utf-8").read()
        prompt = CONTEXT.format(name=name) + "\n" + text
        print(f"[{name}] {len(prompt)} chars ...", flush=True)
        t0 = time.time()
        content, body = call([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}])
        dt = time.time() - t0
        usage = body.get("usage", {})
        print(f"    -> {len(content)} chars in {dt:.0f}s", flush=True)
        reports[name] = content
        with open(f"{OUT}/r2_seg_{name}.md", "w", encoding="utf-8") as fh:
            fh.write(f"# Round-2 reviewer report — {name}\n\n"
                     f"_model: {MODEL}; prompt {len(prompt)} chars; {dt:.0f}s; "
                     f"usage {json.dumps(usage)}_\n\n{content}\n")
        with open(f"{OUT}/_r2_seg_raw.json", "w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=1)
    print("round-2 segments done:", list(reports), flush=True)


if __name__ == "__main__":
    main()
