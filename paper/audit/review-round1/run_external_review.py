"""Segmented external review of main_v2 via a large-context LLM.

The user's constraint: the reviewer model truncates when the whole manuscript is
pushed through in one call, so the paper is submitted in segments. Each call gets
(a) a compact shared preamble describing the paper and what is being asked, and
(b) one segment of the manuscript. Per-segment replies are then consolidated into
a single reviewer verdict by a final synthesis call that sees only the segment
reports (not the manuscript again).

Model: glm-5.3-flash. It is a reasoning model, so max_tokens must be generous:
reasoning tokens are billed inside max_tokens and `content` stays empty until
reasoning completes.
"""
import json
import os
import re
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
    "P2_related_work": f"{PAPER}/sections/_draft_v2/2_related_work.tex",
    "P3_method": f"{PAPER}/sections/_draft_v2/3_method.tex",
    "P4_diagnosis": f"{PAPER}/sections/_draft_v2/4_diagnosis.tex",
    "P5_leverage": f"{PAPER}/sections/_draft_v2/5_leverage.tex",
    "P6_positioning": f"{PAPER}/sections/_draft_v2/6_positioning.tex",
    "P7_conclusion": f"{PAPER}/sections/_draft_v2/7_conclusion.tex",
}

SYSTEM = (
    "You are a senior reviewer for a peer-reviewed applied-ML journal "
    "(Applied Sciences, MDPI). You are reviewing a manuscript submitted in "
    "segments because of context limits. Judge each segment on its own terms and "
    "flag anything that would draw a reviewer objection. Be specific and terse. "
    "Do not restate the manuscript. Do not pad. Never invent numbers that are not "
    "in the text you are given."
)

CONTEXT = """\
## Manuscript under review

**Title**: Diagnosing an Architecture Search on an Industrial Anomaly-Detection
Benchmark: What the Category Explains and What the Scoring Policy Recovers

**Genre**: benchmark / evaluation study, NOT a new-model paper. The paper's
contribution is a set of negative and diagnostic results about how the field
evaluates backbones for unsupervised anomaly detection on MVTec-AD.

**Setup**: 14 backbones x 15 MVTec-AD categories x up to 3 training seeds; a
reverse-distillation detector per cell; a proxy search signal that the pipeline
minimises. Target venue: Applied Sciences (MDPI), free-format first submission,
abstract <= 200 words, GenAI use disclosed in Materials and Methods.

**Three headline claims**
1. Category explains 53-98% of the AUROC variance; backbone 0.54-9.1%; and the
   category share is NOT capturable by design (per-category oracle upper bound
   below 1.5% of the variance).
2. The margin the field contests is 0.5 points, but a single-category single-run
   comparison has a minimum detectable difference of 2.98 points; 240 independent
   training runs per backbone would be needed on one category, or 532 categories
   if pairing across the 15 available.
3. The search axis is inert: 2 of 3 feature dimensions never reach the deployed
   model; the proxy's median span across well-behaved backbones is 5.7%; 2
   backbones silently return 0.0, so all 12 searches selected a collapse-prone
   backbone and scored 80.91% against 81.65% for random. The real leverage is in
   the scoring policy (top-k pooling): 82.80% -> 90.21% with no retraining, and a
   label-free policy selection recovers +7.09 of an 11.65-point oracle gap.

**What we want from you**: read the segment below and answer, in under 400 words:
1. Any factual/arithmetic error, unsupported claim, or internal contradiction.
2. Any place a reviewer would say "this is overstated" or "this needs a control".
3. Any missing citation or unclear definition.
4. A severity ranking (blocking / major / minor) of what you found.
If a segment is clean, say so in one line. Do not speculate about parts of the
manuscript you were not shown.

---

## SEGMENT: {name}
"""


def call(messages, max_tokens=MAX_TOKENS, retries=4):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            API_URL, data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"},
            method="POST",
        )
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
        fr = ch.get("finish_reason")
        if content:
            return content, body
        # reasoning models return empty content while still thinking
        last = f"empty content (finish_reason={fr}); retry with more budget"
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
        path = SRC[name]
        text = open(path, encoding="utf-8").read()
        prompt = CONTEXT.format(name=name) + "\n" + text
        print(f"[{name}] {len(prompt)} chars ...", flush=True)
        t0 = time.time()
        content, body = call([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ])
        dt = time.time() - t0
        usage = body.get("usage", {})
        print(f"    -> {len(content)} chars in {dt:.0f}s "
              f"(reasoning {usage.get('completion_tokens_details', {}).get('reasoning_tokens')})",
              flush=True)
        reports[name] = content
        with open(f"{OUT}/seg_{name}.md", "w", encoding="utf-8") as fh:
            fh.write(f"# Reviewer report — {name}\n\n"
                     f"_model: {MODEL}; prompt {len(prompt)} chars; "
                     f"{dt:.0f}s; usage {json.dumps(usage)}_\n\n{content}\n")
        with open(f"{OUT}/_seg_raw.json", "w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=1)

    print("segments done:", list(reports), flush=True)


if __name__ == "__main__":
    main()
