import os
"""Synthesis call: consolidate the 8 per-segment reviewer reports into ONE verdict.

Sees only the segment reports (not the manuscript). The manuscript text is far too
large for one call, and the constraint set by the PI is one segment per call.
"""
import json, os, time, urllib.error, urllib.request

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
API_KEY = os.environ["GLM_API_KEY"]
MODEL = "glm-5.3-flash"
MAX_TOKENS = 40000
OUT = "G:/submission_KBS/review-stage"

SYSTEM = (
    "You are a senior reviewer for a peer-reviewed applied-ML journal "
    "(Applied Sciences, MDPI). You previously reviewed a manuscript in eight "
    "segments. You now consolidate the eight segment reports into ONE verdict. "
    "Be terse. Never invent numbers. Do not restate the manuscript."
)

PROMPT = """\
## Manuscript

**Title**: The Architecture Axis Is Not the Axis: A Diagnostic Study of Neural
Architecture Search and Score Fusion in Unsupervised Industrial Anomaly Detection

**Genre**: benchmark / evaluation study. Its contribution is a set of negative and
diagnostic results about how the field evaluates backbones for unsupervised
anomaly detection on MVTec-AD. This matters for how you weight the criticisms: the
paper does NOT claim a competitive detector, so a weak absolute AUROC is disclosed
in the text, not concealed.

**Setup**: 14 backbones x 15 MVTec-AD categories x up to 3 training seeds; a
reverse-distillation detector per cell; a proxy search signal that the pipeline
minimises. Target: Applied Sciences (MDPI), free-format first submission, abstract
<= 200 words, GenAI use disclosed in the Materials and Methods, no length limit.

**Three headline claims**
1. Category explains 53-98% of the AUROC variance; backbone 0.54-9.1%; the
   category share is not capturable by design (per-category oracle upper bound
   under 1.5% of the variance).
2. The field contests 0.5 AUROC points, but a paired comparison over the 15
   categories has a minimum detectable difference of 2.98 points; 240 independent
   training runs per backbone would be needed on one category, or 532 categories.
3. The search axis is inert: 2 of 3 feature dimensions never reach the deployed
   model; the proxy's median span across well-behaved backbones is 5.7%; 2
   backbones silently return 0.0, so all 12 searches selected a collapse-prone
   backbone (80.91% vs 81.65% for random). The real leverage is the scoring
   policy: top-k pooling gives +7.4 to +8.1 points with no retraining, and
   label-free policy selection recovers 61% of the oracle gap.

---

## The eight segment reports

{reports}

---

## Your task

The author has already checked many of these criticisms against the released code
and evidence files and found some of them to be false positives — for example the
criticisms that the policy count should be 111 rather than 82 (82 is verified
exactly: 66 top-k simplex at step 0.1, plus 9 single-scale, plus 3 uniform-sum,
plus 4 normalised-scale policies), and that the 1.2th-percentile figure is
inconsistent with the 2.98-point MDD (they are two different tests on two
different distributions). Judge the reports accordingly: do not simply aggregate
them. Weight by whether a criticism is verifiable and material to a headline claim.

Answer, concisely:

1. **Score** this manuscript 1-10 for Applied Sciences (be calibrated: a solid,
   honest, well-evidenced diagnostic study with several remaining expositional
   defects is a 7; reserve 9-10 for work with no material objection left).
2. **Verdict**: exactly one of `ready`, `almost`, `not ready`. "ready" means you
   would not ask for any change before acceptance; "almost" means minor fixes
   only; "not ready" means something material must change first.
3. **Remaining critical weaknesses**, ranked by severity, each with the MINIMUM
   fix. Merge duplicates across segments into one item. Mark each as either
   `blocking`, `major` or `minor`.
4. For each of `blocking` and `major` items only: is it a `text` fix (rewording,
   adding a caveat, adding a citation) or does it need `new computation` (a new
   experiment, a new statistic)? The author can do text fixes immediately and can
   also run new experiments, but wants to know which is which.
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
                     "Authorization": f"Bearer {API_KEY}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:400]}"
            time.sleep(5 * attempt); continue
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(5 * attempt); continue
        ch = body["choices"][0]
        content = (ch["message"].get("content") or "").strip()
        if content:
            return content, body
        last = f"empty content (finish_reason={ch.get('finish_reason')})"
        max_tokens = min(int(max_tokens * 1.5), 200000)
        payload["max_tokens"] = max_tokens
        data = json.dumps(payload).encode()
        time.sleep(2)
    raise RuntimeError(f"call failed: {last}")


def main():
    parts = []
    for name in ["P0_preamble", "P1_introduction", "P2_related_work", "P3_method",
                 "P4_diagnosis", "P5_leverage", "P6_positioning", "P7_conclusion"]:
        txt = open(f"{OUT}/seg_{name}.md", encoding="utf-8").read()
        body = txt.split("\n\n", 2)[-1]
        parts.append(f"### SEGMENT REPORT: {name}\n\n{body.strip()}\n")
    reports = "\n".join(parts)
    prompt = PROMPT.format(reports=reports)
    print(f"synthesis prompt {len(prompt)} chars", flush=True)
    t0 = time.time()
    content, body = call([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}])
    dt = time.time() - t0
    print(f"-> {len(content)} chars in {dt:.0f}s", flush=True)
    with open(f"{OUT}/seg_SYNTHESIS.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Consolidated reviewer verdict\n\n"
                 f"_model: {MODEL}; {dt:.0f}s; usage {json.dumps(body.get('usage', {}))}_\n\n"
                 f"{content}\n")
    json.dump({"synthesis": content}, open(f"{OUT}/_synth_raw.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
