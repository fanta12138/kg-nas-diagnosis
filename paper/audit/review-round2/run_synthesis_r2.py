import os
"""Round-2 synthesis call: consolidate the 7 per-segment Round-2 reports into ONE verdict.

Sees only the Round-2 segment reports (not the manuscript). The prompt tells the
reviewer which Round-1 items were addressed, so it judges the fixes rather than
re-litigating stale wording.
"""
import json, time, urllib.error, urllib.request

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
API_KEY = os.environ["GLM_API_KEY"]
MODEL = "glm-5.3-flash"
MAX_TOKENS = 40000
OUT = "G:/submission_KBS/review-stage"

SYSTEM = (
    "You are a senior reviewer for a peer-reviewed applied-ML journal "
    "(Applied Sciences, MDPI). You reviewed a manuscript in segments in round 1 "
    "and again in round 2. You now consolidate the SEVEN round-2 segment reports "
    "into ONE round-2 verdict, and state explicitly which round-1 blocking items "
    "you now consider resolved. Be terse. Never invent numbers. Do not restate the "
    "manuscript."
)

PROMPT = """\
## Manuscript

**Title**: The Architecture Axis Is Not the Axis: A Diagnostic Study of Neural
Architecture Search and Score Fusion in Unsupervised Industrial Anomaly Detection

**Genre**: benchmark / evaluation study. Its contribution is a set of negative and
diagnostic results about how the field evaluates backbones for unsupervised
anomaly detection on MVTec-AD. The paper does NOT claim a competitive detector, so
a weak absolute AUROC is disclosed in the text, not concealed.

**Setup**: 14 backbones x 15 MVTec-AD categories x up to 3 training seeds; a
reverse-distillation detector per cell; a proxy search signal that the pipeline
minimises. Target: Applied Sciences (MDPI), free-format first submission, abstract
<= 200 words, GenAI use disclosed, no length limit.

## Context: this is round 2

In round 1 you returned **6/10, verdict `not ready`**, with blocking items B1-B5 and
major items M1-M12. Between rounds the author (i) triaged every round-1 finding
against the released code and data, rejecting 12 as false positives with evidence,
and (ii) added four new released control computations
(`pseudo_select/baseline_decomposition.py`, `search_repeats/degenerate_filter_control.py`,
`proxy_diag/rho_wellbehaved_control.py`, `proxy_diag/rank_stability_control.py`), and
then rewrote the affected text. The round-2 reports you are given judge whether those
fixes hold. Several round-2 reports identify NEW defects introduced by the fixes
themselves; those are what you must now weigh.

---

## The seven round-2 segment reports

{reports}

---

## Your task

Judge the round-2 reports, weighting by whether a criticism is verifiable, material
to a headline claim, and newly introduced (versus a stale round-1 item already fixed).

Answer, concisely:

1. **Round-1 blocking items B1-B5**: for each, state `resolved` or `unresolved`, in
   one clause, with the evidence from the round-2 reports.
2. **Score** this manuscript 1-10 for Applied Sciences, calibrated the same way as
   in round 1 (a solid, honest, well-evidenced diagnostic study with only
   expositional defects left is a 7; 9-10 is reserved for no material objection).
3. **Verdict**: exactly one of `ready`, `almost`, `not ready`.
4. **Remaining critical weaknesses**, ranked, each with the MINIMUM fix and marked
   `blocking`, `major` or `minor`. Merge duplicates. Be strict about whether a
   remaining item is genuinely blocking or merely cosmetic.
5. For each `blocking` and `major` item only: `text` fix or `new computation`?
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
    for name in ["P0_preamble", "P1_introduction", "P3_method", "P4_diagnosis",
                 "P5_leverage", "P6_positioning", "P7_conclusion"]:
        txt = open(f"{OUT}/r2_seg_{name}.md", encoding="utf-8").read()
        body = txt.split("\n\n", 2)[-1]
        parts.append(f"### ROUND-2 SEGMENT REPORT: {name}\n\n{body.strip()}\n")
    reports = "\n".join(parts)
    prompt = PROMPT.format(reports=reports)
    print(f"synthesis prompt {len(prompt)} chars", flush=True)
    t0 = time.time()
    content, body = call([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}])
    dt = time.time() - t0
    print(f"-> {len(content)} chars in {dt:.0f}s", flush=True)
    with open(f"{OUT}/r2_seg_SYNTHESIS.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Round-2 consolidated reviewer verdict\n\n"
                 f"_model: {MODEL}; {dt:.0f}s; usage {json.dumps(body.get('usage', {}))}_\n\n"
                 f"{content}\n")
    json.dump({"synthesis": content},
              open(f"{OUT}/_r2_synth_raw.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
