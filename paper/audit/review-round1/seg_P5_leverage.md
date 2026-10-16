# Reviewer report — P5_leverage

_model: glm-5.3-flash; prompt 20436 chars; 239s; usage {"completion_tokens": 14496, "completion_tokens_details": {"reasoning_tokens": 13679}, "prompt_tokens": 5630, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 20126}_

**1. Factual/arithmetic errors and contradictions**

- **Policy count is wrong.** Text says 82 policies; the stated grid is 66 (top-k) + 21 (mean) + 21 (max) + uniform variants = 109+, and Table 3 says **111** candidate policies (66+42+3 exactly). "82" contradicts both its own decomposition and Table 3.
- **Deltas not reproducible from displayed values:** 90.21−82.80=7.41 (quoted 7.42); 89.96−81.90=8.06 (quoted 8.07); 82.39−82.28=0.11 (quoted −0.12); 82.39−81.12=1.27 (quoted −1.28). Say "computed unrounded" or fix.
- **§5.3 baselines are orphans.** Default 82.39 and oracle 93.39 match neither the full-training seed-42 values (82.80/93.72) nor Table 3's 70%-split seed-42 values (81.62/93.19). Split/script/model unstated; this is a fourth default measurement, contradicting the "measured three times" noise-floor paragraph.
- **Unacknowledged cross-script tail discrepancy.** Same policy, same models: worst category 78.37 (Table 1) vs 76.46 (Table 2), Δ=1.91 — 15× the 0.13 agreement the caption claims, and the k=1–2% worst-case recommendation depends on which script is believed. The max row also shifts (86.84→86.74).

**2. Overstatement / missing controls**

- **The label-free claim is benchmarked against the weakest baseline.** Uniform top-k with *no selection at all* — itself label-free — scores 90.21, above the selection procedure's 88.59. Against that honest baseline the "recovery" is negative on average; only the labeled oracle beats static top-k (93.72, +3.5). Reframe or show per-category value where top-k hurts. Blocking for the headline.
- **"The failure is produced by the reduction step"** is demonstrated on one backbone, one detector family. Overturning the manuscript's paradigm-level attribution of the screw failure requires top-k results across the 14 backbones.
- **Same-seed rerun differing by 0.40** (82.80 vs 82.40) implies training nondeterminism — unstated, and that spread is the scale of seed effects interpreted elsewhere.
- Source comment marks pseudo_select (E9) "pending" while §5.4 reports final three-seed numbers; clarify provenance.

**3. Citations/definitions**

- No citations for pooling: top-k aggregation of patch distances is standard (SPADE, PatchCore, CFA); synthetic-anomaly self-supervision precedent exists (CutPaste, NSA). For an evaluation paper this invites "known result" objections.
- Undefined: map alignment/resolution before the three-scale weighted sum; k as a fraction of what; how a policy "top-5 ensemble" is computed; how the 15 perturbation variants aggregate; holdout size; why mean/max get a 0.2 grid while top-k gets 0.1.

**4. Severity**

- **Blocking:** 82-vs-111 policy count; missing uniform-top-k baseline under the label-free headline.
- **Major:** tail script discrepancy; 82.39/93.39 provenance; same-seed nondeterminism; screw causal claim; missing pooling citations.
- **Minor:** delta rounding; undefined mechanics; stale E9 comment; max = top-1 redundancy; Table 1 omits worst-category cells for two rows despite that column being the stated deployment rationale.
