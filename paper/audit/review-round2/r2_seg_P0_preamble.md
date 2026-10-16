# Round-2 reviewer report — P0_preamble

_model: glm-5.3-flash; prompt 12130 chars; 146s; usage {"completion_tokens": 9698, "completion_tokens_details": {"reasoning_tokens": 9099}, "prompt_tokens": 3025, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 12723}_

**1. Round‑1 resolution.** Substantially resolved for what lives in this segment: the abstract's canonical values (0.5, 2.98, 5.7%, 0.54/95.32, twelve searches, 82.47, 95.6–99.6) match the declared canonical set; 12 stable + 2 zero-proxy = 14 backbones is consistent with 210 = 14×15; the B4 reframe is visible (fixed top‑k leads, the criterion "adds little"); the GenAI statement is self-consistent (AI wrote the plotting code, but no figure was *generated* by a generative model). One B1-class recurrence remains (see #2), and one M-round fix was not propagated to the abstract: "worse than random" still appears bare, without the verifiable-selection grounding.

**2. New/remaining numeric issues.**
- The abstract's top‑k endpoints imply 90.28 − 82.36 = **7.92**, while the canonical headline delta is **+7.5**. Either the endpoints or the delta is mis-scoped (seed‑42 vs three-seed mean?), and the abstract gives no scope for 82.36/90.28. This is exactly the drift class B1 targeted.
- The abstract says the ANOVA attributes variance of "**accuracy**"; the decomposition is on **AUROC**. Metric slip.

**3. Overstatement / phrasing.**
- "the searched feature dimension never reaches the detector, so two thirds of the space is deployment-inert": as written, "never reaches" entails 100% inert, not 2/3. The §3 code-path statement (B3) presumably reconciles this; the abstract compression should match it (e.g., quantisation/duplicate mapping).
- "larger than the whole spread of the architectures it ranks" is not checkable from this segment; ensure §4 states the spread figure it rests on.
- Terminology drift: abstract says "twelve **stable** backbones"; the canonical phrasing is "degenerate-free." Pick one.

**4. Severity.**
- 7.92 vs +7.5 endpoints/delta mismatch: **major** (escalate to blocking until reconciled under one scope — same class as B1, on a headline number).
- "accuracy"→"AUROC": **minor** (one word, but misdescribes the analysis).
- Abstract "worse than random" and the "never reaches/two-thirds" phrasing: **minor**, conditional on the body fixes being correct.
- Otherwise this segment is clean: abstract within the 200-word limit (~196–199 depending on hyphen counting); funding, contributions, ORCID, and code/data statements internally consistent; no missing control is assessable at preamble level.
