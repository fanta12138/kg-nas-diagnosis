# Reviewer report — P2_related_work

_model: glm-5.3-flash; prompt 12718 chars; 197s; usage {"completion_tokens": 13929, "completion_tokens_details": {"reasoning_tokens": 13190}, "prompt_tokens": 3102, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 17031}_

**1. Factual/arithmetic errors and contradictions**

- **Metric conflation.** The section attributes the variance decomposition to "accuracy" (0.54% backbone, 95.32% category), but the paper's headline decomposition (category 53–98%, backbone 0.54–9.1%) is over AUROC. Either fix the metric name or state explicitly that two decompositions exist. Worse: 0.54% is the *minimum* of the 0.54–9.1% backbone range; quoting it as "the" backbone share cherry-picks the most damaging value. Report the range or the overall decomposition.
- **Backbone vs. configuration drift.** "One of the two configurations prone to a degenerate evaluation" vs. the setup's "2 backbones silently return 0.0." If the space is backbones × dims, every configuration on those backbones degenerates; "two configurations" is wrong.
- **Taxonomy error.** SimpleNet perturbs features; EfficientAD is student–teacher + autoencoder; Dinomaly is masked reconstruction. Not all "inject synthetic defects"; Dinomaly at minimum is misclassified.

**2. Overstated / needs a control**

- "Several times more accuracy than the architecture" (§2.1): quantify in points and qualify — the backbone share reaches 9.1% on some categories, so the blanket comparative claim invites objection.
- Spearman ρ = −0.12 "over all fifteen categories" is undefined: pooled or averaged per category? Pooled is confounded by category effects (95% of variance). The two zero-returning backbones also create ties, and their proxy rank (last) vs. true rank (mid) alone can drive ρ toward zero/negative. **Report ρ on the 12 well-behaved backbones only** — that is the control that makes "measuring something, but not the thing" hold.
- "Same order as its own run-to-run noise": state the noise magnitude, not just the 5.7% span.
- "Work in this line takes for granted…" generalizes from a single citation (Qian et al.).
- "Neither IM-IAD nor the time-series reproductions reports that quantity": universal negative; verify or soften.

**3. Missing citations / unclear definitions**

- No MVTec-AD dataset citation in this section.
- The NAS-evaluation line rests on Yu et al. alone; add e.g. NAS-Bench-101 and random-search reproducibility work (Li & Talwalkar).
- "Usually left implicit" (map→score step): support or soften; cite prior work on anomaly-map aggregation, if any exists.
- "1.2th percentile": specify the distribution (configurations? backbones?), sample count, and how percentile is defined on a discrete space.
- "Well-behaved" and "every validation batch is rejected" undefined at first use; "surveys" plural with one citation.
- yu2026eventlevel is 2026-dated; verify in-press status or it will draw an objection.
- "210 detectors … fully crossed" vs. "up to 3 seeds": state whether the crossed design is seed-balanced; imbalance matters for the variance decomposition.

**4. Severity**

- Blocking: none in this segment.
- Major: metric conflation + cherry-picked 0.54%; ρ construction/confound; unqualified "several times" claim.
- Minor: everything else above.
