# Reviewer report — P4_diagnosis

_model: glm-5.3-flash; prompt 31014 chars; 402s; usage {"completion_tokens": 23189, "completion_tokens_details": {"reasoning_tokens": 22393}, "prompt_tokens": 8285, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 31474}_

**Blocking**

1. §4.5, the "1.2th percentile / worse than 98.8% of uniform draws" claim is computed against the wrong distribution. The bootstrap s.d. (0.32) is over category resampling of the *mean* random-selection AUROC; a single random draw varies by the spread of backbone means (~3.98 points). By the paper's own numbers (80.91 vs ranks 12/13/14 at ≤80.43), only ~74–79% of draws beat the search — and the reported mean rank of 10.3/14 implies the same. Restate as a test against the expected value or drop the percentile.
2. §4.5 internal contradiction: "only 2 of the 14 backbones and 6 of the 42 configurations appeared at all" — yet the same paragraph reports efficientnet_b0 with 79 trials. Define "appeared" (selected?), and report recorded-trial counts per run; 6/42 coverage under stated uniform sampling needs the failure-vanishing mechanism quantified.

**Major**

3. Table 2 row inconsistency: median span 7.540–7.971 is attributed to "twelve well-behaved backbones," but b0's median is 12.1; the CV span (2.2–74.0) includes b0, the median span excludes it. The 5.7% headline rests on this eleven-vs-twelve slippage — define "well-behaved" once and use it consistently.
4. No multiple-comparison control: 13 backbone contrasts (p = 0.013), 10 band tests (p = 0.058). Any correction changes these conclusions.
5. §4.3 framing: the searched axis is D; the capacity-band experiment varies backbone size (one backbone per band, family confounded with capacity) and tests neither axis cleanly. Say the 14×15 matrix carries the claim.
6. "Unreachable by construction" rests on code reading. Add the empirical control: wire the argument through, train D=128 vs 512, show deployed AUROC unchanged.
7. Statistical consistency: √2·σ_res = 4.67, but σ_Δ = 4.12 — explain. Also "adding two more training seeds per cell, as we do here" is inaccurate: only 90/210 cells are replicated.
8. ResNet-18 is 83.25 (Table 4) but "the best backbone in the space gives 83.34%"; the "+2.4 points" uses the latter. Reconcile.
9. "A quantity that does not exist" overstates: mean ρ = 0.215 over 105 pairs is significantly above zero; there is a weak general factor.

**Minor**

- Rounding errors: weighted mean of 80.43/81.85 = 80.90, not 80.91; 95.32/0.54 = 176.5, not 175; 83.08−82.01 = 1.07, not 1.06; 7.56/0.110 = 68.7×; the 240-run figure computes to 241; 532 to 533.
- Four runs "on the same space," one named "15backbones."
- b0 "degenerate-free" (420 evals) vs two zeros in search runs — explain the setting difference; state epochs for the 420 evaluations; define "140 backbone comparisons" and the pooling of median/CV.
- If not cited earlier: MVTec-AD dataset and reverse distillation. "Roughly a hundred images per category" — report actual per-category counts.
