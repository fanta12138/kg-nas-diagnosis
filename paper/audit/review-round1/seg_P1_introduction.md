# Reviewer report — P1_introduction

_model: glm-5.3-flash; prompt 14030 chars; 144s; usage {"completion_tokens": 9801, "completion_tokens_details": {"reasoning_tokens": 9065}, "prompt_tokens": 3409, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 13210}_

**Blocking**

1. The 2.98-point MDD is attributed inconsistently within the segment. The runinhead paragraph says the *fifteen-category paired test* has MDD 2.98 and that 240 runs/category are needed to resolve 0.5 on one category; Contribution 1 says the *single-category, single-run* comparison resolves 2.98. Both cannot hold: if single-run single-category MDD were 2.98, resolving 0.5 would need ≈36 runs, not 240; 240 runs implies a single-run MDD near 7.7. The body's scaling is the one consistent with the 532-category pairing figure; fix Contribution 1 (and matching abstract text) or the body.
2. Policy-sweep numbers contradict the headline results: intro says 82.5→90.3 (+7.8); the headline numbers are 82.80→90.21 (+7.41). Also reconcile the Scope's 82.47% with whichever baseline survives.

**Major**

3. Citation misattribution: SimpleNet and STPM are not NAS; they do not support "applying neural architecture search to anomaly detection." Cite actual NAS-for-AD work or soften the claim.
4. "Worse than picking uniformly at random" rests on 80.91 vs 81.65 — a 0.74-point difference, below the paper's own 2.98 resolvability floor. State the number of runs, the random baseline's variance, and a significance test, or the claim is self-undermining.
5. Field-wide generalization overreaches: the noise floor (1.96/3.30) is measured for one detector family on one benchmark; "most published backbone comparisons on this benchmark are not measurements" requires evidence that variance estimates transfer across detectors.
6. The inert-dimension mechanism is self-contradictory as written: "the decoder is constructed with the requested one" yet the three values "produce the identical deployed model." Clarify the actual code path.
7. Non-transfer (ρ̄ = 0.215; 31% discordant): no significance test and no correction for attenuation, although the measured within-cell noise mechanically shrinks ρ; the 31% figure is also not compared to the chance level for discordant pairs.
8. Collapse-selection control: 0.0 is returned only 73%/13% of the time for the two backbones, yet both are "selected every time." Show selection survives excluding collapsed evaluations. Also reconcile "four independent runs" here with the 12 searches claimed in the headline results.
9. Decomposition basis unclear: 210 detectors vs. three seeds — did the two-way decomposition use per-cell means? Name the test behind p = 0.039.

**Minor**

10. "2026 state of the art": the cited SOTA (guo2025dinomaly) is 2025. Fix the year or cite a 2026 result.
11. No MVTec-AD dataset citation despite extensive discussion.
12. "Widely deployed" (reverse distillation) is unsupported; verify "averages over all pixels… never justified" against deng2022's actual scoring before asserting it.
13. "The same order as the within-configuration noise" — no number given in this section.
14. "Roughly a hundred images per category" — test-set sizes vary severalfold across the 15 categories; give a mean or range.
