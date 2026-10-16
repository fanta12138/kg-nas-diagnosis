# Reviewer report — P7_conclusion

_model: glm-5.3-flash; prompt 8119 chars; 165s; usage {"completion_tokens": 10707, "completion_tokens_details": {"reasoning_tokens": 9962}, "prompt_tokens": 1900, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 12607}_

**1. Errors / contradictions**

- "Twelve independent runs selected a configuration prone to this collapse **every single time**" is followed by "The four runs whose results the manuscript originally reported chose a backbone that does not collapse." If the four are among the twelve, this is a direct contradiction; if they are additional (16 total), say so. As written it is unreadable, and it touches a headline result.
- "+7.92 points" vs. the headline 82.80 → 90.21, a difference of **7.41**. If 7.92 is a three-seed mean of a different contrast, label it; otherwise the numbers disagree.
- "Two thirds of the search space is spent on **a feature dimension** that cannot reach the deployed model" contradicts the abstract's "2 of 3 feature dimensions never reach the deployed model." One inert dimension vs. two.
- "**Eleven** tight well-behaved backbones": 14 − 2 collapsing = 12. The exclusion of one backbone is unexplained.
- 0.54% / 95.32% are quoted as *the* decomposition, but the abstract claims ranges (0.54–9.1% / 53–98%). Reconcile, or reviewers will ask which is correct.
- Typo: "dominated something else entirely" → "dominated **by**".
- 2.98/3.98 = 75% ✓; ≈6×0.5 ✓; 7.09/11.65 = 61% ✓; 14×15 = 210 ✓.

**2. Overstated / needs control**

- "An order of magnitude more than the architecture": the full 14-backbone spread is 3.98 points; +7.92 is ~2×, not 10×. State the architecture effect in AUROC points.
- "A statement about the benchmark rather than about our pipeline" and "bounds every architecture comparison made on it" contradict the limitations paragraph, which concedes the residual variance is protocol-dependent. Soften both.
- Search-worse-than-random rests on 0.74 points (80.91 vs. 81.65) — below the paper's own 2.98 MDD. The 1.2th-percentile claim is a different test; report the number of random draws and address the apparent double standard explicitly.
- "Any deployed reverse-distillation detector … expect to gain several points": universal claim from one teacher, one dataset, three seeds. Hedge.
- "The manuscript originally reported" — revision-history artifact; delete.
- A 1.96-point sd from three seeds (2 df) is treated as precise.

**3. Citations / definitions**

- "0.5-point margin that separates PatchCore from the strongest results reported since" — citation needed.
- p = 0.039: name the test/model; state how 2.98 follows from the quoted 3.30 and 1.96.

**4. Severity**

- **Blocking** only if the four runs are a subset of the twelve (contradiction); otherwise ambiguous-but-fixable.
- **Major**: 7.92 vs. 7.41; two-thirds vs. 2-of-3; eleven vs. twelve; point estimates vs. abstract ranges; "order of magnitude"; benchmark-universality overstatement; percentile-vs-MDD tension.
- **Minor**: typo, drafting artifact, PatchCore citation, test naming, seed-count wording, "expect several points" hedge.
