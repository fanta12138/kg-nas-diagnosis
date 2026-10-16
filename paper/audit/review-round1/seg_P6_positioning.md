# Reviewer report — P6_positioning

_model: glm-5.3-flash; prompt 7294 chars; 170s; usage {"completion_tokens": 10552, "completion_tokens_details": {"reasoning_tokens": 9855}, "prompt_tokens": 2134, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 12686}_

**1. Factual/arithmetic/internal-consistency issues**

- **Gap arithmetic.** Strongest published value you quote is 99.6%; 99.6 − 82.47 = **17.1 points**, not "on the order of 15." State ~17.
- **Tension with headline claim 3.** Here the search "selected" EfficientNet-B0 (rank 5, 82.47); claim 3 states all 12 searches selected the collapse-prone backbone scoring 80.91 (= ConvNeXt-Tiny, rank 11). Reconcile explicitly which search produced the deployed model, or reviewers will read it as contradiction.
- **Reproduction logic.** One backbone, one pair of runs, 0.07 apart, "confirms that the protocol is stable" — overgeneralized from n=1. If both runs used seed 42, the 0.07 gap measures GPU nondeterminism, not protocol stability. State the seeds.
- Verified clean: rank ordering matches the AUROC column; 83.34−80.91 = 2.43; 89.69/2.81 ≈ 32×; 210 = 14×15; 82.47−82.40 = 0.07.

**2. Overstated / needs a control**

- **Single-seed ranking (worst problem in the segment).** Table 1 is seed 42 only (caption says so), yet you present a 14-way ordering and call it "the ordering it establishes." Rank gaps of 0.08 (ranks 9–10) sit far below your own 2.98-point single-run MDD. Your setup allows up to 3 seeds — report mean±sd or drop the Rank column. This undercuts the table's central interpretive claim and hands reviewers an easy objection.
- **"First comparison of a full anomaly-detection search space under a single protocol."** Unsupported priority claim; scope it and cite the closest unified-protocol benchmarking work, or reviewers will supply counterexamples.
- **Missing control:** reproduce one published configuration under your harness (e.g., RD with its original WideResNet-50 teacher). Without one anchor, readers cannot tell whether the 13–17 point gap reflects the backbone space or a weak harness.

**3. Citations / definitions**

- Cited published numbers lack variant provenance: PaDiM's 97.9 is the WideResNet-50 variant; your harness is ResNet-18-based. Say which variant each number is.
- Verify the CFA, FastFlow, and Dinomaly value/venue pairings, and the "never published" claim for FastFlow.
- "Mean AUROC": image-level? over 15 categories? Define. Caption's "Student (M)" vs text's "deployed size" are inconsistent labels.
- The 0.54% backbone-variance figure: state the decomposition basis (which factors, cell-level vs per-category).

**4. Severity**

- **Blocking:** single-seed ranking, if seed dispersion cannot be added.
- **Major:** contradiction with claim 3; stability-from-one-run; "first" claim; missing harness-validation control.
- **Minor:** 15-vs-17.1 gap; variant provenance; AUROC and variance definitions.
