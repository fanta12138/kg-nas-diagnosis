# Round-2 reviewer report — P1_introduction

_model: glm-5.3-flash; prompt 16103 chars; 212s; usage {"completion_tokens": 24123, "completion_tokens_details": {"reasoning_tokens": 23402}, "prompt_tokens": 3895, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 28018}_

**1. Is the round-1 concern resolved?**

- **B4 (label-free framing): resolved.** The segment leads with fixed top-k (88.99), reports the selector's own −0.40, and credits the pooling.
- **B2/B3: mostly resolved**, with residues flagged below (unstated n; self-contradicting code-path sentence).
- **B1 (canonical values): NOT resolved.** The policy gain now appears as three irreconcilable values: +7.8 (82.5→90.3, run-in head), "≈7.5" (Contribution 5), and the segment's own arithmetic 88.99 − 82.47 ≈ +6.5 for fixed top-k. Nothing in the segment defines a quantity equal to 7.5, and the reduction producing 90.3 is never identified (which policy, which script/seed set). The same stale 7.5 also appears in your own summary as a fixed-top-k figure, contradicting 88.99 − 82.47.

**2. New errors / contradictions introduced**

- The inertness sentence is self-contradictory: "the decoder is constructed with the requested one" cannot coexist with "the final training entry point never passes the argument at all." Read literally, the first clause defeats the "identical deployed model"/28-duplicates conclusion. State which stage's decoder receives the width.
- Contribution 1 misattributes 2.98 to a "single-category, single-run comparison"; the body derives 2.98 for the **fifteen-category paired** test. The single-category single-run MDD is far larger (240 runs/backbone to resolve 0.5).
- Dataset counts: "a few hundred images across all fifteen categories" contradicts your own mean (115 × 15 ≈ 1,725), and the 42–167 range disagrees with the cited dataset's published test counts (leather alone has 229). Verify.

**3. Remaining overstatement / missing control**

- "Worse than picking a backbone uniformly at random" sits badly beside "ranks fifth of fourteen": with a 3.98-point spread, a rank-5 mean should exceed the uniform-pick expectation. Specify the metric and baseline population, give *n* for the independent-seed runs, and reconcile with the 8.0%-of-834 record rate (degenerate filter? different population?).
- "Most published backbone comparisons are not measurements" / "about the field as a whole" extrapolates one detector family's variance (1.96 / 3.30) to other methods. Hedge or add cross-method variance evidence.
- Verify "the published pipeline averages the discrepancy map over all pixels" — several RD implementations use max; if 82.5 is your pipeline's average, attribute it as such.
- Minor: "disagree/reverse" (31%) undefined; 93.72 not labelled as an oracle/label-using selection.

**4. Severity**

- **Blocking:** the 7.8 / 7.5 / 6.5 cluster (B1 residue on the headline gain).
- **Major:** inertness sentence contradiction; Contribution 1 protocol mislabel; "worse than random" vs. rank-5 tension.
- **Minor:** dataset counts; field-wide extrapolation; averaging claim; 31% definition; oracle labelling.
