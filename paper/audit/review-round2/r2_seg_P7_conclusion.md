# Round-2 reviewer report — P7_conclusion

_model: glm-5.3-flash; prompt 10176 chars; 218s; usage {"completion_tokens": 12758, "completion_tokens_details": {"reasoning_tokens": 12073}, "prompt_tokens": 2407, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 15165}_

**1. Round-1 concerns in this segment.** Substantially resolved. The two run populations (12 vs. 4) are stated separately with the luck/selection reading (B2 ✓); inertness is stated once, via the code path (B3 ✓); the pooling finding now leads with the fixed top-k baseline and reports the criterion's own contribution, −0.40 (B4 ✓). Verified arithmetic: 2.98/3.98 = 75%; 2.98/0.5 ≈ 6×; 95.32/0.54/p = 0.039 consistent; 210 = 14×15; mean of (+1.10, +0.52, −2.82) = −0.40; 88.99 − 81.50 = +7.49 ≈ +7.5. B1 is **not** fully clean — see 2(a).

**2. New errors / contradictions.**
- **(a) Resolvability count drift (B1 residue).** Conclusion: "at least 241 … or 533 categories"; the canonical statement is 240 and 532. Note 15·(2.98/0.5)² = 532.8, so 533 (ceiling) is defensible — but then 532 elsewhere is wrong, and vice versa. Exactly the drift B1 was meant to eliminate. Pick one value, propagate.
- **(b) "One teacher."** The limitations say the diagnosis used "one teacher," contradicting the two-teacher distinction added to fix B5 (ImageNet-1k deployed teacher vs. contrastive proxy teacher). Say "one teacher per track."
- **(c) "An order of magnitude more than the architecture."** The only architecture figure in that paragraph is the 3.98-point spread; +7.5 is ~1.9× that, not 10×. If the intended comparator is the backbone sd or the 0.54% variance share, name it; otherwise drop the phrase.

**3. Remaining overstatement.**
- "Most of that is available from a fixed change of pooling": per your own numbers the fixed change *is* the +7.5 (the criterion nets −0.40 on top). Attribute +7.5 directly to fixed pooling — the next paragraph already does; this sentence implies a larger decomposition that doesn't exist.
- Practice paragraph: "any deployed reverse-distillation detector … expect to gain several points" generalizes past the one-backbone-family scope you enforce elsewhere (M-item). Hedge to the tested family or mark as extrapolation.
- Minor: "whose own three-seed spread exceeds its mean" is trivially true since the mean is negative; say "exceeds its |mean|."

The "in most trials" hedge on the 0.0-returns is a precision improvement, not a contradiction.

**4. Severity.** 2(a): **major** (trivial one-line fix, but it is B1's exact failure mode in the concluding headline numbers — hold acceptance on it). 2(b), 2(c), 3(i), 3(ii): minor. Otherwise this segment is close to clean.
