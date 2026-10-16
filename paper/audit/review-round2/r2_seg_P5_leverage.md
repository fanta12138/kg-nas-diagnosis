# Round-2 reviewer report — P5_leverage

_model: glm-5.3-flash; prompt 29228 chars; 218s; usage {"completion_tokens": 16576, "completion_tokens_details": {"reasoning_tokens": 15772}, "prompt_tokens": 7970, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 24546}_

**1. Round‑1 concern resolved?**
Yes, in substance. B4 is genuinely fixed: the fixed top‑k row is the bold baseline, the criterion's own contribution is reported as −0.40, and the text explicitly says the pooling is the recommendation. B1 discipline holds here (every delta is tagged by seed/split/script), the 82‑vs‑111 sets are separated, the cross‑script per‑category disagreement is disclosed in the Table 2 caption, and the pooling result is scoped to one backbone family. All Table 3 column means, the +7.49/−0.40 decomposition, 7.09/11.65 = 61%, the 0.65 span, and the 0.13/1.91/2.49 script deltas check out.

**2. New errors introduced**
- **The 111‑policy enumeration does not parse.** "Simplex grids at step 0.1 and step 0.2 crossed with all three reductions" gives 66+21 = 87 weight vectors × 3 reductions = 261, not 111. The 82 set gets an explicit decomposition (66+9+3+4); the 111 set needs the same, since Table 3 and E3/E9 rest on it. **Major.**
- **Systematic 0.01 mismatches** between stated deltas and the displayed values: 90.21−82.80 = 7.41 (stated 7.42); 82.39−82.28 = 0.11 (stated −0.12); 82.39−81.12 = 1.27 (stated 1.28); 89.96−81.90 = 8.06 (stated 8.07); 89.99−88.90 = 1.09 (stated 1.10); 85.81−88.62 = −2.81 (stated −2.82). Presumably computed pre‑rounding — say so, or recompute. The −0.40 and +7.49 means are unaffected. Minor.
- "Its four variants all land inside a 0.2‑point band that the fixed convention already occupies": the variant means span 88.59–88.78; fixed top‑k is 88.99, *above* the band. Reword. Minor.
- The "best single policy" row is captioned label‑free, but *which* policy is best is selected on test performance; label it a global oracle like the per‑category row. Minor.

**3. Remaining overstatement / missing control**
- "+7.5, the three‑seed central value": the full‑test three‑seed mean is 7.75 (median 7.78); +7.5 matches the 70/30‑split mean (7.49). Defensible as a conservative headline, but "three‑seed central value" is the wrong label for the full‑test figure — anchor it to the split, or say "≈+7.5–7.8".
- The "we do not average two implementations into a single figure" sentence sits immediately next to a three‑seed aggregate (82.51/90.26) that mixes scripts across seeds. Clarify that the prohibition covers same‑model cross‑script readings only.
- Max pooling reads 86.74 (Table 2) vs 86.84 (Table 1) — within disclosed script noise, but add it to the caption's reconciled pairs.
- "Every k between 0.5% and 20%" generalises from six grid points; the figure should show density.

**4. Severity:** one major (111 enumeration — reproducibility of the selection experiment), remainder minor. Not blocking.
