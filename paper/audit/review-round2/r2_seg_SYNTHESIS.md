# Round-2 consolidated reviewer verdict

_model: glm-5.3-flash; 91s; usage {"completion_tokens": 5384, "completion_tokens_details": {"reasoning_tokens": 4237}, "prompt_tokens": 5436, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 10820}_

## 1. Round-1 blocking items

- **B1 — unresolved.** Headline/canonical values still drift across segments: 7.92 vs +7.5 vs +7.8 vs ≈+6.5 for the policy gain (P0/P1), 240/532 vs 241/533 (P7), 82.47 vs 82.40 unnamed (P6), plus P4/P5 rounding drifts — though B1 discipline holds locally in P5.
- **B2 — resolved.** The 12-vs-4 run populations are stated separately with the selection-rule attribution (P4, P7).
- **B3 — resolved.** One coherent code-path "unreachable" account now exists (P4); a *new* self-contradictory inertness sentence in P1 must be repaired but the underlying fix holds.
- **B4 — resolved.** Fixed top-k (88.99) leads, the criterion's −0.40 is reported, pooling is the recommendation (P1, P5, P7).
- **B5 — resolved.** Two teachers distinguished, deviation declared and wired to resolvability (P3); the P3 cross-reference and P7 "one teacher" wording are new defects, not a failure of the fix.

## 2. Score

**7/10.** The substance survived a hostile round-2 audit: all four new controls verified, arithmetic in P4/P6/P7 checks out, no claim rests on a computation that must be redone. What remains is numeric-consistency and wording — expositional — but the B1 residue sits on headline numbers.

## 3. Verdict

`almost`

## 4. Remaining critical weaknesses (ranked)

1. **Canonical-value sweep** — `blocking`. Merge: policy-gain cluster (abstract endpoints 7.92; Contribution-5 ≈7.5; P1 head 7.8; fixed-top-k ≈6.5; true figures 7.75 full-test / 7.49 split / +7.49−0.40), 240/532 vs 241/533, unnamed 82.47-vs-82.40, P4 (+1.06/2.4/175) and P5 (six 0.01) drifts. Minimum fix: one scope per quantity, stated at first use and propagated; all deltas tagged pre-rounding or recomputed.
2. **Mischaracterised control descriptions** — `major`. Filter-control "middle-or-above" is contradicted by rank-12 efficientnet_b1 selections; capacity-band "restricted to backbones inside the band" by out-of-band argmins. Fix: restate as selection-stage restriction; keep only the defensible no-zero-artefact claim.
3. **111-policy set undefined** — `major`. 87×3=261 ≠ 111; Table 3 and E3/E9 rest on it. Fix: decompose 111 as was done for 82.
4. **"No repetition" vs 100 draws with replacement** — `major`. Fix: state determinism, search seed, and whether re-draws retrain.
5. **Teacher-identity contradictions** — `major`. P3 cross-reference re-blurs proxy vs deployed teacher; P7 "one teacher." Fix: correct cross-reference; "one teacher per track."
6. **Protocol misattribution of 2.98** — `major`. Contribution 1 and headline 2 call it single-category single-run; it is the 15-category paired MDD.
7. **P1 inertness self-contradiction** — `major`. "Decoder constructed with the requested one" vs "entry point never passes the argument." Fix: name the stage; align abstract's "never reaches/two-thirds" with the quantisation/duplicate account.
8. **"Confirms protocol stability"** — `major`. A 0.07-point single-backbone agreement is uninformative by the paper's own 2.98 MDD. Fix: "consistent with"; label 82.40 as stability check with script identity.
9. **"Worse than uniform random" vs rank-5-of-14** — `major`. Fix: specify metric, baseline population, n; ground the abstract phrasing.
10. **Minor bundle** — `minor`: "accuracy"→AUROC; "order of magnitude"; "most"→"all adjacent" ranks + explicit non-adjacent reading; "0.2-band" wording; "four of which" slip; oracle labels (best-single-policy row, 93.72, 31% definition); dataset counts; field-wide hedge; pooling-attribution sentence; "exceeds its |mean|"; run-name footnote; max-pooling caption pair; k-grid density; Eq. (1) unit; Methods results claim; contrastive-teacher backbone; stable/degenerate-free terminology.

Items 2–10 are genuinely fixable without re-running anything; only item 1 is blocking, and it is mechanical.

## 5. Fix type (blocking + major)

1. text · 2. text · 3. text (verify against existing run manifests; no retraining) · 4. text · 5. text · 6. text · 7. text · 8. text · 9. text. **No new computations required.**
