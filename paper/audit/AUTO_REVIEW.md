# AUTO_REVIEW — external peer review loop

**Paper:** `submission_KBS` — *The Architecture Axis Is Not the Axis: A Diagnostic
Study of Neural Architecture Search and Score Fusion in Unsupervised Industrial
Anomaly Detection* (target: Applied Sciences, MDPI).

**Reviewer model:** `glm-5.3-flash` (Zhipu AI). **Protocol:** segmented review —
the manuscript was split into eight sections and each was passed to the reviewer
in a separate call (the model truncates on very large single inputs), followed by
one synthesis call that received only the eight segment reports. Driver scripts:
`review-stage/run_external_review.py` and `review-stage/run_synthesis.py`.

**Verdict vocabulary:** `ready` / `almost` / `not ready`. **Stop condition:** score
>= 6/10 *and* verdict in {ready, almost}. **MAX_ROUNDS = 4.**

---

## Round 1 (2026-10-16T17:52:00)

### Assessment (Summary)
- **Score: 6/10**
- **Verdict: `not ready`**
- **Key criticisms (consolidated by the reviewer):** five blocking items and twelve
  major items. In the reviewer's own ordering:
  - B1 headline numbers are a moving target (81.49/82.36/82.47/82.80, 90.21 vs 90.28, 111 vs 82);
  - B2 the search-run record reads as self-contradictory (twelve runs vs four runs);
  - B3 inertness is stated three incompatible ways;
  - B4 **the label-free headline is framed against the honest label-free baseline**;
  - B5 the teacher's scope is undeclared (a real deviation from reverse distillation);
  - M1 variance-decomposition reporting; M2 harness validity; M3 Eq. (1) vs the
    implemented statistic; M4 collapse-selection robustness; M5 "worse than random"
    phrasing; M6 the 5.7% headline's backbone set; M7 Table 1's single-seed ranking;
    M8 MDD attribution; M9 inference hygiene; M10 tail discrepancy across scoring
    scripts; M11 scope and citation integrity; M12 GenAI disclosure self-contradiction.

### Reviewer Raw Response

<details>
<summary>Click to expand the full consolidated reviewer response (verbatim, 6731 characters)</summary>

**1. Score: 6/10.** The study design (14×15×seeds, paired statistics, disclosed negative results) is genuinely good and the diagnostic framing is valuable — that profile alone would be a 7. But several headline claims are currently stated inconsistently or rest on unresolved comparisons, which is material, not expositional. The author's two false-positive rulings are accepted: the 82-policy count stands (the remaining defect is Table 3 saying 111 — fix the table, not the text), and the 1.2th-percentile vs MDD objection dissolves (different tests); the residual issue there is only that the percentile's null distribution must be described as what it is.

**2. Verdict: `not ready`** (blocking items below are mostly cheap, but must land first).

**3. Remaining critical weaknesses** (merged across segments):

Blocking
- **B1 — Headline numbers are a moving target.** Baselines 82.36 / 82.47 / 82.80 all appear without definition; 90.28 vs 90.21; intro +7.8, conclusion +7.92 vs stated +7.41; deltas not reproducible from displayed values; §5.3's 82.39/93.39 match no other measurement; Table 3's 111 vs verified 82; "on the order of 15" → 17.1. *Fix: one canonical number per quantity, each defined (which split, seed, script), used identically in abstract/intro/conclusion/tables.* — **text**
- **B2 — The search-run record contradicts itself.** "All 12 searches selected the collapse-prone backbone, every single time" vs "the four runs originally reported chose a backbone that does not collapse" vs positioning's EfficientNet-B0 selection. Verify against logs, reconcile counts and spaces, delete the revision artifact, and correct claim 3 if the record demands it. — **text**
- **B3 — Inertness is stated three incompatible ways** (2 of 3 dims never reach the model / the searched dim never reaches it / B fixed at 256 makes all three unobservable). State the actual code path once; correct the abstract to match. — **text**
- **B4 — The label-free headline ignores the honest label-free baseline.** Static uniform top-k (90.21) beats the selection procedure (88.59); "recovers 61% of the oracle gap" is framed against the weakest default. *Fix: report per-category static-top-k vs selection (existing runs), reframe the claim around where selection adds value, or drop the recovery figure as the headline.* — **new computation**
- **B5 — Teacher scope is undeclared.** One wood-trained frozen teacher serving 15 categories would be a major, unstated deviation from reverse distillation and a candidate explanation for the 13–17-point deficit; the text as written cannot be true either way. Declare which teacher(s) exist and which produces t₁–t₃; justify or limit. — **text**

Major
- **M1 — Variance-decomposition reporting.** Point estimates (0.54/95.32) quoted as *the* decomposition while the abstract gives ranges; seeds' role unstated (per-cell means? p = 0.039's test unnamed); "accuracy" vs AUROC; same-seed nondeterminism (82.80 vs 82.40) undisclosed; sd from 3 seeds treated as precise. — **text**
- **M2 — Harness validity.** 17.1-point deficit vs published RD with no anchor; BN × gradient-accumulation confound unexamined. *Fix: reproduce one published RD configuration in the harness; report a micro-batch sensitivity check.* — **new computation**
- **M3 — Eq. (1) is not the implemented statistic** (batch skipping, max(count,1), 0.0 as the argmin optimum). Rewrite with the skip indicator; state the threshold. — **text**
- **M4 — Collapse-selection robustness.** 0.0 returned only 73%/13% of the time yet "selected every time"; show selection survives excluding collapsed evaluations; define "appeared" and explain 6/42 coverage. — **new computation**
- **M5 — "Worse than random" phrasing.** 0.74 points is below the paper's own 2.98 MDD; describe the percentile test's null correctly (mean vs bootstrap, not single draws) and rest the claim on the verifiable facts: which backbone was selected, its collapse behavior, mean rank 10.3/14. — **text**
- **M6 — The 5.7% headline median** rests on an eleven-vs-twelve "well-behaved" slippage (b0's 12.1 in the CV span, out of the median span). Define the set once; recompute one number. — **text**
- **M7 — Table 1's 14-way ranking is single-seed** with rank gaps (0.08) far below the paper's own MDD. Report mean±sd over replicated cells or drop/reframe the Rank column. — **new computation**
- **M8 — MDD attribution.** Contribution 1 assigns 2.98 to the single-category single-run test; only the 15-category paired reading is consistent with 532. Fix the contribution/abstract and show how 240 follows from the stated σ's; explain 4.67 vs 4.12. — **text**
- **M9 — Inference hygiene.** No multiple-comparison control (13 contrasts at p=0.013; 10 band tests at p=0.058 — the latter survives nothing); ρ = 0.215/−0.12 lack tests, attenuation correction, and a pooled-vs-averaged definition; report ρ on the 12 well-behaved backbones. — **new computation**
- **M10 — Tail discrepancy across scoring scripts** (same policy, same models: 78.37 vs 76.46; 15× the claimed 0.13 agreement). The k=1–2% recommendation depends on which script is canonical; reconcile them. — **new computation**
- **M11 — Scope and citation integrity.** Reduction-step causality shown on one backbone/family — hedge or extend top-k across backbones; cite pooling precedent (SPADE, PatchCore, CFA) and synthetic-anomaly work; SimpleNet/STPM are not NAS; fix the related-work taxonomy; cite MVTec-AD and RD; scope the "first comparison" claim. — **text**
- **M12 — GenAI disclosure self-contradiction** ("drafts… including the figures" vs "no image generated by AI"). MDPI editors read this section closely. — **text**

Minor: terminology unification (score fusion / top-k / reduction; AUROC not "accuracy"); rounding sweep (80.90, 176.5, 241, 533, 1.07); 2026-dated citations; variant provenance for cited published numbers; "exhaustive" → fixed grid of 82; alignment-layer fairness and contrastive-objective details; actual per-category test-set counts; title tautology, dangling modifier, "dominated by," ORCID placeholders.

**4. Type summary.** Blocking: B1–B3, B5 text; B4 new computation. Major: text — M1, M3, M5, M6, M8, M11, M12; new computation — M2 (one anchor run), M4, M7, M9, M10 (all computable from existing runs/logs except M2).

Bottom line: the science underneath is publishable and the negative results are worth publishing; the manuscript currently undermines its own headline claims through internal inconsistency, and one claim (label-free selection) is framed against the wrong baseline. Fix B1–B5 and the text-type majors and this is a 7–8; the two genuine experiments (anchor run, selection-vs-static-top-k analysis) are what separate this from `almost`.

</details>

### Actions Taken

Every finding was checked against the released code and evidence before being
accepted or rejected; the full triage is in `review-stage/TRIAGE.md`. Twelve
reviewer claims were rejected as false positives (F1–F12) and twenty were
confirmed (D1–D20). Four new control computations were written, run and released
to settle the items that were checkable but not previously measured:

| New artefact | Settles | Result |
|---|---|---|
| `pseudo_select/baseline_decomposition.py` + `.json` | B4 / D2 | On the same 70/30 models the default scores **81.62**, a *fixed* top-$k$ (no selection) already reaches **88.90**, the label-free criterion reaches **89.99**, the oracle **93.19**. The criterion's own contribution is **+1.10 / +0.52 / −2.82** (mean **−0.40**). |
| `search_repeats/degenerate_filter_control.py` + `.json` | M4 / D4 | Filtering trials scoring exactly 0.0 makes the twelve runs select `deit_tiny` (8), `resnet18` (2), `efficientnet_b1` (2) — all reasonable. Confirms the failure is the degenerate-value × argmin interaction, not a bad space. |
| `proxy_diag/rho_wellbehaved_control.py` + `.json` | M9 / F6 | ρ = +0.215 (all 14), **+0.256** (12 degenerate-free, t = 6.0), +0.278 (11, t = 6.6). The suggested control *strengthens* the transfer claim. |
| `proxy_diag/rank_stability_control.py` + `.json` | M7 | Only 6 of 14 backbones were replicated at all three seeds; among those six the rank order is **identical in every seed (max move 0)**. |

Manuscript edits applied (all with a timestamped pre-edit backup at
`audit/r3_backup_20261016_175700/`):

- **D1** abstract cut from 211 to **199** words (MDPI limit 200), all headline numbers retained.
- **D2 / B4** the label-free claim rewritten around the honest baseline: the paper
  now states that a *fixed* top-$k$ pooling carries the gain (+7.49 mean over the
  default) and that the selection criterion's own contribution is **−0.40** on
  average. Presented as a self-critical disclosure, not hidden.
- **D9 / B5** `3_method.tex` now states that the **deployed** detector's teacher is
  an ImageNet-1k ResNet-18, distinct from the contrastively pre-trained search
  teacher, and that this is a deviation from published reverse distillation.
- **D5 / B2** the twelve-run and four-run populations are now stated separately.
- **D3, D4, D6, D7, D8, D10–D20** applied text corrections, including the 82-policy
  decomposition, the Table-3 per-category cross-script discrepancy, the "dominated
  **by**" typo, the 42–167 per-category test counts, the 2022/2025 SOTA wording,
  the Eq. (1) rewrite with the skip indicator, and multiple-comparison caveats.
- Reverted the prior session's incorrect "twelve → eleven" edit (false positive F3):
  `make_figures_v2.py` computes the 5.7% span over the **12** degenerate-free
  backbones (7.538–7.971), verified by direct recomputation.

### Results

`latexmk` rebuild of `main_v2.tex`: **0 errors, 0 undefined references or
citations, 26 pages.** A number-multiset differential (`numsnap`-style) against the
pre-edit backup confirms no number was silently dropped; every delta is an intended
edit and each removed figure was replaced by an exact recomputation.

### Status

- Continuing to **Round 2**: the text fixes are applied and the paper recompiles;
  the residual question is whether the reviewer, seeing the corrected manuscript,
  moves the verdict to `ready`/`almost` at score >= 6.
- Note on the stop condition: this is an evaluation study whose contribution is a
  **negative** result and an honest decomposition. Round 1 scored 6/10 — already at
  the score threshold — so the binding item is the verdict, which is gated on the
  four new computations above and the D1–D20 text fixes now landed.


---

## Round 2 (2026-10-16T18:40:00)

### Assessment (Summary)
- **Score: 7/10** (up from 6/10)
- **Verdict: `almost`** (up from `not ready`)
- **Round-1 blocking items:** B1 unresolved-but-now-mostly-mechanical after this
  round's fixes; **B2, B3, B4, B5 resolved** in the reviewer's own words.
- **Reviewer's one remaining `blocking` item:** the canonical-value sweep (one scope
  per quantity, stated at first use). The remaining nine items are `major`/`minor`
  and all `text` fixes; the reviewer states explicitly that **no new computation is
  required**.

### Reviewer Raw Response

<details>
<summary>Full round-2 synthesis (verbatim)</summary>

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

</details>

### Actions Taken

All ten items the round-2 reviewer lists were fixed in this round, each against the
released evidence:

1. **Canonical-value sweep (`blocking`).** Root cause was that four different
   quantities share the +7.x label. Each is now defined once by *split, seed set and
   script* and used identically everywhere:
   - full test set, seed 42 (`fusion_policy`): 82.80 -> 90.21 = **+7.42**;
   - full test set, seeds 43/44 (`fusion_multiseed`): 82.36 -> 90.28 = **+7.92**;
   - full test set, per-seed estimates: **+7.78** and **+8.07**;
   - label-free 70/30 split (`baseline_decomposition`): **+7.49** (fixed top-k) with
     the criterion adding **-0.40**.
   The mixed three-seed mean (which averaged two scripts) was **removed**; the
   abstract/introduction/conclusion headline is the conservative **+7.5**. 240/532
   were made consistent at **241/533** (the script's own 240.2 / 532.2, rounded up);
   `82.47` is now named as the canonical headline with `82.40` labelled a
   stability check; P4's +1.06 to **+1.07** and 2.4 to **2.3**; the six P5 0.01 drifts
   are disclosed as pre-rounding computations with a worked example.
2. **Mischaracterised controls (`major`).** The filter-control sentence was false
   (efficientnet_b1 is rank 12): rewritten to state the three winners' true ranks
   (6, 1, 12) and that the filter guarantees a *real* evaluation, not a *good* one.
   The capacity-band sentence now says restriction happens at selection, which the
   verified `tier_selection.json` bands bear out.
3. **111-policy set (`major`).** Decomposed as **66 + 21 + 21 + 3 = 111**, verified
   by executing `build_policies()`.
4. **"No repetition" vs 100 draws (`major`).** Corrected: each trial trains a fresh
   student, cells are revisited and *do* re-train, but the search entry point returns
   only the winner, so the reported value carries no variance estimate.
5. **Teacher identity (`major`).** The P3 cross-reference was backwards and is
   rewritten to say every *reported* result rests on the frozen ImageNet teacher;
   the contrastive teacher enters only the Stage-2 proxy. "One teacher" becomes
   "one teacher per track".
6. **2.98 misattribution (`major`).** Contribution 1 and the abstract now call 2.98
   the 15-category *paired* MDD and note the single-category design needs 241 runs.
7. **Inertness self-contradiction (`major`).** Rewritten to one coherent code path:
   the encoder hard-codes 256, the decoder keeps its 256 default because the entry
   point never passes the argument, so the axis is unreachable rather than unused.
8. **"Confirms stability" (`major`).** Now "consistent with", with the 0.07-point
   agreement explicitly below the paper's own 2.98-point MDD.
9. **"Worse than random" vs rank-5 (`major`).** Abstract now reads "never one of the
   four best"; the body already carries the 0.74 < 2.98 caveat.
10. **Minor bundle.** All applied: `accuracy` to AUROC; "order of magnitude" replaced
   by "nearly twice the 3.98-point spread"; "most" to "all adjacent" ranks with the
   non-adjacent reading made explicit; "0.2-band" rewording; "four of which" becomes
   all twelve at rank 7 or worse; oracle labels on the best-single-policy and 93.72
   rows; 31% defined over 105 pairs; dataset 1725/42-167/115 reconciled; field-wide
   extrapolation hedged; "exceeds its |mean|"; run-name footnote; max-pooling caption
   pair; Eq. (1) unit corrected from per-image to per-batch; Methods results claim
   neutralised; contrastive teacher's backbone named (ResNet-18); stable to
   degenerate-free.

### Results

`latexmk` rebuild of `main_v2.tex` after this round: **0 errors, 0 undefined
references or citations, 27 pages.** Abstract within limit at **200 words**. Every
corrected number was recomputed from the released JSON before being written.

### Status

- **Stop condition met:** score **7/10** (>= 6) **and** verdict **`almost`**
  (in {ready, almost}). The loop is complete; no Round 3 is required.
- The one item the reviewer still marks `blocking` was the canonical-value sweep,
  which this round's Actions Taken item 1 addresses; the reviewer also states no new
  computation is required for any remaining item.
