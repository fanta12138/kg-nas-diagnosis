# Triage of the external review against the released evidence

Round 1, 2026-10-16. Reviewer: `glm-5.3-flash`, eight segmented calls plus one
synthesis call. Consolidated verdict: **6/10, `not ready`** (5 blocking, 12 major).

Every finding below was checked against the released code and evidence files
before being accepted or rejected. The prior content audit had a 24 % false
positive rate; this review is comparable, and the synthesis call also inherited
several false positives from the segment reports.

## Rejected as false positive

| ID | Reviewer claim | Why it is wrong |
|---|---|---|
| F1 | "82 policies contradicts its own decomposition (should be 111)" | `fusion_policy.json` has **exactly 82** policy keys = 66 (top-k simplex step 0.1) + 9 (single-scale) + 3 (uniform sum) + 4 (normalised). Verified by enumeration. `111` is the *different* space used in §5.4. **However** the prose that *describes* the 82 space is transcribed from the 111 space and is wrong — see D6. |
| F2 | "F1.2th percentile contradicts the 2.98 MDD" | Two different tests on two different nulls: 2.98 is a paired MDD over 15 categories; 1.2 % is a bootstrap over 200 000 uniform-backbone draws. Both reproduce (`p_random_leq_search = 0.012285`). |
| F3 | "Table 2 mixes twelve and eleven backbones; the 5.7 % rests on slippage" | `make_figures_v2.py` computes the span over the **12** degenerate-free backbones (7.538–7.971 → 5.7 %). The CV figures use mean-of-3-per-dimension CV: range **2.2–12.1 %**, b0 = **74.0 %**, b1 = 169.5 %, mnet = 180.9 %. Every table number reproduces exactly. The set is consistent; only the caption's adjacency of the two ranges is untidy. |
| F4 | "Table 4 band means do not match (vit_tiny, not ResNet-18)" | The table traces to `budget_3M/6M/8M` + **`tier_4M`/`tier_5M`** — 82.01 / 83.14 / 81.55 / 83.25 / 83.08, reproduced exactly. `budget_13M` is an abandoned intermediate run (7 categories only) and is not a source. |
| F5 | "`factor of 175` should be 176.5" | 95.3212 / 0.54455 = 175.05 → "175" is correct truncation of the reported 3-dp shares. |
| F6 | "ρ = −0.12 is pooled and tie-confounded" | `proxy_diag_v2.py:335-353` computes it per backbone (14 points, proxy averaged over the 3 dims), not pooled. Also confirmed: with tie-corrected (average-rank) Spearman, ρ = −0.121 (all 14) and — the reviewer's suggested control — **+0.256 on the 12 well-behaved backbones** (t = 6.0, n = 105 pairs). So the control *strengthens* rather than weakens the claim. |
| F7 | "Table 2's b0 median proxy is 12.1, so the span excludes it" | b0's **pooled median is 7.971**; 12.151 is b0's **maximum** over the 256-dim repeats. The text's phrase "Its median proxy (12.1)" is a genuine typo (see D3) but the span arithmetic is right. |
| F8 | "ResNet-18 = 83.25 but best backbone = 83.34" | Two different tables: `tab:capacity_null` (capacity bands, ResNet-18 at rank 4 of the band study) and `tab:controlled` (14-backbone matrix, ResNet-18 rank 1 at 83.34). Both correct; the paper already cross-references them. |
| F9 | "`+1.06 pp` should be 1.07" | 83.08 − 82.01 = 1.07 on *rounded* band means; using the unrounded values the difference is 1.0620 → 1.06. Display rounding, already disclosed as "computed unrounded". |
| F10 | "√2·σ_res = 4.67 ≠ σ_Δ = 4.12, unexplained" | σ_Δ is measured empirically from the 91 backbone pairs; √2·σ_res is the value it *would* take if the interaction were absent. The gap *is* the interaction term. The paper should say so (D7) but there is no arithmetic error. |
| F11 | "same-seed rerun differs by 0.40 → training nondeterminism" | 82.80 and 82.40 are *different* training runs of the same configuration; the paper documents this ("an independent training run of the same configuration, protocol and seed"). |
| F12 | "82.39 / 93.39 are orphans" | They are the §5.3 normal-data-criterion experiment's own default and oracle on its own held-out split. The paper introduces it as "for the same set of policies on a fixed model". Needs one clarifying clause (D8), not a new measurement. |

## Confirmed defects

| ID | Defect | Severity | Fix type |
|---|---|---|---|
| **D1** | **Abstract is 211 words**, over the MDPI 200-word limit. | blocking | text |
| **D2** | **Label-free claim is framed against the weakest baseline.** On the *same* 70/30 models: default(mean) 81.62, **static top-k (no selection) 88.90**, selection 89.99, oracle 93.19. So **top-k alone recovers 63 %** of the gap and the *criterion* adds only +1.10. Across seeds the criterion adds +1.10 / +0.52 / **−2.82** (mean **−0.40**). The "+7.09, 61 % of the oracle gap" framing attributes top-k's gain to the selection method. | blocking | new computation (done) + rewrite |
| **D3** | Table 2 text says b0's "median proxy (12.1)" — that is its *maximum*; its pooled median is 7.971. | major | text |
| **D4** | "only 2 of the 14 backbones and 6 of the 42 configurations appeared at all" — every run draws **all 14** backbones and **34–42** of the 42 configs. The correct statement is "only 2 of 14 were ever **selected** (returned by argmin)". | major | text |
| **D5** | **The 12-search vs 4-run reconciliation is never stated.** Four `stage2*` runs selected `efficientnet_b0` (rank 5); twelve `search_repeats` runs selected the two collapse-prone backbones (ranks 12 and 7). Both true of different run populations; the text reads as a contradiction. | blocking | text |
| **D6** | The §5.2 prose describing the 82-policy space ("66-point simplex at step 0.1 × top-k, the 21-point grid at step 0.2 × mean and max, plus canonical uniform variants") describes the **111**-policy space of §5.4, not the 82-policy space. 82 = 66 + 9 + 3 + 4. | major | text |
| **D7** | σ_Δ = 4.12 vs √2·σ_res = 4.67 is unexplained; the difference is the interaction term. | minor | text |
| **D8** | §5.3's 82.39/93.39 need one clause saying they belong to that experiment's own holdout split and fixed model. | minor | text |
| **D9** | **`3_method.tex` misdescribes the deployed teacher.** The method section says the teacher is "trained for 100 epochs on the normal training images of the search category, and frozen thereafter". That is Stage 1 (the contrastive teacher behind the *proxy*). The **deployed** RD detector in `rd_detector.py:26-45` builds `TeacherFeatureExtractor` from **ImageNet-1k weights**, frozen, never adapted to the category. This is a real deviation from reverse distillation and is the most likely cause of the 13–17-point deficit. | blocking | text |
| **D10** | **Table 3's caption understates the cross-script disagreement.** Aggregate agreement is 0.13 pp, but per-category the two scripts differ by up to **4.01 pp** (tile: 93.47 vs 90.40), mean |Δ| 0.98 pp, and the worst-category value differs by 1.91 pp (pill: 76.46 vs 78.37) — that column is the stated deployment rationale. | major | text |
| **D11** | SimpleNet and STPM are cited as applying NAS to anomaly detection; neither is a NAS paper. | major | text |
| **D12** | GenAI disclosure self-contradiction: "produced drafts of the manuscript text, **including the figures**" vs "No image or table … was generated by AI". | major | text |
| **D13** | "on the order of 15 points" → 99.6 − 82.47 = **17.1**. | minor | text |
| **D14** | Eq. (1) is not the implemented statistic (batch skipping above a threshold, `total/max(count,1)`, 0.0 as the argmin optimum). | major | text |
| **D15** | "the space of policies … exhaustively" — the simplex is continuous; it is a fixed 82-point grid. | minor | text |
| **D16** | No multiple-comparison control on the 13 backbone contrasts (p = 0.013) or the 10 band tests (p = 0.058). | minor | text (caveat) |
| **D17** | Typo "dominated something else entirely" → "dominated **by**". | minor | text |
| **D18** | MVTec-AD and reverse-distillation citations are introduced late; pooling precedent (SPADE/PatchCore/CFA) and synthetic-anomaly self-supervision (CutPaste/NSA) uncited in §5.2. | minor | text |
| **D19** | "2026 state of the art" while the cited SOTA is 2025. | minor | text |
| **D20** | "roughly a hundred images per category" → actual counts 42–167, mean 115. | minor | text |

## Positively verified (no action)

- All of §5.1 reproduces exactly from `fusion_policy.json` (82.80 / 42.22 / 90.21 /
  78.37 / 86.84 / 93.72 / 90.60); per-category deltas screw +41.8/+45.5,
  metal_nut +19.65, grid +18.71, toothbrush +13.61, carpet +13.44, capsule +11.61,
  transistor −2.00, zipper −3.02, tile −4.62, pill −4.80, cable −5.06.
- MDD arithmetic: z = 1.95996 + 0.84162 = 2.8016; 2.8016 × 4.1173/√15 = **2.978**.
- 240 runs/arm and 532 categories: 240.21 and 532.21.
- Truth matrix: 210 cells at seed 42, fully crossed, 390 entries total, 90
  replicated cells.
- Search-vs-random: 80.91 % vs 81.65 %, best 83.34 %, mean rank 10.33/14,
  percentile 1.2285, P(all 12 in the 2-element subset) = 7.22e−11, 0 of 12 runs
  selected a top-4 backbone. Search mean beats 64 % of individual draws and sits
  at P(mean) = 0.021 under a 12-draw null.
- **New control (answers M4):** if trials scoring exactly 0.0 are filtered, the
  twelve runs select `deit_tiny` (9), `resnet18` (2), `efficientnet_b1` (1) —
  all reasonable backbones. This *confirms* the mechanistic thesis: the failure
  is the degenerate-value × argmin interaction, not an intrinsically bad space.
- **New control (answers F6/M9):** ρ = +0.256 (t = 6.0) on the 12 well-behaved
  backbones vs −0.121 on all 14.
