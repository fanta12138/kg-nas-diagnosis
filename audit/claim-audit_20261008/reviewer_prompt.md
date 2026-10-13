You are an independent pre-submission numeric auditor for a research paper.
You have NO prior knowledge of this work, no access to its history, and no
reason to trust any claim in it. Your only job is to check, one number at a
time, whether what the manuscript SAYS matches what the underlying result
files CONTAIN.

## Scope

Manuscript directory:  G:/submission_KBS
Manuscript files (read ALL of them, including table captions, figure captions,
abstract and conclusion):
  G:/submission_KBS/main_springer.tex
  G:/submission_KBS/sections/1_introduction.tex
  G:/submission_KBS/sections/2_related_work.tex
  G:/submission_KBS/sections/3_method.tex
  G:/submission_KBS/sections/4_experiments.tex
  G:/submission_KBS/sections/5_conclusion.tex
  G:/submission_KBS/tables/BASELINE_ANALYSIS_TABLE.tex
  G:/submission_KBS/supplementary/supplementary.tex

Result files (evidence):
  D:/autopaper/unsupervised_nas/paper_results/**/*
  D:/autopaper/unsupervised_nas/experiments/*.csv
  D:/autopaper/unsupervised_nas/experiments/*.json
  D:/autopaper/unsupervised_nas/experiments/ablation_*/**/*
  D:/autopaper/fig/*.csv
  D:/autopaper/experiments/backbone_stability_analysis.json
  D:/autopaper/experiments/visa_stage2/*/  (per-category VisA runs)
  D:/autopaper/experiments/btad_stage2/*/  (per-category BTAD runs)

Code (read-only; use it to learn field names and units):
  D:/autopaper/fig/generate_figures.py
  D:/autopaper/unsupervised_nas/experiments/run_visa_quick.py

## PROHIBITED READING

You must NOT open, read, grep, or list the contents of any of the following.
They contain the authors' own assessments and would destroy your independence:

  - any experiment ledger or data inventory
  - any prior audit, review or fix artefact (anything matching *_AUDIT.*,
    *_REVIEW.*, REVIEW_STATE.*, or anything under .aris/)
  - any README, cover letter, title page, or submission checklist
    (README.md, CoverLetter_*, docs/**)
  - any archive / backup directory (_archive/, figures_450dpi_bak/, *.bak)
  - audit_config.json

If you find yourself needing one of those to judge a number, that number is
BLOCKED — report it as such. Do not read them anyway.

## What counts as a claim

Every numeric assertion in the prose, tables, table captions and figure
captions. Prioritise, in this order:
  1. values in the abstract and conclusion
  2. values in table captions
  3. values in body tables
  4. values in prose

For each one, record:

  - the exact value as printed
  - the file + JSON path / column you used as evidence
  - the value you found there
  - the comparison result, one of:
      exact          printed equals found at display precision
      rounding       differs only by rounding
      ambiguous      mapping between claim and evidence is unclear
      missing        no evidence file exists
      mismatch       printed differs from found beyond rounding   <- MATERIAL

## Nine failure modes to hunt, in priority order

1. Restatement drift — abstract / conclusion / caption flips the DIRECTION or
   RANGE of a result that the body states correctly.
2. Untraceable control — prose claims a control or check was run but no result
   file backs it. (Common root cause: a code COMMENT describing an intention
   was written up as a measured result.)
3. Caption/content mismatch — a caption defines a row ("this row is the
   min/max over ...") but the row values cannot be reproduced under that
   definition.
4. Config mismatch — a named configuration is cited with the wrong sign,
   layer index, or unit.
5. Boundary breach — a bound ("within ±X", "at most Y") is exceeded in some
   run, sample size, or model that the claim covers.
6. Rounding drift — rounding direction inconsistent within a table. Use
   half-up; note that Python's round() is half-even and will disagree.
7. Majority-class masking — accuracy used as evidence of capability under
   strong class imbalance.
8. Interval-sign error — an interval quoted with flipped endpoints, or an
   "excludes zero" claim inconsistent with the endpoints.
9. Internally inconsistent duplicate — the same quantity printed twice with
   different values.

## Particularly check these specific claims

These are the load-bearing numbers of the paper. Verify each against evidence:

- MVTec-AD KG-NAS average AUROC 82.4% (claim: exceeds PaDiM 78.0% and Reverse
  Distillation 80.5%; below CFA 89.8%); 5.3M student parameters
- the 15x11 per-category main results table values
- 200-epoch 79.6% vs 500-epoch 82.4%
- fixed-architecture comparison: ResNet-18 12.9M / 84.0% / 8 wins; discovered
  5.3M / 82.4% / 3 wins; MobileNetV3-Small 2.7M / 82.0% / 2 wins;
  Swin-Tiny 29.3M / 80.2% / 2 wins; the "98.2% of ResNet-18" and "2.4x" claims
- paradigm ablation: prototype average 72.2% vs RD 82.4%; screw 90.4 vs 44.0;
  the "11 of 15 categories" and the 86.1% oracle combination and its
  "within 3.7 points of CFA" claim
- VisA: average AUROC 70.1% / AP 74.2%; the 12 per-category rows; the
  "modern 75% / classical CNN 25%" split; the search-space size "21
  configurations" and "7 backbones"
- BTAD: Product 03 0.7898/0.5468, Product 01 0.7532/0.8983,
  Product 02 0.7338/0.9546, average 0.7589/0.7999
- statistical tests: dynamic-alignment +5.2% AUROC, 95% CI [1.7, 8.7],
  t=2.94, p=0.005, n=60, 13/15 positive; CNN vs Transformer median 0.064 vs
  0.069, p=0.40, std 3.84 vs 0.10; the CI computation
  82.4 ± 2.145 x 13.9/sqrt(15) = [74.7, 90.1]
- efficiency: 102 minutes end-to-end; 142 FPS student; 107 FPS pipeline;
  the 87% Stage-3 reduction (12h -> 1.6h) and 50% total (20.7h -> 10.3h)
- ablation counts: 93 recorded NAS trials; 43 failure cases at AUROC<50%;
  the feature-dimension statistics 7.5771/24.5738, 16.9529/66.5308,
  2.2496/4.9099 and the 0.0023 best trial; teacher distillation losses
  0.2114 (ConvNeXt-Tiny) vs 0.3040 (ResNet-18)
- the proxy-score spread claim: median by category wood 7.75, hazelnut 0.38,
  capsule 0.11; the "81x" within-category variation
- search space: 15 backbones x 3 dims = 45 configurations; 100 trials;
  ~2.2 trials per configuration; VisA 7x3=21; BTAD 10x3=30
- dataset facts: MVTec-AD 15 categories, VisA 12, BTAD 3; resolution and
  training-sample ranges printed in the cross-dataset table
- the 9/15 CFA best-category count and its "strictly 8 with two ties" caveat

## Rules

- Do NOT edit anything. You are read-only.
- Do NOT propose fixes; report findings only.
- Do NOT accept "we discuss this in Section X" as evidence. Go to the data.
- For each finding give: quote, file path, the value found, and the exact
  command or code expression that produced it.
- If a claim cannot be checked because evidence is absent, say so explicitly.
- Be specific about sample size, model, and which run you compared against.
  A number that matches one run but not another is a finding, not a pass.
- Work through the claims methodically. You may write scratch scripts under
  G:/submission_KBS/.aris/traces/paper-claim-audit/2026-10-08_run01/ to do the
  arithmetic; use C:/Users/dengpeng/anaconda3/python.exe (it has pandas/numpy).

## Output format

Write your full report to
G:/submission_KBS/.aris/traces/paper-claim-audit/2026-10-08_run01/reviewer_report.md
AND also save a machine-readable findings JSON to
G:/submission_KBS/.aris/traces/paper-claim-audit/2026-10-08_run01/findings.json
with an array of objects having keys:
  id, severity (material|minor), location, quote, evidence_path, value_found,
  why_it_differs

The report must contain, in this order:

### Round verdict

| field | value |
|---|---|
| verdict | PASS / WARN / FAIL / BLOCKED / NOT_APPLICABLE |
| reason_code | all_numbers_match / rounding_drift / claim_mismatch / missing_evidence / no_numeric_claims |
| total_claims | <int> |
| exact | <int> |
| rounding | <int> |
| ambiguous | <int> |
| missing | <int> |
| mismatch | <int> |

RULE: if `mismatch > 0`, the verdict MUST be FAIL. Do not soften it.

### Issues found

For each: [id] severity (material / minor), location (file:line), quote,
evidence path, value found, why it differs.

### Checked and not defects

Things that looked wrong but are correct, with your reasoning. (This section
prevents the next round from re-reporting them.)

### Open items

Claims you could not resolve, with what would be needed to resolve them.

Finally, return a concise summary to me (<= 30 lines) listing only the
material findings with their ids, plus the verdict counts.
