# S6 discussion + conclusions

## PAPER TEXT (sent)

% ============================================================================
% DRAFT v2 -- Section 6: Honest positioning
% Published values are as reported by the original authors (not re-run); the
% controlled comparison is our own (proxy_diag/phaseB_truth.json, 210 detectors).
% ============================================================================

\section{Discussion: Where This Pipeline Stands}
\label{sec:positioning}

This paper reports a diagnostic study, not a new state-of-the-art detector, and it
is therefore necessary to state plainly where the pipeline of
Section~\ref{sec:method} stands. Two comparisons are relevant. As \emph{reported by
the original authors}, image-level AUROC on MVTec-AD runs from $78.0\%$
(AutoEncoder) and $85.0\%$ (VAE)~\cite{bergmann2018improving} through $95.6\%$
(STPM)~\cite{wang2021stpm}, $96.0\%$ (SPADE)~\cite{cohen2020spade}, $97.9\%$
(PaDiM)~\cite{defard2021padim}, $98.5\%$
(reverse distillation)~\cite{deng2022reverse}, $99.1\%$
(PatchCore~\cite{roth2022patchcore}, EfficientAD~\cite{batzner2024efficientad}) and
$99.3$--$99.6\%$
(CFA~\cite{lee2022cfa}, FastFlow~\cite{yu2021fastflow},
SimpleNet~\cite{liu2023simplenet}, Dinomaly~\cite{guo2025dinomaly}). FastFlow
reports $99.4\%$. These numbers were not re-run here and their protocols are not
identical to ours, so they are for orientation only. Two of them, SPADE and
FastFlow, are preprints that were never published at a venue; they are included
because they are standard baselines in this literature, and their preprint status
is declared here rather than left implicit. The second comparison,
Table~\ref{tab:controlled}, is our own, in which all 14 backbones of the search
space were trained under one protocol on all 15 categories.

\begin{table}[t]
\centering
\caption{Controlled comparison: every backbone of the search space trained on
every MVTec-AD category under an identical protocol (500 epochs, batch size 64,
seed 42, ResNet-18 teacher, reverse distillation; 210 detectors). Deployed size is
the student encoder plus decoder. The manuscript's selected architecture is marked
with~$\dagger$.}
\label{tab:controlled}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
Backbone & Student (M) & Mean AUROC (\%) & Rank \\
\midrule
ResNet-18        & 13.04 & \textbf{83.34} & 1 \\
ResNet-34        & 23.15 & 82.99 & 2 \\
ResNet-50        & 25.77 & 82.75 & 3 \\
MobileOne-S0     &  6.27 & 82.58 & 4 \\
EfficientNet-B0$^\dagger$ & 5.41 & 82.47 & 5 \\
DeiT-Tiny        &  7.31 & 82.03 & 6 \\
MobileNetV3-Small & 2.81 & 81.85 & 7 \\
ViT-Tiny         &  7.31 & 81.51 & 8 \\
ConvNeXt-Small   & 51.39 & 81.43 & 9 \\
ConvNeXtV2-Tiny  & 29.80 & 81.35 & 10 \\
ConvNeXt-Tiny    & 29.75 & 80.91 & 11 \\
EfficientNet-B1  &  7.92 & 80.43 & 12 \\
Swin-Tiny        & 29.26 & 80.11 & 13 \\
ConvNeXtV2-Base  & 89.69 & 79.36 & 14 \\
\bottomrule
\end{tabular}}

\end{table}

The pipeline described in this paper reaches $82.47\%$ mean AUROC on MVTec-AD with
the architecture the search selected, \texttt{efficientnet\_b0} (5.41M parameters),
against reported values of $95.6\%$--$99.6\%$ for the established detectors. We
therefore do not claim a competitive operating point: the gap to the strongest
published single-class detectors is on the order of $15$ points, and we report it
rather than obscure it. Nor is the gap explained by the parameter budget:
Section~\ref{subsec:capacity_null} varies the deployed size over a $9.2\times$
range across five capacity bands and finds no statistically resolvable effect, and
across the full space of Table~\ref{tab:controlled} the deployed size spans
$32\times$ while the backbone accounts for $0.54\%$ of the variance. One caveat
applies to the reported values: they come from implementations, augmentation
policies and training schedules that differ from ours, so the comparison is
indicative rather than exact. Table~\ref{tab:controlled} is the within-study
comparison on which the analysis in Section~\ref{sec:diagnosis} rests; it was
produced by the same code path, on the same data split, with one variable, the
backbone, changed at a time.

Two things are worth taking from Table~\ref{tab:controlled} in their own right.
It is, to our knowledge, the first comparison of a full anomaly-detection search
space under a single protocol, and as such it is a reusable resource: the ordering
it establishes is what makes the negative result in Section~\ref{sec:diagnosis}
measurable rather than anecdotal. And it shows that the search's output, at rank 5
of 14, is a reasonable but unremarkable choice from a space whose top eleven
candidates span $2.43$ points. The truth matrix also reproduces the manuscript's
headline number: \texttt{efficientnet\_b0} attains $82.47\%$ here against the
$82.40\%$ reported previously, a difference of $0.07$ points. Since the earlier
figure came from an independent training run, this confirms that the protocol is
stable and that the differences discussed here are not artefacts of a drifting
pipeline. The search itself does \emph{not} reproduce as closely;
Section~\ref{subsec:proxy_stability} documents why.


% ============================================================================
% DRAFT v2 -- Section 7: Conclusion
% Genre: benchmark / evaluation study.
% ============================================================================

\section{Conclusions}
\label{sec:conclusion}

This paper belongs to the class of studies that measure how a field evaluates
itself. Such studies do not add a detector; they establish what the field's
existing measurements are worth, and they are useful to the extent that the
answer changes how the next comparison is read. Our answer is that the standard
evaluation design for unsupervised anomaly detection cannot resolve the
differences the field reports, and that in the pipeline we instrumented the
design axis those comparisons are usually about contributes almost nothing.

The binding constraint is resolution. A two-way decomposition of the 210-detector
matrix we trained leaves a residual standard deviation of $3.30$ points, and the
run-to-run standard deviation of a single detector at a fixed (backbone,
category) is $1.96$ points over three seeds. Comparing two backbones across the
fifteen categories the benchmark provides is therefore a paired test whose minimum
detectable difference is $2.98$ points --- larger than the entire $3.98$-point
spread of the fourteen backbones, and six times the $0.5$-point margin that
separates PatchCore from the strongest results reported since. Resolving that
margin on a single category would take $240$ independent training runs per
backbone, or $532$ categories. This is a statement about the benchmark rather than
about our pipeline, and it bounds every architecture comparison made on it.

Within that constraint, the diagnosis of the architecture axis is unambiguous.
Two thirds of the search space is spent on a feature dimension that cannot reach
the deployed model, because the encoder that consumes it hard-codes a width while
the decoder honours the requested one. The search signal does not distinguish
architectures: it separates the twelve well-behaved backbones by $5.7\%$ of its
median, which is the same order as its own run-to-run noise, and it returns
exactly $0.0$ for two of them whenever their student diverges and every validation
batch is rejected. And the quantity it is used to predict is dominated by
something else entirely: architecture accounts for $0.54\%$ of the variance in
detection accuracy and the category for $95.32\%$
(Section~\ref{subsec:capacity_null}).

The interaction between the first two facts produces the result we did not
anticipate. Because the objective can silently return the optimum of its own
range, and the search minimises a single sample per candidate, the rule does not
select the best architecture but the one with the heaviest left tail --- which is
the one that failed. Twelve independent runs selected a configuration prone to
this collapse every single time, and the architectures they returned perform
\emph{worse} than uniform random selection, at the $1.2$th percentile of the
random distribution (Section~\ref{subsec:search_vs_random}). The four runs whose
results the manuscript originally reported chose a backbone that does not
collapse, which on this account was luck.

The constructive finding is where the accuracy actually is. The reduction that
turns a multi-scale discrepancy map into an image-level score is worth $+7.92$
points of mean AUROC across three training seeds --- an order of magnitude more
than the architecture --- and it is worth those points without retraining
anything. The default choice, an unweighted mean over all pixels, is not merely
suboptimal: it fails below chance on \texttt{screw}, and top-$k$ pooling repairs
that failure to $84\%$. The gain is robust across two orders of magnitude in $k$
(Section~\ref{sec:leverage}). And the reduction can be chosen without any anomaly
label, by perturbing held-out normal images with synthetic defects and selecting
the policy that best separates them from the clean ones: averaged over three
training seeds this recovers $61\%$ of the oracle gap, though the seeds range from
$41\%$ to $72\%$, so it is a direction rather than a settled number.

Three limitations bound these claims. The diagnosis is established on MVTec-AD,
with one teacher, one detector family and one training protocol; whether the same
collapse appears in other pipelines is a question our evidence cannot settle,
though the mechanism --- a thresholded objective that can return its own optimum,
combined with a minimising selection rule --- is not specific to this codebase.
Our recommendation for the architecture axis is deliberately narrow: the backbone
effect is statistically detectable ($p = 0.039$) and merely two orders of
magnitude smaller than the category effect, so we do not conclude that
architecture does not matter, nor that NAS for anomaly detection is worthless in
principle. And the resolvability figure is conditional on this benchmark's fifteen
categories and this task's residual variance; it should be recomputed rather than
assumed for another benchmark or detector family --- which is exactly the
calculation a benchmark paper ought to publish, and which here decides the
question.

Two things follow for practice. A search whose objective can silently return the
optimum of its own range must filter degenerate evaluations before minimising, and
should report the distribution of its per-configuration scores rather than a
single minimum. And any deployed reverse-distillation detector currently scoring
images by the mean over its discrepancy map can be re-scored, on the models it
already has, with a $k$ of two percent, and expect to gain several points of
accuracy and to stop failing below chance on the categories that currently defeat
it. Neither of those requires convincing anyone that a new architecture is better,
which is convenient, because on this benchmark that cannot be shown.


## EVIDENCE FILES SENT

- anomalib_baselines/results.json
- adbench/results/analysis_p1_p3.json

## REVIEWER RAW RESPONSE

Audited S6–S7 claims against the two supplied files. The paper's cited source (proxy_diag/phaseB_truth.json) is absent, so the table and all headline statistics are untraceable; the locality file does not corroborate the reduction-gain claims.

```json
{
  "claims": [
    {"id": 1, "location": "S6 published-values list", "paper_text": "$78.0\\%$ (AutoEncoder) and $85.0\\%$ (VAE)", "paper_value": "AE 78.0%, VAE 85.0%", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Literature values; no file records them. Expected for citations, listed per protocol."},
    {"id": 2, "location": "S6 published-values list", "paper_text": "$95.6\\%$ (STPM), $96.0\\%$ (SPADE)", "paper_value": "STPM 95.6%, SPADE 96.0%", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Literature values, unverifiable from provided files."},
    {"id": 3, "location": "S6 published-values list", "paper_text": "$97.9\\%$ (PaDiM), $98.5\\%$ (reverse distillation)", "paper_value": "PaDiM 97.9%, RD 98.5%", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Literature values, unverifiable from provided files."},
    {"id": 4, "location": "S6 published-values list", "paper_text": "$99.1\\%$ (PatchCore, EfficientAD)", "paper_value": "99.1% both", "evidence_file": "anomalib_baselines/results.json", "evidence_value": "PatchCore bottle image_auroc = null (run errored)", "status": "missing_evidence", "details": "Published value not in files; failed PatchCore re-run exists but paper claims no re-run, so no contradiction."},
    {"id": 5, "location": "S6 published-values list", "paper_text": "$99.3$--$99.6\\%$ (CFA, FastFlow, SimpleNet, Dinomaly). FastFlow reports $99.4\\%$", "paper_value": "99.3-99.6%; FastFlow 99.4%", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Literature values; internally consistent (99.4 within range) but untraceable."},
    {"id": 6, "location": "S6 preprint note", "paper_text": "SPADE and FastFlow, are preprints that were never published at a venue", "paper_value": "both preprints only", "evidence_file": "none", "evidence_value": "absent", "status": "unsupported_claim", "details": "Publication-venue claim not verifiable from evidence files."},
    {"id": 7, "location": "S6 + Table caption", "paper_text": "all 14 backbones ... trained under one protocol on all 15 categories ... 210 detectors (500 epochs, batch size 64, seed 42, ResNet-18 teacher, reverse distillation)", "paper_value": "14 backbones, 15 categories, 210 detectors, protocol constants", "evidence_file": "none (cites proxy_diag/phaseB_truth.json)", "evidence_value": "file not provided", "status": "missing_evidence", "details": "Cited source file absent; scope and protocol unverifiable."},
    {"id": 8, "location": "Table tab:controlled", "paper_text": "ResNet-18 13.04 83.34 1 ... ConvNeXtV2-Base 89.69 79.36 14 (14 rows)", "paper_value": "14 param counts, mean AUROCs, ranks", "evidence_file": "none (cites proxy_diag/phaseB_truth.json)", "evidence_value": "absent", "status": "missing_evidence", "details": "All 42 table cells untraceable; source file not among evidence."},
    {"id": 9, "location": "S6", "paper_text": "reaches $82.47\\%$ mean AUROC ... \\texttt{efficientnet\\_b0} (5.41M parameters)", "paper_value": "82.47%, 5.41M", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Headline number and parameter count untraceable."},
    {"id": 10, "location": "S6", "paper_text": "the gap to the strongest published single-class detectors is on the order of $15$ points", "paper_value": "~15 points", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Derived from unsupported 82.47% and literature values; arithmetic gives 13.1-17.1."},
    {"id": 11, "location": "S6 capacity null", "paper_text": "$9.2\\times$ range across five capacity bands and finds no statistically resolvable effect", "paper_value": "9.2x, 5 bands, null effect", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Capacity-band analysis not in provided files."},
    {"id": 12, "location": "S6", "paper_text": "the deployed size spans $32\\times$", "paper_value": "32x", "evidence_file": "none (Table)", "evidence_value": "89.69/2.81 = 31.9 (table unsupported)", "status": "missing_evidence", "details": "Arithmetic consistent with table but table itself untraceable."},
    {"id": 13, "location": "S6", "paper_text": "the backbone accounts for $0.54\\%$ of the variance", "paper_value": "0.54%", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Variance decomposition not in provided files."},
    {"id": 14, "location": "S6 reproduction check", "paper_text": "$82.47\\%$ here against the $82.40\\%$ reported previously, a difference of $0.07$ points", "paper_value": "82.47 vs 82.40, diff 0.07", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Difference arithmetic correct; both inputs untraceable."},
    {"id": 15, "location": "S6", "paper_text": "at rank 5 of 14 ... top eleven candidates span $2.43$ points", "paper_value": "rank 5/14; 2.43-pt span", "evidence_file": "none (Table)", "evidence_value": "83.34-80.91 = 2.43 (table unsupported)", "status": "missing_evidence", "details": "Consistent with table arithmetic; table unverified."},
    {"id": 16, "location": "S6", "paper_text": "the first comparison of a full anomaly-detection search space under a single protocol", "paper_value": "priority claim", "evidence_file": "none", "evidence_value": "absent", "status": "unsupported_claim", "details": "Novelty claim cannot be verified from evidence; hedged with 'to our knowledge'."},
    {"id": 17, "location": "S7", "paper_text": "leaves a residual standard deviation of $3.30$ points", "paper_value": "3.30 pts", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Two-way decomposition not in provided files."},
    {"id": 18, "location": "S7", "paper_text": "run-to-run standard deviation of a single detector ... $1.96$ points over three seeds", "paper_value": "1.96 pts, 3 seeds", "evidence_file": "adbench/results/analysis_p1_p3.json", "evidence_value": "seed_sd I-AUROC = 0.0035 (0.35 pts), different analysis", "status": "missing_evidence", "details": "Only seed SD in files is 0.35 pts, different scope; 1.96 untraceable."},
    {"id": 19, "location": "S7", "paper_text": "minimum detectable difference is $2.98$ points", "paper_value": "2.98 pts", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Derived statistic; inputs untraceable."},
    {"id": 20, "location": "S7", "paper_text": "larger than the entire $3.98$-point spread of the fourteen backbones", "paper_value": "3.98 pts", "evidence_file": "none (Table)", "evidence_value": "83.34-79.36 = 3.98 (table unsupported)", "status": "missing_evidence", "details": "Arithmetic consistent with unverified table."},
    {"id": 21, "location": "S7", "paper_text": "six times the $0.5$-point margin that separates PatchCore from the strongest results reported since", "paper_value": "6x of 0.5 pts", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "2.98/0.5 arithmetic fine; 0.5-pt margin is literature claim, untraceable."},
    {"id": 22, "location": "S7", "paper_text": "would take $240$ independent training runs per backbone, or $532$ categories", "paper_value": "240 runs; 532 categories", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Power calculation not reproducible from provided files."},
    {"id": 23, "location": "S7 mechanism", "paper_text": "the encoder that consumes it hard-codes a width while the decoder honours the requested one", "paper_value": "width-mismatch mechanism", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Code-level mechanism claim; no code/config evidence supplied."},
    {"id": 24, "location": "S7", "paper_text": "separates the twelve well-behaved backbones by $5.7\\%$ of its median", "paper_value": "5.7% of median", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Search-signal spread not in provided files."},
    {"id": 25, "location": "S7", "paper_text": "returns exactly $0.0$ for two of them whenever their student diverges", "paper_value": "0.0 for 2 backbones", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Divergence/zero-score behavior untraceable."},
    {"id": 26, "location": "S7", "paper_text": "architecture accounts for $0.54\\%$ of the variance ... category for $95.32\\%$", "paper_value": "0.54% / 95.32%", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Variance shares untraceable."},
    {"id": 27, "location": "S7", "paper_text": "Twelve independent runs selected a configuration prone to this collapse every single time", "paper_value": "12/12 runs", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Search-run outcomes not in provided files."},
    {"id": 28, "location": "S7", "paper_text": "at the $1.2$th percentile of the random distribution", "paper_value": "1.2th percentile", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Random-distribution comparison untraceable."},
    {"id": 29, "location": "S7", "paper_text": "The four runs whose results the manuscript originally reported chose a backbone that does not collapse", "paper_value": "4 runs, non-collapsing backbone", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Prior-manuscript runs not evidenced."},
    {"id": 30, "location": "S7", "paper_text": "worth $+7.92$ points of mean AUROC across three training seeds", "paper_value": "+7.92 pts, 3 seeds", "evidence_file": "adbench/results/analysis_p1_p3.json", "evidence_value": "global mean 85.13; best-k 89.58; delta +4.45 pts, single run, no seed breakdown", "status": "ambiguous_mapping", "details": "Locality table delta is +3.3 to +4.5 pts, one run; 7.92 and 3 seeds untraceable."},
    {"id": 31, "location": "S7", "paper_text": "it fails below chance on \\texttt{screw}", "paper_value": "screw AUROC < 50% for mean", "evidence_file": "adbench/results/analysis_p1_p3.json", "evidence_value": "no category labels; no reduction-row value < 0.50", "status": "ambiguous_mapping", "details": "Cannot map columns to categories; no below-chance value for mean rows."},
    {"id": 32, "location": "S7", "paper_text": "top-$k$ pooling repairs that failure to $84\\%$", "paper_value": "84% on screw", "evidence_file": "adbench/results/analysis_p1_p3.json", "evidence_value": "no value = 0.84 in any reduction row", "status": "ambiguous_mapping", "details": "No 0.84 entry and no screw label; untraceable."},
    {"id": 33, "location": "S7", "paper_text": "The gain is robust across two orders of magnitude in $k$", "paper_value": "k spans 2 orders", "evidence_file": "adbench/results/analysis_p1_p3.json", "evidence_value": "k rows: 4, 2, 1, 0 (span < 1 order)", "status": "ambiguous_mapping", "details": "File k-range does not show two orders of magnitude."},
    {"id": 34, "location": "S7", "paper_text": "recovers $61\\%$ of the oracle gap ... range from $41\\%$ to $72\\%$", "paper_value": "61%; 41-72% seeds", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Oracle-gap recovery stats untraceable."},
    {"id": 35, "location": "S7", "paper_text": "the backbone effect is statistically detectable ($p = 0.039$)", "paper_value": "p = 0.039", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "p-value not in provided files."},
    {"id": 36, "location": "S7 practice", "paper_text": "with a $k$ of two percent, and expect to gain several points", "paper_value": "k = 2%, several points", "evidence_file": "none", "evidence_value": "absent", "status": "missing_evidence", "details": "Recommendation rests on untraceable reduction-gain evidence."}
  ],
  "segment_verdict": "FAIL",
  "notes": "Paper's cited source proxy_diag/phaseB_truth.json was not provided: table, variance shares, MDD, percentiles and p-value are all untraceable. Provided locality file shows a reduction delta of +3.3 to +4.5 pts (single run), materially below the claimed +7.92 across three seeds, though dataset scope (adbench vs MVTec-AD) is uncertain. Failed PatchCore baseline run does not contradict the paper, which claims no re-run of published baselines."
}
```
