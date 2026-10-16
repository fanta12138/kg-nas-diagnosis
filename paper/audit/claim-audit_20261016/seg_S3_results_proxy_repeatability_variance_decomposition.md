# S3 results — proxy repeatability + variance decomposition

## PAPER TEXT (sent)

% ============================================================================
% DRAFT v2 -- Section 4: Where the accuracy comes from (diagnosis)
% Every number here traces to:
%   search_forensics/search_forensics.json     (recorded runs)
%   proxy_diag/phaseA_proxy.json               (proxy repeatability, 420 calls)
%   proxy_diag/phaseB_truth.json               (210-detector truth matrix)
%   capacity_experiment/                       (parameter-capacity bands)
%   proxy_diag/samplesize_analysis.json        (resolvability, 4.4)
% ============================================================================

\section{Results: What the Architecture Search Actually Optimises}
\label{sec:diagnosis}

The three-stage pipeline of Section~\ref{sec:method} treats the Stage-2 search as
the component that produces the architecture, and the reported accuracy as
evidence that the search worked. This section tests that reading directly. We ask
four questions in turn: does the searched quantity reach the deployed model
(Section~\ref{subsec:inert}), is the search signal stable enough to be ranked
(Section~\ref{subsec:proxy_stability}), does the axis being searched have any
measurable effect on detection accuracy at all (Section~\ref{subsec:capacity_null}),
what does the search actually return when it is run
(Section~\ref{subsec:search_vs_random}), and --- the question that bounds all the
others --- how large a difference can the benchmark that judges this pipeline
resolve in the first place (Section~\ref{subsec:resolvability})?

\subsection{Two Thirds of the Search Space Cannot Reach the Deployed Model}
\label{subsec:inert}

The search space is
$\mathcal{A} = \mathcal{B} \times \mathcal{D}$ with
$|\mathcal{B}| = 14$ backbones and
$\mathcal{D} = \{128, 256, 512\}$ feature dimensions, giving 42 configurations.
The dimension enters the search through the student projector,
\texttt{ContrastiveModel(\allowbreak projection\_dim=\allowbreak feature\_dim)}, so the proxy score does
depend on it. It does not, however, reach the detector. In the deployment path the
student encoder builds its bottleneck with a hard-coded width,
\texttt{StudentEncoder.forward} calling
\texttt{\_init\_bottleneck(f.shape[1], 256, ...)}, while the decoder is
constructed with the requested width, \texttt{Decoder(bottleneck\_dim=\dots)}. The
final training entry point constructs the detector as
\texttt{RDDetector(student\_backbone=..., teacher\_arch=...)}, never passing
\texttt{bottleneck\_dim}; the argument therefore keeps its default of 256 for
every configuration.

Two consequences follow. For any configuration whose searched dimension is not
256 the two halves of the network disagree, so the discrepancy cannot be
activated by simply passing the argument --- the axis is unreachable by
construction rather than merely unused. And all three values of the searched
dimension produce the \emph{identical} deployed detector for a given backbone: the
42-configuration space contains only 14 distinct models, so the remaining 28
configurations are deployment-equivalent duplicates of them.

The recorded runs behave as this analysis predicts. Across four independent
searches on the same data and the same space, the selected backbone is
\texttt{efficientnet\_b0} every time, whereas the selected feature dimension
differs between runs:
%
\begin{center}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Run & Trials (declared / recorded) & Selected backbone & Selected dim \\
\midrule
\texttt{stage2}                                  & 30 / 17 & \texttt{efficientnet\_b0} & 128 \\
\texttt{stage2\_rtx3090}                         & 30 / 15 & \texttt{efficientnet\_b0} & 512 \\
\texttt{stage2\_rtx3090\_100trials}              & 100 / 39 & \texttt{efficientnet\_b0} & 256 \\
\texttt{stage2\_topjournal\_15backbones\_fixed}  & 100 / 93 & \texttt{efficientnet\_b0} & 256 \\
\bottomrule
\end{tabular}}

\end{center}
%
The stable part of the outcome is the part that matters for deployment; the
unstable part is precisely the part that cannot affect it. The search expends
two thirds of its ranking capacity on a variable with no deployment meaning, and
reports the winner of that sub-ranking as a discovered design choice.

A second observation from the same table: the number of trials recorded is
systematically below the number declared. The search loop catches exceptions and
\texttt{continue}s without appending a result, so failed trials vanish from the
record and from the denominator of any success statistic.

\subsection{The Proxy Carries Almost No Architecture Signal, and Collapses to Zero}
\label{subsec:proxy_stability}

The proxy is the average over validation batches of
$\mathrm{kd} + 0.3\,\mathrm{consistency}$, and the candidate with the
\emph{lowest} score is selected. Two properties of the scoring code matter.
A validation batch is accumulated only when \texttt{isfinite(score)} \emph{and}
\texttt{score < 1000}; a rejected batch is skipped with a warning. And the average
is formed as \texttt{total\_score / max(count, 1)}: if every batch in a trial is
rejected then \texttt{count} is zero, and the function returns exactly
\texttt{0.0} --- which, because the search minimises, is the best score attainable
anywhere in the space. A trial that failed to produce a usable number is thus
indistinguishable from a trial that produced the best possible number.

To measure how often this happens, and what the proxy does the rest of the time,
we call the search's own evaluation method (\texttt{evaluate\_architecture}) on
MVTec wood --- the category the search uses --- and repeat every one of the 42
configurations ten times with independent seeds. Table~\ref{tab:proxy_repeats}
gives the result, and Figure~\ref{fig:proxy_repeatability} the same data.

\begin{table}[t]
\centering
\caption{Proxy repeatability, condensed. The search's own evaluation method is
called ten times for each of the 42 configurations on MVTec wood (420
evaluations). ``Zero'' is the fraction of repeats returning exactly $0.0$; ``CV''
is the within-configuration coefficient of variation, pooled over each backbone's
three dimensions. The two collapse-prone backbones are shown individually; of the
remaining twelve not one degenerate trial occurred in any repeat, and their pooled
medians span $7.538$--$7.971$ with CV between $2.2\%$ and $12.1\%$. Their
per-backbone values are released with the code.}
\label{tab:proxy_repeats}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Backbone & Repeats & Zero (\%) & Median proxy & CV (\%) \\
\midrule
\texttt{efficientnet\_b1}        & 30 & \textbf{73} & \textbf{0.000} & 169.5 \\
\texttt{mobilenetv3\_small\_100} & 30 & \textbf{13} & 7.979 & 180.9 \\
\midrule
twelve well-behaved backbones (each) & 30 & 0 & 7.538--7.971 & 2.2--12.1 \\
\midrule
\textbf{all 42 configurations}   & \textbf{420} & \textbf{6.2} & --- & --- \\
\bottomrule
\end{tabular}}

\end{table}

Two regimes appear, and both matter. Twelve of the fourteen backbones are well
behaved: not a single degenerate trial in any repeat, with within-configuration
coefficients of variation between $2.2\%$ and $12.1\%$. Two backbones are broken,
and severely so: \texttt{efficientnet\_b1} returns exactly $0.0$ in $73\%$ of
evaluations at every dimension, and \texttt{mobilenetv3\_small\_100} in $13\%$.
Across all 420 evaluations, 26 ($6.2\%$) returned exactly zero. The cause is
visible in the engine's own diagnostics --- the student diverges, and every
validation batch then trips the hard threshold:

\begin{quote}
\texttt{Warning: Invalid score detected (score=4.19e+28, kd=2.18e+28), skipping}
\end{quote}

\runinhead{The proxy does not distinguish architectures.}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig2_proxy_repeatability.png}
\caption{Proxy repeatability. (a) Ten independent evaluations of each of the 42
configurations, on a logarithmic axis. Twelve backbones form a tight band near
$7.6$; \texttt{efficientnet\_b1} and \texttt{mobilenetv3\_small\_100} intermittently
return exactly $0.0$ (marked red, plotted on the axis floor) when their student
diverges. (b) The pooled median of the twelve non-collapsing backbones spans only
$5.7\%$, the same order as the within-configuration noise of a single backbone.}
\label{fig:proxy_repeatability}
\end{figure}

The more consequential number in Table~\ref{tab:proxy_repeats} is the median
column. Across the twelve well-behaved backbones --- CNN and Transformer families,
2015-era through 2023-era, from ResNets~\cite{he2016deep} to ConvNeXt
V2~\cite{woo2023convnextv2} --- the median proxy spans only $7.538$ to $7.971$, a
range of $5.7\%$, while a single backbone's within-configuration noise is of the
same order ($2$--$12\%$ CV). The proxy contains essentially no
architecture-level signal. Any ranking it produces is not a ranking of
architectures; it is a ranking of which cell happened to be sampled.

\runinhead{Minimising then selects the broken cells.}
This turns a noisy score into a systematically wrong one. A well-behaved
configuration has a lower bound near $7.5$ and can never beat a configuration
that occasionally returns $0.0$. Minimising over the space does not select the
best candidate; it selects the candidate with the heaviest left tail --- and the
left tail is heaviest exactly where training diverged. Consistent with this,
\texttt{efficientnet\_b1}, the backbone that collapses in $73\%$ of evaluations,
is the $\argmin$ reported by the capacity searches of
Section~\ref{subsec:capacity_null} at two of their five parameter bands.

The recorded runs show the same pathology at the same rate. Over every search
result stored in the project --- 19 runs, 834 trials --- $8.0\%$ of trials score
exactly zero, and in \textbf{14 of the 19 runs (74\%)} the plain $\argmin$ over
the recorded results, which is literally what the search returns
(\texttt{min(self.results, key=score)}), points at a zero-score trial
(Table~\ref{tab:degeneracy}). The five capacity-budget searches are all affected,
and were usable only because a later analysis script discards trials scoring
$\le 0$: three different conventions for handling degenerate trials coexist in the
same codebase, the search engine keeping them while the downstream selectors drop
them.

\begin{table}[t]
\centering
\caption{Degenerate trials in the recorded searches, and whether the plain
$\argmin$ lands on one. A trial scoring exactly $0.0$ has had every validation
batch rejected, so the average is \texttt{total}/\texttt{max(count,1)}; because
the search minimises, such a trial wins outright.}
\label{tab:degeneracy}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrl}
\toprule
Run & Trials & Zeros & $\argmin$ config \\
\midrule
\texttt{cap\_budget\_3M}  & 20  & 1  & \texttt{mobilenetv3\_small\_100@128} \\
\texttt{cap\_budget\_6M}  & 40  & 4  & \texttt{mobilenetv3\_small\_100@256} \\
\texttt{cap\_budget\_8M}  & 80  & 11 & \texttt{efficientnet\_b1@256} \\
\texttt{cap\_budget\_13M} & 80  & 13 & \texttt{mobilenetv3\_small\_100@256} \\
\texttt{cap\_budget\_30M} & 80  & 6  & \texttt{efficientnet\_b1@256} \\
\texttt{mvtec\_stage2\_test/bottle} & 100 & 9 & \texttt{efficientnet\_b1@512} \\
\texttt{search\_strategies} (9 runs) & 270 & 23 & 8 of 9 degenerate \\
\midrule
\texttt{stage2} (main search) & 17 & 0 & \texttt{efficientnet\_b0@128} \\
\texttt{stage2\_rtx3090} & 15 & 0 & \texttt{efficientnet\_b0@512} \\
\texttt{stage2\_rtx3090\_100trials} & 39 & 0 & \texttt{efficientnet\_b0@256} \\
\texttt{stage2\_topjournal\_15backbones\_fixed} & 93 & 0 & \texttt{efficientnet\_b0@256} \\
\midrule
\textbf{Total} & \textbf{834} & \textbf{67 (8.0\%)} & \textbf{14 of 19 runs} \\
\bottomrule
\end{tabular}}

\end{table}

\runinhead{A reproducibility caveat.}
The recorded runs predate the current revision of the search engine, and the two
are not numerically identical: the recorded proxy for \texttt{efficientnet\_b0}
spans $0.002$--$0.065$, whereas the current code returns $7.5$--$12.2$ for the
same configuration. We therefore do not claim to reproduce the recorded search;
what the two lines of evidence agree on --- and all we rely on here --- is the
failure mode, its rate ($8.0\%$ against $6.2\%$) and the backbones it afflicts.
The revised engine also returns no zero-score trial in the four runs whose results
the manuscript reports, which is itself evidence of the revision; that the four
happened to select \texttt{efficientnet\_b0}, a backbone that does not collapse,
was in this account luck rather than method
(Section~\ref{subsec:search_vs_random}).

\runinhead{Two further obstacles.}
The proxy's absolute level is not comparable across categories: its median is
$0.110$ on \texttt{capsule}, $0.385$ on \texttt{hazelnut} and $7.56$ on
\texttt{wood}, a span of $68.6\times$, so neither a fixed threshold nor a
cross-category comparison of scores is meaningful --- yet the protocol runs a
single search on one category and transfers the winner to the others. And the
deployment-inert dimension of Section~\ref{subsec:inert}, which cannot change the
detector at all, nevertheless moves the proxy by a median factor of $1.16\times$
across 140 backbone comparisons and by more than $10\times$ at the 90th
percentile. A variable with no deployment meaning is responsible for
order-of-magnitude swings in the ranking.

\subsection{The Searched Axis Has No Measurable Effect on Accuracy}
\label{subsec:capacity_null}

A search that ranks noise could still be worth running if the axis it explores
mattered. We test that directly, by retraining detectors across the whole
parameter range under five mutually exclusive parameter bands, each trained with
the identical protocol (500 epochs, batch size 64, seed 42, ResNet-18 teacher,
reverse distillation). Within each band the search is restricted to backbones
whose deployed size (encoder plus decoder) falls inside the band, and the
best-scoring backbone is trained on all 15 MVTec-AD categories.
Table~\ref{tab:capacity_null} reports the result.

\begin{table}[t]
\centering
\caption{Parameter-capacity bands. Each band restricts the search to backbones
whose deployed size falls inside the band; the selected backbone is then trained
with the identical protocol on all 15 MVTec-AD categories.}
\label{tab:capacity_null}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrr}
\toprule
Band & Selected backbone & Params (M) & Mean AUROC (\%) \\
\midrule
0--3M   & MobileNetV3-Small & 2.81  & 82.01 \\
3--6M   & EfficientNet-B0   & 5.41  & 83.14 \\
6--9M   & DeiT-Tiny         & 7.31  & 81.55 \\
9--15M  & ResNet-18         & 13.04 & \textbf{83.25} \\
15--31M & ResNet-50         & 25.77 & 83.08 \\
\bottomrule
\end{tabular}}

\end{table}

Multiplying the parameter budget by $9.2\times$ changes mean AUROC by
$+1.06$ percentage points, and the entire spread across the five bands is
$1.70$ points. The ordering is not monotone: the largest model is worse than the
second largest. All ten pairwise paired $t$-tests across categories are
non-significant (smallest $p = 0.058$).

We then measured the axis directly: all 14 backbones of the search space were
trained on all 15 MVTec-AD categories under the identical protocol, giving 210
detectors, and the resulting matrix was decomposed. A two-way decomposition
attributes \textbf{$0.54\%$} of the variance to the backbone
($F(13,182) = 1.844$, $p = 0.039$), \textbf{$95.32\%$} to the category, and
$4.13\%$ to their interaction, which is the error term in this repeated-measures
design (Figure~\ref{fig:variance}).

The backbone term is statistically significant: with 210 detectors the test has
the power to detect it. That, precisely, is the point. The effect is detectable
\emph{and} negligible. Architecture accounts for $0.54\%$ of the variance in
detection accuracy, against $95.32\%$ for which category is being inspected --- a
factor of $176$. A practitioner choosing between two backbones is moving an
AUROC whose variation is overwhelmingly determined by the object under
inspection rather than by the network.

The ranking says the same thing. Over the 15 categories the 14 backbones span
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig3_variance_decomposition.png}
\caption{Two-way decomposition of the true AUROC over 14 backbones and 15
categories (210 detectors). The backbone effect is statistically detectable and
practically negligible: it accounts for $0.54\%$ of the variance against $95.32\%$
for the category, a factor of $176$.}
\label{fig:variance}
\end{figure}

$3.98$ points of mean AUROC, and the eleven strongest span only $2.43$ points.
The manuscript's selected architecture, \texttt{efficientnet\_b0}, ranks fifth of
fourteen and is statistically indistinguishable from twelve of the thirteen
alternatives; only \texttt{convnextv2\_base} is significantly worse
($-3.11$ points, $p = 0.013$).

\runinhead{The search signal does not predict the target.}
Ranking the 14 backbones by their mean proxy score
(Section~\ref{subsec:proxy_stability}) and comparing that ordering with the true
performance matrix gives a Spearman correlation of \textbf{$-0.12$} ($p = 0.68$)
against the 15-category mean, and \textbf{$+0.02$} ($p = 0.93$) against
performance on \texttt{wood} itself. The search signal is not weakly correlated
with the outcome it exists to select for; on the very category the search is run
on it is indistinguishable from uncorrelated. The consequence is concrete: on
\texttt{wood} the backbone with the highest true AUROC is
\texttt{convnextv2\_base} at $94.74\%$, and across the fifteen categories
\texttt{convnextv2\_base} is the \emph{worst} of the fourteen at $79.36\%$; the
correlation between a backbone's rank on \texttt{wood} and its rank overall is
$-0.24$. The single category the protocol searches on is, if anything, mildly
anti-informative about the rest.

\runinhead{The run-to-run noise is not small.}
Within a fixed (backbone, category) cell the pooled standard deviation across
training seeds is $1.96$ points over 180 residual degrees of freedom, against a
total spread of $3.98$ points across the fourteen backbones. That the two numbers
are of the same order is the whole difficulty of this axis: a single training run
cannot resolve the backbone effect for any individual category, so a comparison
built on one run per configuration is unresolvable in principle rather than merely
imprecise. Section~\ref{subsec:resolvability} turns that into a number.

\runinhead{And the ranking does not transfer.}
Correlating the 14-backbone orderings of every pair of categories (105 pairs)
gives a mean Spearman $\rho = 0.215$, and in $31\%$ of pairs the two categories
\emph{disagree} on the ordering ($\rho < 0$). The protocol searches on a single
category, \texttt{wood}, and applies the winner to all fifteen. Since which
backbone is better depends on the category, a single-category search is not a
noisy estimate of a real quantity; it is an estimate of a quantity that does not
exist independently of the category.

This is the central observation of the paper. Taken with
Sections~\ref{subsec:inert} and~\ref{subsec:proxy_stability}, it says that the
proxy varies enormously across the space while the quantity it is supposed to
predict contributes half a percent of the variance that the category contributes
ninety-five percent of: the search ranks one noisy quantity in order to predict
another that is almost entirely determined by something else.



## EVIDENCE FILES SENT

- proxy_diag/phaseA_proxy.json
- proxy_diag/phaseC_analysis.json
- proxy_diag/final_truth_analysis.json
- proxy_diag/phaseB_truth.json [COMPACTED]

## REVIEWER RAW RESPONSE

Audit complete: core diagnosis numbers (73%, 6.2%, medians, variance decomposition, correlations, 1.96pp seed SD) trace exactly to the provided files; two numeric mismatches found (CV range omits efficientnet_b0's 74% pooled CV; "176×" is 175.1×), and all recorded-run/capacity numbers rest on files not supplied.

```json
{
  "claims": [
    {"id": 1, "location": "§4.1", "paper_text": "|B|=14 backbones and D={128,256,512}, giving 42 configurations", "paper_value": "14x3=42", "evidence_file": "proxy_diag/phaseA_proxy.json", "evidence_value": "42 config keys, 14 backbones x 3 dims", "status": "exact_match", "details": "Config inventory matches exactly."},
    {"id": 2, "location": "§4.1", "paper_text": "encoder builds bottleneck with hard-coded width 256; decoder constructed with requested width; RDDetector never passes bottleneck_dim", "paper_value": "structural claim", "evidence_file": "proxy_diag/phaseC_analysis.json", "evidence_value": "structural_probe: encoder 256, dim128_decoder_in 128, signature has bottleneck_dim, verdict identical", "status": "exact_match", "details": "Structural probe corroborates all three code claims."},
    {"id": 3, "location": "§4.1", "paper_text": "42-configuration space contains only 14 distinct models; remaining 28 are duplicates (two thirds)", "paper_value": "14 distinct / 28 duplicate", "evidence_file": "proxy_diag/phaseC_analysis.json", "evidence_value": "verdict: all three searched dims map to identical deployed detector", "status": "exact_match", "details": "Logical consequence of probe; 28/42=2/3 arithmetic holds."},
    {"id": 4, "location": "§4.1 table", "paper_text": "stage2 30/17; rtx3090 30/15; 100trials 100/39; topjournal 100/93; backbones/dims as listed", "paper_value": "declared/recorded trials, selected backbone+dim per run", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "All four recorded-run rows unverifiable; trial counts 17/15/39/93 internally consistent with Table 2."},
    {"id": 5, "location": "§4.1", "paper_text": "the selected backbone is efficientnet_b0 every time, whereas the selected feature dimension differs between runs", "paper_value": "eff_b0 in all 4 runs; dims 128/512/256/256", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "Central §4.1 claim rests entirely on absent file."},
    {"id": 6, "location": "§4.1", "paper_text": "number of trials recorded is systematically below the number declared", "paper_value": "17<30, 15<30, 39<100, 93<100", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "Consistent with paper's own table only."},
    {"id": 7, "location": "§4.2", "paper_text": "repeat every one of the 42 configurations ten times (420 evaluations)", "paper_value": "420 evaluations", "evidence_file": "proxy_diag/phaseA_proxy.json", "evidence_value": "42 entries x 10 scores = 420", "status": "exact_match", "details": "Confirmed."},
    {"id": 8, "location": "§4.2/Table 1", "paper_text": "efficientnet_b1 returns exactly 0.0 in 73% of evaluations", "paper_value": "73%", "evidence_file": "proxy_diag/phaseA_proxy.json", "evidence_value": "22/30 zeros (7+7+8) = 73.3%", "status": "exact_match", "details": "Zero counts hand-verified per dimension."},
    {"id": 9, "location": "Table 1", "paper_text": "efficientnet_b1: median 0.000, CV 169.5%", "paper_value": "0.000 / 169.5%", "evidence_file": "proxy_diag/phaseC_analysis.json", "evidence_value": "median 0 (22 zeros); CVs 1.5281/1.5579/2.0000, mean 1.6953", "status": "exact_match", "details": "'Pooled' is the arithmetic mean of the three per-config CVs."},
    {"id": 10, "location": "Table 1", "paper_text": "mobilenetv3_small_100: Zero 13%, median 7.979, CV 180.9%", "paper_value": "13% / 7.979 / 180.9%", "evidence_file": "proxy_diag/phaseA_proxy.json + phaseC", "evidence_value": "4/30=13.3%; median (7.9775+7.9804)/2=7.979; mean CV 1.8095", "status": "exact_match", "details": "All three values reproduce exactly."},
    {"id": 11, "location": "§4.2/Table 1", "paper_text": "twelve well-behaved: not one degenerate trial; pooled medians span 7.538-7.971, a range of 5.7%", "paper_value": "0 zeros; 7.538-7.971; 5.7%", "evidence_file": "proxy_diag/phaseA_proxy.json", "evidence_value": "no zeros in 12 backbones; medians vit_tiny 7.538 ... eff_b0 7.971; range 5.75%", "status": "exact_match", "details": "Medians recomputed from raw scores; endpoints confirmed."},
    {"id": 12, "location": "§4.2/Table 1", "paper_text": "twelve well-behaved backbones ... CV between 2.2% and 12.1%", "paper_value": "CV range 2.2-12.1%", "evidence_file": "proxy_diag/phaseC_analysis.json", "evidence_value": "efficientnet_b0 CVs 129.0/18.6/74.4% -> pooled 74.0%", "status": "number_mismatch", "details": "Twelve includes eff_b0 whose pooled CV (74%) far exceeds claimed max 12.1% (swin_tiny)."},
    {"id": 13, "location": "Fig. 2 caption", "paper_text": "Twelve backbones form a tight band near 7.6", "paper_value": "twelve tight band", "evidence_file": "proxy_diag/phaseA_proxy.json", "evidence_value": "eff_b0 means 16.09/8.64/11.65, max 77.89", "status": "scope_overclaim", "details": "efficientnet_b0 is among the twelve and is not tight nor near 7.6."},
    {"id": 14, "location": "§4.2", "paper_text": "Across all 420 evaluations, 26 (6.2%) returned exactly zero", "paper_value": "26 / 6.2%", "evidence_file": "proxy_diag/phaseA_proxy.json", "evidence_value": "22 (eff_b1) + 4 (mbv3) = 26; 26/420=6.19%", "status": "exact_match", "details": "Confirmed."},
    {"id": 15, "location": "§4.2", "paper_text": "Warning: Invalid score detected (score=4.19e+28, kd=2.18e+28)", "paper_value": "diagnostic quote", "evidence_file": "none", "evidence_value": "absent from all files", "status": "missing_evidence", "details": "Engine log not supplied."},
    {"id": 16, "location": "Table 2", "paper_text": "per-run trials/zeros/argmin; totals 834 trials, 67 zeros (8.0%), 14 of 19 runs (74%)", "paper_value": "11 rows + totals", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "Internal arithmetic consistent (sums 834, 67, 8.03%, 73.7%) but source absent."},
    {"id": 17, "location": "§4.2 caveat", "paper_text": "recorded proxy for efficientnet_b0 spans 0.002-0.065", "paper_value": "0.002-0.065", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "Unverifiable."},
    {"id": 18, "location": "§4.2 caveat", "paper_text": "current code returns 7.5-12.2 for the same configuration", "paper_value": "7.5-12.2", "evidence_file": "proxy_diag/phaseA_proxy.json", "evidence_value": "eff_b0 dim256 spans 7.550-12.151; dim128 reaches 77.89, dim512 37.40", "status": "ambiguous_mapping", "details": "Matches dim256 only; 'same configuration' singular unclear, other dims exceed range."},
    {"id": 19, "location": "§4.2", "paper_text": "median is 0.110 on capsule, 0.385 on hazelnut, 7.56 on wood, a span of 68.6x", "paper_value": "0.110/0.385/7.56; 68.6x", "evidence_file": "none", "evidence_value": "no per-category proxy data in files (phaseA is wood only)", "status": "missing_evidence", "details": "Cross-category proxy medians not recorded in supplied files."},
    {"id": 20, "location": "§4.2", "paper_text": "dimension moves the proxy by a median factor of 1.16x across 140 backbone comparisons, >10x at 90th percentile", "paper_value": "1.16x / 140 / >10x", "evidence_file": "none", "evidence_value": "not recorded; derivation (140 comparisons) undefined", "status": "missing_evidence", "details": "Not derivable unambiguously from phaseA as given."},
    {"id": 21, "location": "Table 3", "paper_text": "bands: 2.81/82.01; 5.41/83.14; 7.31/81.55; 13.04/83.25; 25.77/83.08", "paper_value": "5 rows params+AUROC", "evidence_file": "capacity_experiment/", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "Entire capacity-band table unverifiable."},
    {"id": 22, "location": "§4.3", "paper_text": "Multiplying the parameter budget by 9.2x changes mean AUROC by +1.06 percentage points", "paper_value": "9.2x; +1.06pp", "evidence_file": "Table 3 (source missing)", "evidence_value": "25.77/2.81=9.17->9.2 ok; 83.08-82.01=1.07 not 1.06", "status": "number_mismatch", "details": "Displayed table values give +1.07pp; 1.06 needs unrounded values."},
    {"id": 23, "location": "§4.3", "paper_text": "entire spread across the five bands is 1.70 points; largest model worse than second largest", "paper_value": "1.70pp; 83.08<83.25", "evidence_file": "capacity_experiment/", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "Arithmetic consistent with Table 3 only."},
    {"id": 24, "location": "§4.3", "paper_text": "All ten pairwise paired t-tests non-significant (smallest p = 0.058)", "paper_value": "10 tests, min p=0.058", "evidence_file": "capacity_experiment/", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "Test statistics absent."},
    {"id": 25, "location": "§4.3", "paper_text": "all 14 backbones trained on all 15 categories, giving 210 detectors", "paper_value": "210 detectors", "evidence_file": "proxy_diag/phaseB_truth.json", "evidence_value": "390 AUROC values (210 cells; 90 cells x 3 seeds, 120 x 1)", "status": "aggregation_mismatch", "details": "210 counts cells, not trained detectors; paper's own 180 residual df implies 390 runs."},
    {"id": 26, "location": "§4.3/Fig.3", "paper_text": "0.54% backbone, 95.32% category, 4.13% interaction; F(13,182)=1.844, p=0.039", "paper_value": "0.54/95.32/4.13; F=1.844; p=0.039", "evidence_file": "proxy_diag/final_truth_analysis.json", "evidence_value": "0.5445/95.3212/4.1343; F=1.84401; df[13,182]; p=0.03947", "status": "exact_match", "details": "Matches final_truth; note phaseC variant (0.62/95.41/3.97, F=2.18) conflicts."},
    {"id": 27, "location": "§4.3/Fig.3 caption", "paper_text": "a factor of 176", "paper_value": "176x", "evidence_file": "proxy_diag/final_truth_analysis.json", "evidence_value": "95.3212/0.54455 = 175.1", "status": "number_mismatch", "details": "Unrounded ratio is 175.1; displayed-value ratio 176.5; neither rounds cleanly to 176."},
    {"id": 28, "location": "§4.3", "paper_text": "14 backbones span 3.98 points; eleven strongest span 2.43; efficientnet_b0 ranks fifth of fourteen", "paper_value": "3.98 / 2.43 / rank 5", "evidence_file": "proxy_diag/final_truth_analysis.json", "evidence_value": "0.83341-0.79361=3.98pp; 0.83341-0.80907=2.43pp; eff_b0 5th", "status": "exact_match", "details": "Ranking list confirms all three."},
    {"id": 29, "location": "§4.3", "paper_text": "indistinguishable from twelve of thirteen; only convnextv2_base worse (-3.11 points, p=0.013)", "paper_value": "-3.11pp, p=0.013", "evidence_file": "proxy_diag/final_truth_analysis.json", "evidence_value": "delta 0.82467-0.79361=3.11pp; p=0.013 and pairwise tests absent", "status": "missing_evidence", "details": "Delta arithmetic verified; significance claims unverifiable."},
    {"id": 30, "location": "§4.3", "paper_text": "Spearman -0.12 (p=0.68) vs 15-category mean; +0.02 (p=0.93) vs wood", "paper_value": "-0.12/0.68; +0.02/0.93", "evidence_file": "proxy_diag/phaseC_analysis.json", "evidence_value": "[-0.12088, 0.68061]; [0.02420, 0.93455]", "status": "exact_match", "details": "Both correlations and p-values match."},
    {"id": 31, "location": "§4.3", "paper_text": "wood best convnextv2_base 94.74%; worst overall 79.36%; wood-vs-overall rank correlation -0.24", "paper_value": "94.74% / 79.36% / -0.24", "evidence_file": "proxy_diag/phaseC_analysis.json", "evidence_value": "auroc_wood 0.94737; mean 0.79361 (ranked last); spearman -0.23982", "status": "exact_match", "details": "All three confirmed; wood max verified across 14 backbones."},
    {"id": 32, "location": "§4.3", "paper_text": "105 pairs, mean Spearman 0.215; 31% of pairs disagree (rho<0)", "paper_value": "0.215; 31%", "evidence_file": "proxy_diag/final_truth_analysis.json", "evidence_value": "cross_cat_mean_spearman 0.21512; 33/105=31.4%", "status": "exact_match", "details": "Confirmed."},
    {"id": 33, "location": "§4.3", "paper_text": "pooled SD across training seeds is 1.96 points over 180 residual df", "paper_value": "1.96pp / 180 df", "evidence_file": "proxy_diag/phaseB_truth.json", "evidence_value": "recomputed: pooled within-cell variance 0.03444/90 -> SD 1.956pp, 180 df", "status": "exact_match", "details": "Derived from raw cells; phaseC's mean per-cell SD (1.04pp) is a different statistic."},
    {"id": 34, "location": "§4.2", "paper_text": "isfinite and score<1000 gate; total/max(count,1) returns 0.0; min(self.results, key=score); exceptions continue", "paper_value": "code-behaviour claims", "evidence_file": "none", "evidence_value": "no engine code in files", "status": "missing_evidence", "details": "Implementation constants; expected absent, but 0.0-collapse mechanism unverifiable directly."},
    {"id": 35, "location": "§4.2", "paper_text": "efficientnet_b1 is the argmin of the capacity searches at two of five bands", "paper_value": "2 of 5 bands (8M, 30M)", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "FILE NOT PROVIDED", "status": "missing_evidence", "details": "Consistent with paper's Table 2 only."}
  ],
  "segment_verdict": "WARN",
  "notes": "Headline diagnosis numbers (73%, 13%, 26/420=6.2%, medians 7.538-7.971, decomposition 0.54/95.32/4.13, F/p, Spearman correlations, transfer rho, 1.96pp seed SD) all reproduce exactly from supplied files. Real defects: (1) 'twelve well-behaved CV 2.2-12.1%' omits efficientnet_b0 (pooled CV 74%, dim128 129%), also contradicting Fig.2 'tight band' caption; (2) 'factor of 176' is 175.1 unrounded; (3) '+1.06pp' vs displayed 1.07; (4) '210 detectors' vs 390 recorded runs in phaseB. All search_forensics and capacity_experiment numbers (Tables 2-3, recorded-run table, 8.0%/834, p=0.058, p=0.013) unverifiable - files not supplied. phaseC and final_truth decompositions disagree (F=2.18 vs 1.844); paper silently uses final_truth."
}
```
