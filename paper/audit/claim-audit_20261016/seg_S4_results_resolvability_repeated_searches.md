# S4 results — resolvability + repeated searches

## PAPER TEXT (sent)

\subsection{What This Benchmark Design Can Resolve}
\label{subsec:resolvability}

The preceding subsections diagnose the search. This one asks a different
question, about the benchmark rather than about the pipeline: \emph{how small a
difference can the standard evaluation design detect?} The question is forced by
the state of the field. On MVTec-AD, PatchCore reached $99.1\%$ image-level
AUROC~\cite{roth2022patchcore} and the strongest methods reported since reach
$99.6\%$~\cite{liu2023simplenet,guo2025dinomaly}; the contested margin between
successive generations is $0.5$ points, on test sets holding roughly a hundred
images per category. If published comparisons cannot resolve differences of that
size, the ranking they produce is not a ranking of methods.

The 210-detector matrix of Section~\ref{subsec:capacity_null} and the replicated
cells within it supply every quantity the calculation needs. Three numbers come
out of them, all in AUROC points and all computed by
\texttt{proxy\_diag/samplesize\_analysis.py}:

\begin{itemize}
\item $\sigma_{\text{seed}} = 1.96$, the pooled within-cell standard deviation
over training seeds ($90$ cells replicated three times; $180$ residual degrees of
freedom). The distribution of per-cell standard deviations is heavily skewed ---
median $0.58$, maximum $7.74$ --- so the pooled figure is the honest single number
and the median understates the tail.
\item $\sigma_{\text{res}} = 3.30$, the two-way ANOVA residual on the single-seed
matrix. This is the working error for a comparison \emph{between} backbones,
because it also contains the backbone $\times$ category interaction that the
seed-only figure excludes.
\item $\sigma_{\Delta} = 4.12$, the empirical standard deviation of the
per-category difference between two backbones, averaged over the $91$ backbone
pairs. This is what a paired comparison across categories actually sees.
\end{itemize}

Two comparisons follow, and both are unfavourable to the design in common use. On
a \emph{single category}, with two backbones trained independently, the difference
has resolution $\sigma_{\text{seed}} = 1.96$ per arm; detecting the $0.5$-point
contested margin at $80\%$ power with a two-sided test at $\alpha = 0.05$ requires
\textbf{$240$ independent training runs per backbone}. Paired instead across the
fifteen categories the benchmark provides, the minimum detectable difference is
\textbf{$2.98$ points} --- larger than the entire $3.98$-point spread of the
fourteen backbones, and six times the margin the field argues about. Adding two
more training seeds per cell, as we do here, moves this only to $2.95$ points. To
resolve $0.5$ points by pairing over categories would take $532$ categories; the
benchmark has fifteen.

The requirement is also method-specific, which is a further reason it has to be
reported rather than assumed. Running the same calculation on a PatchCore-style
memory-bank detector, benchmarked through the harness released with this paper,
gives a seed standard deviation of $0.51$ points and a requirement of only $17$
runs per backbone --- a factor of $14$ below the reverse-distillation pipeline
studied here. No single number of runs is therefore sufficient in this field; the
figure belongs with the method it describes.

The implication is not that this benchmark is worthless, but that a
single-category, single-run comparison on it cannot support a claim at the
resolution the literature reports. A backbone ranking must either be averaged over
many categories --- which is what Table~\ref{tab:controlled} does, and why that
table needed $210$ detectors --- or be reported with an explicit statement of what
it can and cannot distinguish. This is also why the architecture effect, though
statistically detectable at $210$ detectors ($p = 0.039$), cannot be
\emph{ranked} reliably by any smaller design: the power that makes the effect
visible is the same power the field's usual single-category design lacks. We adopt
this discipline for the rest of the paper, making every backbone comparison on the
15-category mean and reporting differences smaller than about three points as
unresolved rather than as rankings.

\subsection{Running the Search Selects the Broken Backbones}
\label{subsec:search_vs_random}

The preceding subsections describe a search space that is two thirds inert, a
signal that does not discriminate between architectures and collapses to a score
of exactly zero for two of them, and a target quantity that architecture barely
moves. This subsection asks what they amount to in practice, by simply running the
search. We invoke the search engine's own entry point
(\texttt{NASSearchEngine.search}) with the manuscript's settings --- 100 trials,
three quick training epochs per candidate, uniform sampling from the 42 cells ---
and repeat it with twelve independent seeds. Because the searched feature
dimension cannot reach the detector (Section~\ref{subsec:inert}), the deployed
model depends only on the selected backbone, so the true quality of a run's output
can be read directly off the truth matrix of
Section~\ref{subsec:capacity_null}. Every run returned one of the two backbones
that collapse to zero: eight of the twelve returned \texttt{efficientnet\_b1}
(true rank 12 of 14, $80.43\%$ mean AUROC) and four returned
\texttt{mobilenetv3\_small\_100} (rank 7 of 14, $81.85\%$). None of the other
twelve backbones was ever selected, and only 2 of the 14 backbones and 6 of the 42
configurations appeared at all. Under uniform selection the probability of twelve
runs all landing inside that two-element subset is $7.2\times10^{-11}$. The search
does not merely fail to find a good architecture; it is reliably drawn to the two
configurations that break.

The consequence is quantitative, and it is not marginal
(Figure~\ref{fig:search_vs_random}). Averaged over the twelve
runs the search's output attains \textbf{80.91}\% mean AUROC. Picking a backbone
uniformly at random from the same space gives \textbf{81.65}\% (bootstrap
s.d.\ 0.32 points), and the best backbone in the space gives \textbf{83.34}\%.
The search result sits at the \textbf{1.2th percentile} of the random-selection
distribution --- it is worse than $98.8\%$ of uniform draws --- with a mean true
rank of $10.3$ of $14$ against $7.5$ for random, and it never once selected one of
the four strongest backbones. Replacing the search with a random number generator
would have improved the deployed detector; replacing it with the constant
\texttt{resnet18} would have improved it by $2.4$ points.

The mechanism is the one established in Section~\ref{subsec:proxy_stability}: a
well-behaved configuration is bounded below by roughly $7.5$, while the two
collapse-prone backbones return exactly $0.0$ whenever their student diverges, so
minimising a single sample per configuration selects the heaviest left tail in the
space --- and the heaviest left tails belong to the configurations that failed.
More trials make this more certain rather than less, because every additional
sample of a collapse-prone backbone is another chance to return the minimum of the
space. Our recommendation therefore follows from the mechanism rather than from
any single run: \textbf{a search whose objective can silently return the optimum
of its own range must filter degenerate evaluations before minimising, and should
report the distribution of its per-configuration scores rather than a single
minimum.}

\runinhead{Why the earlier searches looked successful.}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig4_search_vs_random.png}
\caption{The search against chance. (a) Bootstrap distribution of the true mean
AUROC obtained by selecting a backbone uniformly at random from the same space,
with the search's output and the oracle marked. The search result sits at the
$1.2$th percentile. (b) True rank of the backbone returned by each of the twelve
runs: the four strongest backbones were never selected.}
\label{fig:search_vs_random}
\end{figure}

The four search runs whose results the manuscript reports selected
\texttt{efficientnet\_b0} (Section~\ref{subsec:inert}), which does not collapse.
That this is luck rather than method is the point of
Section~\ref{subsec:search_vs_random}, and it is the reason we report both lines
of evidence rather than either alone.


## EVIDENCE FILES SENT

- proxy_diag/samplesize_analysis.json
- adbench/results/power_summary.json
- search_repeats/search_repeats.json
- search_repeats/search_repeats_summary.json
- search_repeats/search_vs_random_final.json
- search_forensics/search_forensics.json [COMPACTED]

## REVIEWER RAW RESPONSE

Short audit: most figures trace exactly to `samplesize_analysis.json` and `search_vs_random_final.json`, but the "2.98 larger than 3.98" comparison is arithmetically false, the mobilenetv3 AUROC (81.85%) conflicts with the only per-backbone truth provided (81.95%), and the PatchCore comparison is unsupported by any given file.

```json
{
  "claims": [
    {"id": 1, "location": "resolvability item 1", "paper_text": "sigma_seed = 1.96, pooled within-cell SD over training seeds (90 cells replicated three times; 180 residual df)", "paper_value": "1.96 / 90 / 180", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "pooled_sd=1.9559; n_multiseed_cells=90; df=180.0", "status": "exact_match", "details": "All three components match; 1.9559 rounds to 1.96."},
    {"id": 2, "location": "resolvability item 1", "paper_text": "median 0.58, maximum 7.74", "paper_value": "0.58 / 7.74", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "median_of_cell_sd=0.5755; max_of_cell_sd=7.7409", "status": "exact_match", "details": "Standard rounding to displayed precision."},
    {"id": 3, "location": "resolvability item 2", "paper_text": "sigma_res = 3.30, two-way ANOVA residual", "paper_value": "3.30", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "sigma_anova_residual=3.29856", "status": "exact_match", "details": "sqrt(ms_res)=3.2986 pp; rounds to 3.30."},
    {"id": 4, "location": "resolvability item 3", "paper_text": "sigma_delta = 4.12 ... averaged over the 91 backbone pairs", "paper_value": "4.12 / 91", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "paired_diff_sd_pp.mean=4.11727; n_pairs=91", "status": "exact_match", "details": "14x13/2=91 pairs; mean 4.1173 rounds to 4.12."},
    {"id": 5, "location": "resolvability intro", "paper_text": "The 210-detector matrix", "paper_value": "210", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "n_cells=210", "status": "exact_match", "details": "Matches; 14 backbones x 15 categories."},
    {"id": 6, "location": "resolvability para 2", "paper_text": "requires 240 independent training runs per backbone", "paper_value": "240", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "n_per_arm_single_category_unpaired=240.208", "status": "exact_match", "details": "Rounds to 240; formula consistent with sigma=1.956, delta=0.5."},
    {"id": 7, "location": "resolvability para 2", "paper_text": "minimum detectable difference is 2.98 points", "paper_value": "2.98", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "mdd_with_15_categories_paired=2.9783", "status": "exact_match", "details": "Rounds to 2.98."},
    {"id": 8, "location": "resolvability para 2", "paper_text": "2.98 points --- larger than the entire 3.98-point spread of the fourteen backbones", "paper_value": "2.98 > 3.98", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "backbone_spread_pp=3.9795; mdd=2.9783", "status": "number_mismatch", "details": "Arithmetic/comparison error: 2.98 is SMALLER than 3.98, not larger. Both values exist in file; 'larger than' is false."},
    {"id": 9, "location": "resolvability para 2", "paper_text": "six times the margin the field argues about", "paper_value": "2.98/0.5 = 6", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "2.9783/0.5=5.96", "status": "rounding_ok", "details": "5.96 rounds to 'six times'."},
    {"id": 10, "location": "resolvability para 2", "paper_text": "Adding two more training seeds per cell ... moves this only to 2.95 points", "paper_value": "2.95", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "mdd_with_15cat_x3seed_paired=2.95257", "status": "exact_match", "details": "Rounds to 2.95; 3 seeds/cell consistent with df=180."},
    {"id": 11, "location": "resolvability para 2", "paper_text": "would take 532 categories; the benchmark has fifteen", "paper_value": "532 / 15", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "categories_needed_paired=532.21; anova b=15", "status": "exact_match", "details": "Rounds to 532; 15 categories confirmed."},
    {"id": 12, "location": "resolvability para 3", "paper_text": "PatchCore-style ... seed standard deviation of 0.51 points and a requirement of only 17 runs ... factor of 14 below", "paper_value": "0.51 / 17 / 14x", "evidence_file": "(none provided)", "evidence_value": "absent", "status": "missing_evidence", "details": "No evidence file contains PatchCore harness numbers. 240/17=14.1 consistent, but inputs unverifiable."},
    {"id": 13, "location": "resolvability para 4", "paper_text": "statistically detectable at 210 detectors (p = 0.039)", "paper_value": "p=0.039", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "F_rows=1.8440, a=14, df_res=180", "status": "ambiguous_mapping", "details": "p not stored; F(13,180)=1.844 implies p~0.04, consistent with 0.039 but not directly verifiable."},
    {"id": 14, "location": "resolvability intro", "paper_text": "PatchCore reached 99.1% ... strongest methods reach 99.6% ... roughly a hundred images per category", "paper_value": "99.1 / 99.6 / ~100", "evidence_file": "(none provided)", "evidence_value": "absent", "status": "missing_evidence", "details": "External literature citations; expected absent from these files."},
    {"id": 15, "location": "search_vs_random para 1", "paper_text": "100 trials ... twelve independent seeds", "paper_value": "100 / 12", "evidence_file": "search_repeats/search_repeats.json", "evidence_value": "trials=100 all runs; seeds 5000-5011 (n_runs=12)", "status": "exact_match", "details": "Note run09 recorded only 90/100 trials; selection claims unaffected."},
    {"id": 16, "location": "search_vs_random para 1", "paper_text": "three quick training epochs per candidate, uniform sampling from the 42 cells", "paper_value": "3 epochs / 42 cells", "evidence_file": "(none provided)", "evidence_value": "absent", "status": "missing_evidence", "details": "Config constants not recorded; 42=14x3 consistent with space but sampling scheme unverified."},
    {"id": 17, "location": "search_vs_random para 1", "paper_text": "the searched feature dimension cannot reach the detector", "paper_value": "true", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "inert_axis.note: feature_dim cannot reach deployed detector (256-d bottleneck hard-coded)", "status": "exact_match", "details": "Directly supported."},
    {"id": 18, "location": "search_vs_random para 1", "paper_text": "eight of the twelve returned efficientnet_b1 ... four returned mobilenetv3_small_100", "paper_value": "8 / 4", "evidence_file": "search_repeats/search_repeats_summary.json", "evidence_value": "backbone_counts: efficientnet_b1=8, mobilenetv3_small_100=4", "status": "exact_match", "details": "Matches."},
    {"id": 19, "location": "search_vs_random para 1", "paper_text": "efficientnet_b1 (true rank 12 of 14, 80.43% mean AUROC)", "paper_value": "rank 12 / 80.43%", "evidence_file": "search_repeats/search_vs_random_final.json + summary", "evidence_value": "true_rank_of_selected=12; truth 0.804324=80.43%", "status": "exact_match", "details": "Both components match."},
    {"id": 20, "location": "search_vs_random para 1", "paper_text": "mobilenetv3_small_100 (rank 7 of 14, 81.85%)", "paper_value": "rank 7 / 81.85%", "evidence_file": "search_vs_random_final.json + search_repeats_summary.json", "evidence_value": "rank=7 ok; truth_per_backbone=0.819547=81.95%", "status": "number_mismatch", "details": "Rank matches, but only per-backbone truth given says 81.95%, not 81.85%; final_truth_analysis per-backbone values not provided."},
    {"id": 21, "location": "search_vs_random para 1", "paper_text": "only 2 of the 14 backbones and 6 of the 42 configurations appeared at all", "paper_value": "2 / 6", "evidence_file": "search_repeats/search_repeats_summary.json", "evidence_value": "unique_backbones_selected=2; unique_configs_selected=6", "status": "exact_match", "details": "Matches."},
    {"id": 22, "location": "search_vs_random para 1", "paper_text": "probability ... 7.2x10^-11", "paper_value": "7.2e-11", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "p_all_in_collapse_subset=7.2248e-11", "status": "exact_match", "details": "(2/14)^12 = 7.22e-11; verified."},
    {"id": 23, "location": "search_vs_random para 2", "paper_text": "search's output attains 80.91% mean AUROC", "paper_value": "80.91%", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "search_mean_true_auroc=0.8090584", "status": "exact_match", "details": "Rounds to 80.91; matches final file (summary file gives 80.94)."},
    {"id": 24, "location": "search_vs_random para 2", "paper_text": "random gives 81.65% (bootstrap s.d. 0.32 points)", "paper_value": "81.65% / 0.32", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "random_mean=0.8165089; bootstrap_sd=0.0032187", "status": "exact_match", "details": "0.81651 -> 81.65%; 0.003219 -> 0.32 pp."},
    {"id": 25, "location": "search_vs_random para 2", "paper_text": "best backbone in the space gives 83.34%", "paper_value": "83.34%", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "oracle_true_auroc=0.8334095 (resnet18)", "status": "exact_match", "details": "Matches final file; summary file's 83.81% is the other truth source."},
    {"id": 26, "location": "search_vs_random para 2 + caption", "paper_text": "sits at the 1.2th percentile ... worse than 98.8% of uniform draws", "paper_value": "1.2 / 98.8%", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "percentile_of_search_in_random=1.2285", "status": "exact_match", "details": "1.2285 -> 1.2; 100-1.23=98.77 -> 98.8."},
    {"id": 27, "location": "search_vs_random para 2", "paper_text": "mean true rank of 10.3 of 14 against 7.5 for random", "paper_value": "10.3 / 7.5", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "mean_true_rank_of_selected=10.333; random_expected_rank=7.5", "status": "exact_match", "details": "Matches."},
    {"id": 28, "location": "search_vs_random para 2 + caption", "paper_text": "never once selected one of the four strongest backbones", "paper_value": "0 runs", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "runs_selecting_top4=0", "status": "exact_match", "details": "Selected ranks 12 and 7 only."},
    {"id": 29, "location": "search_vs_random para 2", "paper_text": "constant resnet18 would have improved it by 2.4 points", "paper_value": "2.4", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "0.83341-0.80906=0.02435=2.43 pp", "status": "rounding_ok", "details": "2.43 rounds to 2.4."},
    {"id": 30, "location": "search_vs_random para 3", "paper_text": "well-behaved configuration is bounded below by roughly 7.5, while the two collapse-prone backbones return exactly 0.0", "paper_value": "floor ~7.5; zeros exclusive to 2 backbones", "evidence_file": "search_repeats/search_repeats.json", "evidence_value": "efficientnet_b0 scored 0.0 (run08 t63, run10 t32); min non-zero trial 4.56", "status": "scope_overclaim", "details": "A third backbone (efficientnet_b0) returned exact 0.0 twice; 'roughly 7.5' floor loose (non-zero minimum 4.56)."},
    {"id": 31, "location": "why-earlier-searches box", "paper_text": "The four search runs whose results the manuscript reports selected efficientnet_b0", "paper_value": "4 runs -> efficientnet_b0", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "stage2, stage2_rtx3090, stage2_rtx3090_100trials, stage2_topjournal all best=efficientnet_b0", "status": "ambiguous_mapping", "details": "4 of 5 stage2-type runs selected b0; mvtec_stage2_test/bottle selected b1; which 4 are 'reported' unverifiable."},
    {"id": 32, "location": "why-earlier-searches box", "paper_text": "efficientnet_b0 ... which does not collapse", "paper_value": "b0 never collapses", "evidence_file": "search_repeats/search_repeats.json", "evidence_value": "efficientnet_b0 score 0.0 in run08 t63 and run10 t32", "status": "scope_overclaim", "details": "Proxy scores of 0.0 observed for b0, contradicting exclusivity of collapse to b1/mnv3 (rare: 2/~1195 trials)."},
    {"id": 33, "location": "search_vs_random para 1", "paper_text": "a search space that is two thirds inert", "paper_value": "2/3 inert", "evidence_file": "search_forensics/search_forensics.json", "evidence_value": "inert_axis note: feature_dim cannot reach detector", "status": "rounding_ok", "details": "28/42=2/3 implied by 3 dims collapsing to backbone-only; consistent."},
    {"id": 34, "location": "resolvability para 4", "paper_text": "reporting differences smaller than about three points as unresolved", "paper_value": "~3 points", "evidence_file": "proxy_diag/samplesize_analysis.json", "evidence_value": "mdd_paired_15cat_pp=2.978", "status": "rounding_ok", "details": "'About three' consistent with 2.98."}
  ],
  "segment_verdict": "FAIL",
  "notes": "Driver of FAIL: claim 8 asserts 2.98 > 3.98 ('larger than the entire spread'), which is arithmetically false per the files (both values correct individually). Secondary: mobilenetv3 81.85% conflicts with the only per-backbone truth present (81.95%); PatchCore figures (0.51 pp, 17 runs) unsupported. run09 recorded 90/100 trials though paper says 100 per run. Paper consistently uses search_vs_random_final.json values (80.91/81.65/83.34/0.32/1.2) where summary file differs slightly."
}
```
