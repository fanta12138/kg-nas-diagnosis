# S1 abstract / introduction / related work

## PAPER TEXT (sent)

## manuscript.tex (title, abstract, keywords, declarations)
% ============================================================================
% main_v2.tex -- parallel build of the rewritten manuscript.
%
% Purpose: compile the new narrative (sections/) side by side with the
% existing main_springer.tex, so page count and cross-references can be checked
% before switching over. The original main_springer.tex and sections/ are left
% untouched as the fallback.
%
% Same class, same preamble, same bibliography. Only the \input paths differ.
% ============================================================================

\documentclass[pdflatex,iicol,sn-mathphys-num]{sn-jnl}

%%%% Standard Packages
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,calc}

%%%% Fix: sn-jnl v3.1 (Dec 2024) + TeX Live 2026 xcolor/hyperref hook conflict.
\usepackage{hyperref}

%%%% Custom commands (shared with the Elsevier build)
\input{math_commands}

%%%% Restore the standard `table` environment (see main_springer.tex for why).
\let\table\tableorg
\let\endtable\endtableorg

% Float placement
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.75}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.80}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{8}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}

\raggedbottom
\emergencystretch=2em

\begin{document}

\title[Diagnosing the Architecture Axis]{The Architecture Axis Is Not the Axis:
A Diagnostic Study of Neural Architecture Search and Score Fusion in
Unsupervised Industrial Anomaly Detection}

%%=============================================================%% Author block
\author[1]{\fnm{Hong} \sur{Li}}\email{lihong@gxust.edu.cn}

\author[1]{\fnm{Yuanwu} \sur{Zhou}}\email{zhouyuanwu@gxust.edu.cn}

\author*[1]{\fnm{Peng} \sur{Deng}}\email{dengpeng@gxust.edu.cn}

\affil*[1]{\orgdiv{School of Civil and Architectural Engineering}, \orgname{Guangxi University of Science and Technology}, \orgaddress{\street{No. 2 Wenchang Road}, \city{Liuzhou}, \postcode{545006}, \state{Guangxi}, \country{China}}}

%%==================================%% Abstract
\abstract{Unsupervised anomaly detection models normality so that inspection can
run on conforming parts alone. Two decisions determine whether such a system
works: the architecture, and how per-pixel discrepancies are reduced to an
image-level score --- the axis the literature documents least. We measure both, and
what the benchmark resolves. On MVTec-AD the margin contested between
successive state-of-the-art generations is $0.5$ AUROC points, while the standard
single-category comparison resolves $2.98$ --- larger than the whole spread of the
architectures it ranks. Instrumenting a reverse-distillation pipeline, two thirds
of its space proves deployment-inert: the searched feature dimension never reaches
the detector. The distillation signal that ranks candidates separates the twelve
stable backbones by $5.7\%$ of its median and returns exactly zero for two,
which the minimising rule prefers. Training 210 detectors attributes $0.54\%$ of
the accuracy variance to the architecture and $95.32\%$ to the category; twelve
searches all returned a collapse-prone backbone, worse than random. The leverage is
in the reduction step: top-$k$ pooling raises mean AUROC from $82.36\%$ to
$90.28\%$ without retraining, and can be selected without labels, recovering
$61\%$ of the oracle gap. The pipeline reaches $82.47\%$ against published values
of $95.6$--$99.6\%$, a gap we report rather than hide.}

\keywords{Industrial visual inspection, Unsupervised anomaly detection, Neural
architecture search, Knowledge distillation, Score fusion, Proxy reliability}

\maketitle

%%=============================================================================%% Body
\input{sections/1_introduction}
\input{sections/2_related_work}
\input{sections/3_method}
\input{sections/4_diagnosis}
\input{sections/5_leverage}
\input{sections/6_positioning}
\input{sections/7_conclusion}

%%=============================================================================%% Back matter
\backmatter

\bmhead{Acknowledgements}

This work was supported by the Guangxi Natural Science Foundation (Grant No. 2025GXNSFHA069076) and the Guangxi University of Science and Technology Doctoral Fund (Grant Nos. 26Z15, 26Z25).

\section*{Statements and Declarations}

\begin{itemize}
\item \textbf{Funding}: This research was supported by the Guangxi Natural Science Foundation under Grant No. 2025GXNSFHA069076 (awarded to P.D.) and by the Guangxi University of Science and Technology Doctoral Fund under Grant Nos. 26Z15 (awarded to P.D.) and 26Z25 (awarded to H.L.). The funders had no role in the design of the study, in the collection, analysis or interpretation of data, or in the decision to publish.

\item \textbf{Competing interests}: The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\item \textbf{Ethics approval and consent to participate; consent for publication; materials availability}: Not applicable. This study did not involve human participants or animals, and no new materials were generated.

\item \textbf{Data availability}: This work uses only the publicly available MVTec-AD benchmark~\cite{bergmann2021mvtec}; no new image data were generated. Every derived artefact is released together with the code (see below): the architecture-search logs, the per-configuration proxy evaluations, the 210-detector AUROC matrix, the per-run search records, the auxiliary benchmark harness, the synthetic-mechanism study, and the scripts that regenerate every table and figure. Each number reported in this paper can therefore be traced to the run that produced it.

\item \textbf{Code availability}: The complete source code and the raw experimental logs are publicly released at \url{https://github.com/fanta12138/kg-nas-diagnosis}, under a MIT licence. They comprise the pipeline under study, the diagnostic experiments (proxy repeatability, the truth matrix, the repeated searches), the synthetic reproduction of the degeneracy mechanism, the resolvability analysis, the benchmark harness, and the analysis scripts that produce every table and figure in this paper, together with a SHA-256 manifest of every released file.

\item \textbf{ORCID}: P.D. \url{https://orcid.org/0000-0002-2892-4251}. The remaining authors will supply their identifiers to the editorial office at submission.

\item \textbf{Author contributions}: All three authors are colleagues in the same school and jointly planned the study. Using the CRediT taxonomy:
\textbf{Conceptualization}, P.D.; \textbf{methodology}, P.D., H.L. and Y.Z.; \textbf{validation}, P.D.; \textbf{formal analysis}, P.D.; \textbf{writing---original draft preparation}, P.D.; \textbf{writing---review and editing}, H.L. and Y.Z.; \textbf{supervision}, P.D.; \textbf{project administration}, P.D.; \textbf{funding acquisition}, H.L. and P.D. All authors have read and agreed to the published version of the manuscript. P.D. is the corresponding author.

\item \textbf{Use of large language models}: An AI assistant (a large language model) was used substantially in this work. It wrote the code for the diagnostic experiments, the synthetic reproduction and the benchmark harness, executed those experiments, and produced the numerical results reported here; it also produced drafts of the manuscript text. The authors specified the research questions, directed the work, verified the reported results against the recorded experimental outputs, and take full responsibility for the content, the claims and their accuracy.
\end{itemize}

%%=============================================================================%% References
\bibliography{references}

\end{document}


## 1_introduction.tex
% ============================================================================
% DRAFT v2 -- Section 1: Introduction
% Genre: benchmark / evaluation study (not a new-detector paper).
% Genre precedents cited: Bouthillier (variance in ML benchmarks), IM-IAD
% (AD evaluation protocol), Yu et al. 2020 (NAS vs random), Cao et al.
% (streaming AD benchmark), Yu et al. 2026 (event-level evaluation).
%
% Numbers trace to proxy_diag/samplesize_analysis.json, phaseB_truth.json and
% docs/03_实验记录.md. Checked with adbench/numsnap.py after each edit.
% ============================================================================

\section{Introduction}
\label{sec:introduction}

Visual inspection on a production line has to work without defect samples.
Collecting and annotating the failures of a manufacturing process is expensive,
and the failure modes that matter are by definition the ones not yet seen, so the
established approach is to model normality and flag deviations. Reverse
distillation~\cite{deng2022reverse} is a representative and widely deployed member
of this family: a frozen teacher extracts features at several scales, a student is
trained to reproduce them on normal images, and at test time the discrepancy
between the two is taken as the anomaly signal.

Two engineering decisions have to be made before such a system can be deployed.
The first is the \emph{architecture}: what network the student should be. The
second is the \emph{scoring policy}: how the per-pixel discrepancy maps are
combined into the single number that decides whether a part is rejected. The
second decision is usually made implicitly --- the published pipeline averages the
discrepancy map over all pixels, a choice that is rarely stated and, to our
knowledge, never justified. The literature has overwhelmingly invested in the
first, applying neural architecture search (NAS) to anomaly
detection~\cite{liu2023simplenet,wang2021stpm} on the argument that the optimal
architecture is dataset-dependent and should be discovered rather than chosen.

\runinhead{The margins the field argues about, and the resolution of its
benchmark.} Progress on the standard benchmark is reported in tenths of a
percentage point. PatchCore reached $99.1\%$ image-level AUROC on MVTec-AD in
2022~\cite{roth2022patchcore}; the strongest methods reported since reach
$99.6\%$~\cite{liu2023simplenet,guo2025dinomaly}. The contested margin is
therefore $0.5$ points, on a benchmark whose test sets hold roughly a hundred
images per category. Before asking which design decisions produce such margins,
it is worth asking whether the standard evaluation design can resolve them at all.
We measure that first, because the answer determines how every subsequent
comparison in this literature should be read.

This paper is accordingly a \emph{benchmark-and-evaluation study}, not a new
detector. Work of this kind measures the evaluation practice of a field rather
than adding a method, and it has clear precedents in this area. Yu et
al.~\cite{yu2020evaluating} compared architectures returned by several NAS
algorithms against architectures drawn uniformly at random from the same search
space, and found the search phase uninformative. IM-IAD~\cite{xie2024imiad}
consolidated competing evaluation protocols for industrial anomaly detection onto
one footing. Bouthillier et al.~\cite{bouthillier2021accounting} showed that
unacknowledged run-to-run variance invalidates comparisons in machine-learning
benchmarks and derived how many runs such comparisons require; the same
re-examination has since been carried out for
streaming~\cite{cao2024revisiting} and industrial time-series~\cite{yu2026eventlevel}
anomaly detection. We continue that line into a deployed unsupervised
image-anomaly-detection pipeline: we instrument it, we re-run its search, and we
train the complete comparison it never ran.

\runinhead{What we instrument.}
Our own earlier pipeline, KG-NAS, is a three-stage system of this family: a search
ranks 42 configurations (14 backbones $\times$ 3 feature dimensions) on normal
data alone using a knowledge-distillation signal, and the winning configuration is
then trained as a reverse-distillation detector. We ask a question this literature
has largely assumed rather than measured: \emph{how much accuracy does the
architecture choice actually buy, is the search that selects it doing its job, and
what can the benchmark that judges both actually resolve?} We answer by
measurement: we instrument the search, we train all 14 backbones of its search
space on all 15 MVTec-AD categories under one protocol --- 210 detectors --- and
we sweep the scoring policy on frozen models.

\runinhead{The benchmark resolves $3$ points; the field argues about $0.5$.}
Two-way decomposition of the 210-detector matrix leaves a residual standard
deviation of $3.30$ points, and the run-to-run standard deviation of a single
detector at a fixed (backbone, category) is $1.96$ points over three seeds.
Comparing two backbones across the fifteen categories is therefore a paired test
whose minimum detectable difference is \textbf{$2.98$ points} --- larger than the
entire $3.98$-point spread of the fourteen backbones, and six times the margin
that separates the 2022 and 2026 state of the art. To resolve $0.5$ points on a
single category would take $240$ independent training runs per backbone. This is
the paper's measurement about the field as a whole rather than about this
pipeline.

\runinhead{Two thirds of the search space cannot reach the deployed model.}
The searched feature dimension enters the search through the student projector, so
the search signal depends on it, but it does not reach the detector: in the
deployment path the student encoder builds its bottleneck with a hard-coded width
while the decoder is constructed with the requested one, and the final training
entry point never passes the argument at all. The three values of the searched
dimension therefore produce the identical deployed model for a given backbone. The
42-configuration space contains 14 distinct models; the other 28 cells are
deployment-equivalent duplicates. The recorded searches behave as this predicts:
across four independent runs the selected backbone is the same every time, while
the selected dimension differs (Section~\ref{subsec:inert}).

\runinhead{The search signal does not distinguish architectures, and collapses
for some of them.} Calling the search's own evaluation method ten times per
configuration gives two results. Across the twelve well-behaved backbones the
median score spans only $5.7\%$ --- the same order as the within-configuration
noise of a single backbone --- so the signal carries essentially no
architecture-level information. And two backbones fail in a specific way:
\texttt{efficientnet\_b1} returns exactly $0.0$ in $73\%$ of evaluations and
\texttt{mobilenetv3\_small\_100} in $13\%$, because the student diverges, every
validation batch trips a hard threshold, and the score is then silently computed
as an average over zero samples. Since the search minimises, $0.0$ is the best
score attainable anywhere in the space
(Section~\ref{subsec:proxy_stability}); over the project's own record of 834
search trials the collapse occurs in $8.0\%$ of them.

\runinhead{Architecture is detectable, negligible, and does not transfer.}
Training 210 detectors under an identical protocol and decomposing the resulting
AUROC matrix attributes $0.54\%$ of the variance to the backbone and $95.32\%$ to
the category, with the backbone effect statistically significant ($p = 0.039$) and
practically irrelevant. The manuscript's selected architecture ranks fifth of
fourteen and is statistically indistinguishable from twelve of the thirteen
alternatives. More consequentially, the ranking does not transfer: the mean
correlation between two categories' orderings of the 14 backbones is
$\rho = 0.215$, and in $31\%$ of category pairs the categories disagree. Yet the
protocol searches on a single category and applies the winner to all fifteen
(Section~\ref{subsec:capacity_null}).

\runinhead{The consequence is measurable, and it is worse than chance.}
Running the search itself with independent seeds, the two backbones that collapse
to zero are selected every time, and the architecture the search returns is worse
than picking a backbone uniformly at random. A design axis that cannot reach the
model, ranked by a signal that does not discriminate, on a quantity that does not
transfer, is selected --- reliably --- by a failure mode
(Section~\ref{subsec:search_vs_random}).

\runinhead{The leverage is in the decision nobody documents.} On frozen models,
changing only the reduction that turns discrepancy maps into an image score moves
mean AUROC from $82.36\%$ to $90.28\%$ --- $+7.92$ points, reproduced across three
training seeds --- and choosing the policy per category reaches $93.72\%$. The
policy can further be chosen without any anomaly label, by perturbing held-out
normal images with synthetic defects and selecting the policy that best separates
perturbed from clean: across three training seeds this recovers $7.09$ of the
$11.65$ points available from an oracle, on the same trained models
(Sections~\ref{subsec:policy_results} and~\ref{subsec:pseudo_results}).

\subsection*{Contributions}

\begin{enumerate}
  \item \textbf{The resolvability of the standard benchmark design.} We anchor the
  noise floor of this task to the margin the field actually contests, and show
  that a single-category, single-run comparison resolves $2.98$ points against a
  contested margin of $0.5$ --- so most published backbone comparisons on this
  benchmark are not measurements. This is a measurement about the evaluation
  practice of the field, and it is our primary contribution.
  \item \textbf{A diagnosis of the architecture axis.} Two thirds of a standard
  search space is deployment-inert; its search signal does not discriminate
  between architectures and degenerates to a score of exactly zero for a subset of
  them; the axis contributes $0.54\%$ of the variance in detection accuracy
  against $95.32\%$ for the category.
  \item \textbf{A controlled comparison of a full search space.} 14 backbones
  $\times$ 15 categories under one protocol (210 detectors), which makes the
  negative result measurable rather than anecdotal.
  \item \textbf{Evidence that a single-category search cannot work.} The backbone
  ranking does not transfer across categories ($\bar\rho = 0.215$; $31\%$ of
  category pairs reverse), so a search run on one category estimates a quantity
  that does not exist independently of the category.
  \item \textbf{A label-free method for the axis that does matter.} Selecting the
  scoring policy by separation of synthetic pseudo-anomalies recovers $61\%$ of the
  oracle gap --- $7.09$ of $11.65$ points, averaged over three training seeds ---
  without any anomaly label and without retraining.
\end{enumerate}

\subsection*{Scope and honesty}

We do not claim state of the art: the pipeline reaches $82.47\%$ mean AUROC on
MVTec-AD, against published values of $95.6\%$--$99.6\%$ for the established
methods, and Section~\ref{sec:positioning} reports that gap rather than obscuring
it. The contribution is not a better number but a quantitative account of which
design choices produce the number, of how much of the number the benchmark can
measure at all, and of the finding that the axis the field searches is not the
axis that matters. Section~\ref{sec:related_work} positions the work,
Section~\ref{sec:method} describes the pipeline under study,
Section~\ref{sec:diagnosis} reports the diagnosis and the resolvability analysis,
Section~\ref{sec:leverage} the scoring-policy analysis,
Section~\ref{sec:positioning} states where the pipeline stands, and
Section~\ref{sec:conclusion} concludes.


## 2_related_work.tex
% ============================================================================
% DRAFT v2 -- Section 2: Related work
% Genre: benchmark / evaluation study. Subsection 2.3 places the paper in that
% genre explicitly. All \cite keys verified to exist in references.bib.
% ============================================================================

\section{Related Work}
\label{sec:related_work}

\subsection{Unsupervised Anomaly Detection}
\label{subsec:related_ad}

Unsupervised anomaly detection models normality and flags deviations, because
defect samples are scarce and the failures that matter are the ones not yet seen.
Four families dominate. Reconstruction methods learn to reproduce normal images
and score by reconstruction error, from autoencoders scored with a structural
similarity measure~\cite{bergmann2018improving} to generative
adversarial~\cite{akcay2019ganomaly,schlegl2019f} and memory-augmented
formulations~\cite{gong2019memorizing}. Feature-embedding methods characterise
the distribution of pre-trained deep features, either parametrically as in
PaDiM~\cite{defard2021padim}, by nearest-neighbour matching as in
SPADE~\cite{cohen2020spade} and PatchCore~\cite{roth2022patchcore}, or by density
estimation with normalizing flows~\cite{yu2021fastflow}. Teacher--student methods
distill a frozen teacher into a student that reproduces it on normal data, using
multi-scale feature matching~\cite{wang2021stpm,bergmann2020uninformed} or, in
reverse distillation, a decoder that reconstructs the teacher's features from a
bottleneck embedding~\cite{deng2022reverse}; the present paper studies a pipeline
of this last kind. Discriminatively trained methods inject synthetic
defects~\cite{zavrtanik2021draem} and have produced the strongest recent
results~\cite{liu2023simplenet,batzner2024efficientad,guo2025dinomaly,lee2022cfa};
surveys of the area are available~\cite{liu2024deep}.

Two properties of this literature matter here. The architecture is treated as a
given --- either the backbone of a pre-trained feature
extractor~\cite{defard2021padim,roth2022patchcore} or a fixed encoder--decoder
pair~\cite{deng2022reverse} --- and the step that turns a per-pixel anomaly map
into the single number on which a part is accepted or rejected is usually left
implicit. The pipeline we study reduces the multi-scale discrepancy map by an
unweighted mean over all pixels; Section~\ref{sec:leverage} shows that this one
undocumented choice is worth several times more accuracy than the architecture,
and that it can be revisited without retraining.

\subsection{Neural Architecture Search and the Evaluation of Its Search Phase}
\label{subsec:related_nas}

Neural architecture search automates the choice of network topology, from
early evolutionary~\cite{xie2017genetic} and reinforcement-learning
formulations~\cite{zoph2018learning,liu2018progressive} through differentiable
relaxation~\cite{liu2019darts}, platform-aware multi-objective
search~\cite{tan2019mnasnet} and regularised
evolution~\cite{real2019regularized}; see~\cite{ren2021comprehensive} for a survey.
Two lines of work are directly relevant.

The first applies NAS to anomaly detection. For image anomaly localisation,
Qian et al.~\cite{qian2024knowledgeguided} generate anomalous samples with a
knowledge-guided generator and use them as a reinforcement-learning reward, so
that architectures can be compared on a validation set that would otherwise be
unavailable. Work in this line takes for granted that a better architecture will
produce a better detector, and measures success by the downstream result alone.

The second line asks whether the search phase itself deserves that trust, and it
is the immediate predecessor of this paper. Yu et al.~\cite{yu2020evaluating}
compared the architectures returned by several NAS algorithms against
architectures sampled uniformly at random from the same search space, and found
that, on average, the NAS algorithms perform \emph{similarly to the random
policy}; they attribute this to weight sharing degrading the ranking of
candidates so that it no longer reflects true performance. Our findings agree in
direction and go further in three respects. First, from ``similar to random'' to
``worse than random'': in Section~\ref{subsec:search_vs_random} twelve independent
runs of the search all return one of the two configurations prone to a degenerate
evaluation, and the mean accuracy of the architectures they return sits at the
$1.2$th percentile of the distribution obtained by uniform sampling ($80.91\%$
against $81.65\%$, with the best backbone at $83.34\%$). Second, a mechanism
rather than a symptom: the cause is not weight sharing but an interaction between
the objective and the selection rule, as the evaluation can silently return
exactly $0.0$ when every validation batch is rejected, and since the search
minimises, that value is the optimum of the search's own range
(Section~\ref{subsec:proxy_stability}). Third, structure rather than signal
alone: two thirds of the search space cannot reach the deployed model at all
(Section~\ref{subsec:inert}), and the quantity being ranked contributes $0.54\%$
of the variance in accuracy against $95.32\%$ for the category
(Section~\ref{subsec:capacity_null}). Where Yu et al.\ study image classification
and language modelling with weight-sharing supernets, we study single-class
unsupervised anomaly detection with an independent quick-training proxy, and we
additionally show that the backbone ordering does not transfer between categories
($\bar\rho = 0.215$; $31\%$ of category pairs reverse), which makes a
single-category search unsound regardless of the size of the architecture effect.

\subsection{Benchmark and Evaluation Studies}
\label{subsec:related_benchmark}

The present paper belongs to a class of work whose contribution is a measurement
about how a field evaluates itself, rather than a new method. The class is
established, and we adopt three of its methods.

Bouthillier et al.~\cite{bouthillier2021accounting} showed that machine-learning
benchmarks routinely compare algorithms on a single run, that the unacknowledged
run-to-run variance is then large enough to reverse conclusions, and that the
number of runs a comparison requires can be derived from the variance. Their
analysis is the methodological template for
Section~\ref{subsec:resolvability}: we estimate the residual variance of the task,
fold in the externally reported effect size, and report the smallest difference
the standard design can resolve. IM-IAD~\cite{xie2024imiad} performed the
corresponding consolidation for industrial image anomaly detection, reproducing
published methods under one protocol and showing how much of the reported spread
survives standardisation. For anomaly detection outside the image domain the same
re-examination has been carried out on streaming
data~\cite{cao2024revisiting} and, most recently, on industrial multivariate time
series, where unified reproduction was combined with an event-level
evaluation~\cite{yu2026eventlevel}.

Our paper extends this line in two ways that we have not seen for unsupervised
image anomaly detection. It supplies the fully crossed configuration $\times$
category comparison that a variance decomposition requires --- 210 detectors,
every backbone of a search space on every category of the benchmark, one protocol
--- which is what turns a diagnosis of the architecture axis into a measurement
rather than an anecdote. And it anchors the resulting noise floor to the margin
the field actually contests, so that the resolvability of the standard benchmark
design can be stated in the same units as the claims it is used to support.
Neither IM-IAD nor the time-series reproductions reports that quantity; for this
task it turns out to be decisive.

\subsection{Knowledge Distillation}
\label{subsec:related_kd}

Knowledge distillation transfers the behaviour of a strong model to a smaller one
by matching outputs or intermediate representations~\cite{gou2021knowledge},
including similarity-preserving~\cite{tung2019similarity}, structured~\cite{liu2019structured}
and self-distilled~\cite{zhang2022self} variants and applications to
detection~\cite{wang2022distilling,yang2023attention}. In anomaly detection,
distillation is predominantly a \emph{detection} device: the teacher--student
discrepancy is the anomaly
signal~\cite{wang2021stpm,deng2022reverse,bergmann2020uninformed}. Using it as the
\emph{search metric} --- ranking candidates by how well a student distils from a
teacher, rather than by the accuracy of the detector it will eventually become ---
has been explored for NAS generally~\cite{kang2020towards} and, for anomaly
localisation, by Qian et al.~\cite{qian2024knowledgeguided}.

That reuse is the premise our results call into question.
Section~\ref{subsec:proxy_stability} evaluates the distillation proxy directly, by
repeating the search's own evaluation ten times per configuration, and finds that
it separates the twelve well-behaved backbones by only $5.7\%$ of its median ---
the same order as its own run-to-run noise --- while collapsing to an exact zero
for two of them. Its ordering is uncorrelated with the true ordering of the same
backbones: Spearman $\rho = -0.12$ over all fifteen categories and $\rho = 0.02$
on the very category the search is run on. The proxy is measuring something, but
not the thing it is used to select for.

\subsection{Positioning}
\label{subsec:related_summary}

This paper is a benchmark-and-evaluation study of a deployed design, not a new
detector, and we claim no priority for combining knowledge distillation with
architecture search. It differs from the NAS-for-anomaly-detection literature in
asking whether the search phase works rather than assuming it; it extends the
evaluation framework of Yu et al.~\cite{yu2020evaluating} to unsupervised anomaly
detection, quantifying the failure rather than merely detecting it; it adopts the
variance discipline of Bouthillier et al.~\cite{bouthillier2021accounting} and the
protocol consolidation of IM-IAD~\cite{xie2024imiad}, and reports the
resolvability of the benchmark that both leave unstated; and it contributes a
controlled comparison of a complete search space under one protocol
(Section~\ref{sec:positioning}) which makes the diagnosis measurable. On the
constructive side, it identifies the inference-time scoring policy as the axis
that actually moves accuracy and gives a label-free procedure for selecting it
(Section~\ref{subsec:pseudo_results}).


## EVIDENCE FILES SENT

- adbench/results/analysis_p1_p3.json
- search_repeats/search_vs_random_final.json
- search_repeats/search_repeats_summary.json

## REVIEWER RAW RESPONSE

Audit of S1: 31 quantitative claims extracted. Five match the search-repetition evidence exactly, two are ambiguous mappings, and the remainder trace to analyses (resolvability, variance decomposition, proxy repeatability, policy sweep) whose files were not supplied. No arithmetic errors found.

```json
{
  "claims": [
    {"id": 1, "location": "abstract", "paper_text": "the margin contested between successive state-of-the-art generations is 0.5 AUROC points", "paper_value": "0.5 points", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Literature SOTA values 99.1/99.6 not present in provided files."},
    {"id": 2, "location": "abstract; intro contributions", "paper_text": "the standard single-category comparison resolves 2.98", "paper_value": "2.98 points", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Resolvability/minimum-detectable-difference analysis file not provided."},
    {"id": 3, "location": "abstract; intro resolvability paragraph", "paper_text": "larger than the whole spread of the architectures it ranks; entire 3.98-point spread of the fourteen backbones", "paper_value": "3.98-point spread", "evidence_file": "search_repeats/search_vs_random_final.json + search_repeats_summary.json", "evidence_value": "best 0.83341 (final file); min 0.79361 (summary file only)", "status": "ambiguous_mapping", "details": "3.98 = 0.83341−0.79361 mixes two files; final matrix's own per-backbone values absent."},
    {"id": 4, "location": "abstract; intro inert paragraph", "paper_text": "two thirds of its space proves deployment-inert; 42-configuration space contains 14 distinct models; other 28 cells duplicates", "paper_value": "28/42 = 2/3", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Code-level claim; arithmetic 42−14=28, 28/42=2/3 internally consistent only."},
    {"id": 5, "location": "abstract; intro; related work 2.4", "paper_text": "separates the twelve stable backbones by 5.7% of its median", "paper_value": "5.7%", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Proxy repeatability statistics absent from provided files."},
    {"id": 6, "location": "abstract; intro; related work", "paper_text": "returns exactly zero for two; efficientnet_b1 0.0 in 73% of evaluations, mobilenetv3_small_100 in 13%", "paper_value": "73%; 13%", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Zero-collapse evaluation rates not in provided files."},
    {"id": 7, "location": "abstract; intro; related work 2.2", "paper_text": "attributes 0.54% of the variance to the backbone and 95.32% to the category (p = 0.039)", "paper_value": "0.54%; 95.32%; p=0.039", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Truth-matrix variance decomposition file not provided."},
    {"id": 8, "location": "abstract", "paper_text": "twelve searches all returned a collapse-prone backbone, worse than random", "paper_value": "12/12 collapse-prone; worse than random", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "n_runs 12; n_collapse_prone_selected 12; 0.8091<0.8165; p=0.0123", "status": "exact_match", "details": "Fully supported: 12 runs, all selecting collapse-prone backbones, mean below random."},
    {"id": 9, "location": "abstract; intro leverage paragraph", "paper_text": "top-k pooling raises mean AUROC from 82.36% to 90.28% without retraining", "paper_value": "82.36% → 90.28%", "evidence_file": "adbench/results/analysis_p1_p3.json", "evidence_value": "row means 85.13/89.58/89.37/88.47/83.69/47.82", "status": "missing_evidence", "details": "Neither value appears; provided locality-table means do not match either number."},
    {"id": 10, "location": "abstract; contributions", "paper_text": "recovering 61% of the oracle gap — 7.09 of 11.65 points", "paper_value": "61%; 7.09/11.65", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Oracle-gap values absent; 7.09/11.65=60.9%≈61% arithmetic internally consistent."},
    {"id": 11, "location": "abstract; intro scope", "paper_text": "pipeline reaches 82.47% against published values of 95.6–99.6%", "paper_value": "82.47%; 95.6–99.6%", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Headline AUROC and published-range endpoints not in provided files."},
    {"id": 12, "location": "intro", "paper_text": "PatchCore reached 99.1% image-level AUROC in 2022; strongest methods since reach 99.6%", "paper_value": "99.1%; 99.6%", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Literature citation values; not verifiable from given files."},
    {"id": 13, "location": "intro what-we-instrument", "paper_text": "search ranks 42 configurations (14 backbones × 3 feature dimensions)", "paper_value": "42 = 14×3", "evidence_file": "search_repeats/search_repeats_summary.json", "evidence_value": "14 backbones listed; dims {128,256,512}", "status": "ambiguous_mapping", "details": "14 backbones exact; three dims inferable from selected_dims; 42 itself not stated."},
    {"id": 14, "location": "intro", "paper_text": "train all 14 backbones on all 15 MVTec-AD categories — 210 detectors", "paper_value": "14×15 = 210", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "source: 'seed 42, 15 categories, 14 backbones'", "status": "exact_match", "details": "Design statement matches cited truth-matrix scope."},
    {"id": 15, "location": "intro", "paper_text": "residual standard deviation of 3.30 points", "paper_value": "3.30 points", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Residual SD absent from provided files."},
    {"id": 16, "location": "intro", "paper_text": "run-to-run standard deviation of a single detector is 1.96 points over three seeds", "paper_value": "1.96 points; 3 seeds", "evidence_file": "adbench/results/analysis_p1_p3.json", "evidence_value": "seed_sd I-AUROC 0.00355 (0.36 pts, different model)", "status": "missing_evidence", "details": "1.96 absent; file's seed_sd belongs to a different locality study."},
    {"id": 17, "location": "intro", "paper_text": "minimum detectable difference is 2.98 points", "paper_value": "2.98 points", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Same unprovided resolvability analysis as claim 2."},
    {"id": 18, "location": "intro", "paper_text": "six times the margin that separates the 2022 and 2026 state of the art", "paper_value": "6× (2.98/0.5 = 5.96)", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Ratio arithmetic consistent; depends on unverified 2.98 and 0.5."},
    {"id": 19, "location": "intro", "paper_text": "To resolve 0.5 points on a single category would take 240 independent training runs per backbone", "paper_value": "240 runs", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Sample-size calculation file not provided."},
    {"id": 20, "location": "intro inert paragraph", "paper_text": "across four independent runs the selected backbone is the same every time, while the selected dimension differs", "paper_value": "4 runs; constant backbone", "evidence_file": "search_repeats/search_repeats_summary.json", "evidence_value": "n_runs 12; two backbones (8/4); dims vary", "status": "missing_evidence", "details": "Four-run record absent; provided 12-run files show two distinct backbones, uncorroborating."},
    {"id": 21, "location": "intro; related work 2.4", "paper_text": "Calling the search's own evaluation method ten times per configuration", "paper_value": "10 evaluations/config", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Proxy protocol constant; supporting file not provided."},
    {"id": 22, "location": "intro", "paper_text": "over the project's own record of 834 search trials the collapse occurs in 8.0% of them", "paper_value": "834 trials; 8.0%", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Trial-log aggregate absent; summary file covers a different 12×100=1200 trials."},
    {"id": 23, "location": "intro capacity-null paragraph", "paper_text": "selected architecture ranks fifth of fourteen and is statistically indistinguishable from twelve of the thirteen alternatives", "paper_value": "5th of 14; 12 of 13", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Ranking and post-hoc comparison results absent from provided files."},
    {"id": 24, "location": "intro; contributions; related work", "paper_text": "mean correlation between two categories' orderings of the 14 backbones is ρ = 0.215; 31% of category pairs disagree", "paper_value": "ρ=0.215; 31%", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Transfer-correlation analysis file not provided."},
    {"id": 25, "location": "intro consequence paragraph", "paper_text": "the two backbones that collapse to zero are selected every time", "paper_value": "selected in 12/12 runs", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "only efficientnet_b1 / mobilenetv3_small_100; 12 runs", "status": "exact_match", "details": "Selection set and run count match exactly."},
    {"id": 26, "location": "intro consequence paragraph", "paper_text": "the architecture the search returns is worse than picking a backbone uniformly at random", "paper_value": "worse than random", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "0.80906 vs 0.81651; 1.23rd pct; p=0.0123", "status": "exact_match", "details": "Direction and significance both supported."},
    {"id": 27, "location": "intro leverage paragraph", "paper_text": "+7.92 points, reproduced across three training seeds", "paper_value": "+7.92; 3 seeds", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Delta arithmetic 90.28−82.36=7.92 correct; base values and seed count unverified."},
    {"id": 28, "location": "intro leverage paragraph", "paper_text": "choosing the policy per category reaches 93.72%", "paper_value": "93.72%", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Per-category oracle value absent from provided files."},
    {"id": 29, "location": "related work 2.2", "paper_text": "Yu et al. found the NAS algorithms perform similarly to the random policy", "paper_value": "qualitative literature claim", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "External-literature characterization; not auditable here."},
    {"id": 30, "location": "related work 2.2", "paper_text": "sits at the 1.2th percentile of the distribution obtained by uniform sampling (80.91% against 81.65%, with the best backbone at 83.34%)", "paper_value": "1.2; 80.91%; 81.65%; 83.34%", "evidence_file": "search_repeats/search_vs_random_final.json", "evidence_value": "1.2285; 0.8090584; 0.8165089; 0.8334095", "status": "exact_match", "details": "All four values match to displayed precision."},
    {"id": 31, "location": "related work 2.4", "paper_text": "Spearman ρ = −0.12 over all fifteen categories and ρ = 0.02 on the very category the search is run on", "paper_value": "−0.12; 0.02", "evidence_file": "none", "evidence_value": "—", "status": "missing_evidence", "details": "Proxy-vs-truth correlation file not provided."}
  ],
  "segment_verdict": "WARN",
  "notes": "Search-repetition claims (12 runs, collapse-prone selection, worse-than-random, percentile, means) verify exactly. Resolvability (2.98/3.30/1.96/240), variance shares (0.54/95.32/p=0.039), proxy stats (5.7%, 73%/13%, 834, ρ=−0.12/0.02), policy sweep (82.36→90.28, 93.72, 7.09/11.65) and headline 82.47% belong to analyses whose files are absent in this segment. The 3.98-point spread is only reconstructable by mixing values from two different truth matrices; the 'four runs, same backbone' claim is uncorroborated and in tension with the provided 12-run record (two distinct backbones). Grant/ORCID numbers are administrative and not audited."
}
```
