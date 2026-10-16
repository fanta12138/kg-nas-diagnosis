# S2 materials and methods

## PAPER TEXT (sent)

% ============================================================================
% DRAFT v2 -- Section 3: The pipeline under study
% Reports what the pipeline does, not what it is worth.
% ============================================================================

\section{Materials and Methods}
\label{sec:method}

This section describes the pipeline whose two design axes are examined in
Sections~\ref{sec:diagnosis} and~\ref{sec:leverage}. It is a three-stage
knowledge-distillation pipeline built on reverse
distillation~\cite{deng2022reverse}: a teacher is pre-trained on normal images,
an architecture is searched using a distillation signal, and the winning
architecture is trained as a reverse-distillation detector.
Figure~\ref{fig:pipeline} sketches the three stages and marks the two axes this
paper measures. We describe the pipeline precisely, because several of the
findings that follow are properties of specific implementation decisions rather
than of the method in the abstract. Stage~1 pre-trains a teacher encoder on normal
training images with a contrastive objective. Stage~2 searches an architecture for
the student, scoring each candidate by how well it distils from the frozen teacher
on normal data using a small number of training epochs. Stage~3 trains the
selected architecture --- student encoder plus decoder --- as a reverse-distillation
detector. The search runs on a single category and its result is applied to all
categories of the dataset.

\subsection{Stage 1: Teacher Pre-training}
\label{subsec:teacher}
\input{figures/fig1_pipeline}

The teacher is a ResNet-18~\cite{he2016deep} with a 128-dimensional projection
head, trained for 100 epochs on the normal training images of the search category
with a contrastive objective, and frozen thereafter. Its role is to define the
feature space that both the search signal and the final detector compare against.
At test time it contributes intermediate features at three resolutions: the
outputs of \texttt{layer1}, \texttt{layer2} and \texttt{layer3}, of shape
$64\times56\times56$, $128\times28\times28$ and $256\times14\times14$ for an input
resized to $224\times224$.

\subsection{Stage 2: Architecture Search}
\label{subsec:nas}

The search space is the Cartesian product of 14 backbones and three feature
dimensions, giving 42 configurations. The backbones span convolutional and
attention-based families from 2015 to 2023, including
ResNets~\cite{he2016deep}, EfficientNet variants, MobileNetV3 and MobileOne, ViT
and DeiT variants, Swin-Tiny, and the ConvNeXt and ConvNeXt~V2
families~\cite{woo2023convnextv2}; Section~\ref{sec:positioning} lists them with
their deployed sizes. The three feature dimensions are 128, 256 and 512.

\subsubsection*{The evaluation signal}
Each candidate is evaluated by building a student with the candidate's backbone
and projection dimension, training it for a small number of epochs on the normal
training images of the search category, and then measuring a composite
distillation loss on held-out normal images:
%
\begin{equation}
\label{eq:proxy}
s(a) \;=\; \frac{1}{|\mathcal{V}|}\sum_{x \in \mathcal{V}}
\Bigl[\, \mathcal{L}_{\mathrm{kd}}(x) \;+\; 0.3\,\mathcal{L}_{\mathrm{cons}}(x) \Bigr],
\end{equation}
%
where $\mathcal{L}_{\mathrm{kd}}$ is the mean-squared error between the student's
projection and the teacher's, and $\mathcal{L}_{\mathrm{cons}}$ is the
mean-squared error between the student's projections of two independently
augmented views of the same image. Lower is better. Every candidate is scored once;
there is no repetition and therefore no estimate of the signal's own variance.
Two details of this computation are examined in
Section~\ref{subsec:proxy_stability} and are stated here for completeness. A
validation batch contributes to the sum only if its loss is finite and below a
fixed threshold; otherwise it is skipped. And the average is formed as
$\text{total} / \max(\text{count}, 1)$, so if every batch of a trial is skipped the
signal returns exactly zero. Because the backbones produce features of different
dimensionality while the teacher projects into a 128-dimensional space, a
candidate whose projection dimension differs from 128 is additionally given a
learned linear layer that maps the student's output into the teacher's space
before the losses are computed; this alignment layer exists only during search.

The search draws candidates uniformly at random from the 42 cells and returns the
one with the smallest score,
%
\begin{equation}
\label{eq:argmin}
a^\star \;=\; \arg\min_{a \in \mathcal{A}_{\text{sampled}}} \; s(a),
\end{equation}
%
and the number of draws is a protocol parameter (the recorded runs used 100, of
which a variable number complete successfully).

\subsection{Stage 3: Reverse-Distillation Detection}
\label{subsec:rd_detection}

The selected configuration is re-instantiated as a detector. A student encoder
takes the backbone's final feature map, adaptively pools it to $14\times14$, and
projects it through a $1\times1$ convolution with batch normalisation and ReLU to
a bottleneck of width $B$. A decoder then reconstructs the teacher's features in
reverse order: a first block produces the $14\times14$ stage at 256 channels, the
output is upsampled and passed through a second block producing the $28\times28$
stage at 128 channels, and upsampled again through a third block producing the
$56\times56$ stage at 64 channels. We note for later use that the bottleneck width
is fixed at $B = 256$ in the deployment path, independently of the feature
dimension that the search may have selected;
Section~\ref{subsec:inert} shows that this makes the searched dimension
unobservable in the deployed detector. The detector is trained on the normal
training images of each category separately, with AdamW at a learning rate of
$10^{-3}$ and weight decay $10^{-4}$, a cosine schedule, and random horizontal
flipping with probability $0.5$; the loss is the sum of mean-squared errors
between the student's reconstructions and the teacher's features at the three
scales.

\subsubsection*{Anomaly scoring}
\label{subsubsec:scoring}
Given a test image $x$, the teacher produces features $t_1, t_2, t_3$ and the
student produces reconstructions $s_1, s_2, s_3$. For each scale $k$ we form a
discrepancy map
%
\begin{equation}
\label{eq:diffmap}
m_k(x) \;=\; \bigl\| s_k(x) - t_k(x) \bigr\|_2 ,
\end{equation}
%
taking the $L_2$ norm across channels, and upsample each map to the input
resolution. An image-level score is then obtained by combining the maps with
non-negative weights $\mathbf{w} = (w_1, w_2, w_3)$ on the probability simplex and
reducing the resulting map to a scalar with a pooling operator $\rho$:
%
\begin{equation}
\label{eq:policy}
\begin{aligned}
S(x) &\;=\; \rho\!\left( \textstyle\sum_{k} w_k \, m_k(x) \right), \\
\rho &\in \{\,\text{mean},\ \text{top-}k,\ \max\,\}.
\end{aligned}
\end{equation}
%
We refer to the pair $(\mathbf{w}, \rho)$ as the \emph{scoring policy}. The
published pipeline uses $\mathbf{w} = (\tfrac13,\tfrac13,\tfrac13)$ and
$\rho = \text{mean}$, that is, an unweighted average over all pixels of the sum of
the three maps. This choice is not forced by the method: given a trained detector
the maps $m_k$ are already computed, so any policy can be evaluated on the same
model. Section~\ref{sec:leverage} evaluates the space of policies in
Equation~\eqref{eq:policy} exhaustively, and shows that this reduction is worth
substantially more than the architecture that the search selects for.

\subsection{Protocol and Implementation Details}
\label{subsec:implementation}

Unless stated otherwise, the detector above is trained for 500 epochs with batch
size 64 and seed 42, with a ResNet-18 teacher, on the MVTec-AD
dataset~\cite{bergmann2021mvtec}. The search uses three quick training epochs per
candidate with batch size 16 on the \texttt{wood} category, and is repeated 100
times per run. Inputs are resized to $224\times224$ and normalised with ImageNet
statistics. All figures report image-level AUROC averaged over the 15 categories.
Where a backbone's memory footprint exceeds the device at batch size 64, training
uses gradient accumulation with a smaller micro-batch, keeping the effective batch
size at 64; the micro-batch used is recorded per run and this affects only the
largest backbones.

\subsection{Use of Generative AI in This Work}
\label{subsec:genai}

In accordance with the journal's disclosure requirement, we state the role of
generative AI explicitly rather than only in the back matter. An AI assistant (a
large language model) was used substantially. It wrote the code for the diagnostic
experiments, the synthetic reproduction and the benchmark harness; it executed
those experiments and produced the numerical results reported here; and it
produced drafts of the manuscript text, including the figures. It did not
determine the research questions, the experimental design or the interpretation of
the results. The authors specified the research questions, directed the work,
verified every reported result against the recorded experimental outputs, and take
full responsibility for the content, the claims and their accuracy. No image or
table in this paper was generated by AI in the sense the policy contemplates: the
figures are direct plots of recorded measurements, produced by the released script
\texttt{make\_figures\_v2.py}. All code, logs and derived artefacts are released
so that every number can be traced to the run that produced it
(Section~\ref{sec:conclusion}).


## EVIDENCE FILES SENT

- proxy_diag/backbone_params.json
- proxy_diag/probe_degenerate.json
- synthetic_mechanism/synthetic_mechanism.json

## REVIEWER RAW RESPONSE

```json
{
  "claims": [
    {
      "id": 1,
      "location": "sec:method, intro + subsec:teacher",
      "paper_text": "a teacher is pre-trained on normal images ... The teacher is a ResNet-18 with a 128-dimensional projection head",
      "paper_value": "teacher = ResNet-18, projection dim 128",
      "evidence_file": "proxy_diag/probe_degenerate.json",
      "evidence_value": "align_layer=false for all dim=128 rows (4/4); align_layer=true for dim=256 and dim=512 rows (8/8)",
      "status": "ambiguous_mapping",
      "details": "The 128-d teacher space is indirectly confirmed (no alignment layer needed exactly at dim 128). The teacher being ResNet-18 is not directly evidenced: resnet18 appears in backbone_params.json only as a student candidate (student_M=13.043); no teacher checkpoint/log provided."
    },
    {
      "id": 2,
      "location": "subsec:teacher",
      "paper_text": "trained for 100 epochs on the normal training images of the search category with a contrastive objective, and frozen thereafter",
      "paper_value": "100 epochs teacher pre-training",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "No teacher-training log in evidence set."
    },
    {
      "id": 3,
      "location": "subsec:teacher",
      "paper_text": "outputs of layer1, layer2 and layer3, of shape 64x56x56, 128x28x28 and 256x14x14 for an input resized to 224x224",
      "paper_value": "feature shapes 64x56x56 / 128x28x28 / 256x14x14",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "Shapes are standard ResNet-18 shapes but no raw feature-dump file is provided to trace them."
    },
    {
      "id": 4,
      "location": "subsec:nas",
      "paper_text": "The search space is the Cartesian product of 14 backbones and three feature dimensions, giving 42 configurations.",
      "paper_value": "14 backbones x 3 dims = 42",
      "evidence_file": "proxy_diag/backbone_params.json",
      "evidence_value": "exactly 14 backbone keys (convnextv2_tiny, convnextv2_base, mobileone_s0, convnext_tiny, convnext_small, vit_tiny_patch16_224, deit_tiny_patch16_224, swin_tiny_patch4_window7_224, efficientnet_b0, efficientnet_b1, mobilenetv3_small_100, resnet18, resnet34, resnet50); probe_degenerate.json shows exactly 3 dims {128,256,512}; 14x3=42",
      "status": "exact_match",
      "details": "Count, dim set, and arithmetic all verified."
    },
    {
      "id": 5,
      "location": "subsec:nas",
      "paper_text": "including ResNets, EfficientNet variants, MobileNetV3 and MobileOne, ViT and DeiT variants, Swin-Tiny, and the ConvNeXt and ConvNeXt V2 families",
      "paper_value": "listed backbone families",
      "evidence_file": "proxy_diag/backbone_params.json",
      "evidence_value": "all named families present: resnet18/34/50, efficientnet_b0/b1, mobilenetv3_small_100, mobileone_s0, vit_tiny, deit_tiny, swin_tiny, convnext_tiny/small, convnextv2_tiny/base",
      "status": "ambiguous_mapping",
      "details": "All named families match JSON keys. Minor scope looseness: 'ViT and DeiT variants' is plural but only one variant of each (tiny) exists in the space; 'EfficientNet variants' (2) and ConvNeXt families (2 each) are accurate."
    },
    {
      "id": 6,
      "location": "subsec:nas",
      "paper_text": "The backbones span convolutional and attention-based families from 2015 to 2023",
      "paper_value": "year range 2015-2023",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "Only the 2023 upper bound is indirectly supported by the woo2023convnextv2 citation key; no per-backbone release dates in evidence. Attention-based (ViT/DeiT/Swin) and convolutional families are both present among the 14 keys."
    },
    {
      "id": 7,
      "location": "subsec:nas",
      "paper_text": "The three feature dimensions are 128, 256 and 512.",
      "paper_value": "{128, 256, 512}",
      "evidence_file": "proxy_diag/probe_degenerate.json",
      "evidence_value": "rows contain dim=128, dim=256, dim=512 only",
      "status": "exact_match",
      "details": "Exactly these three dimensions appear in the probe evidence."
    },
    {
      "id": 8,
      "location": "subsec:nas, eq:proxy",
      "paper_text": "L_kd(x) + 0.3 L_cons(x)",
      "paper_value": "consistency-loss weight = 0.3",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "Loss composition and weight not recoverable from probe scores alone."
    },
    {
      "id": 9,
      "location": "subsec:nas",
      "paper_text": "Every candidate is scored once; there is no repetition and therefore no estimate of the signal's own variance.",
      "paper_value": "single evaluation per candidate in search",
      "evidence_file": "none",
      "evidence_value": "no search-run logs supplied",
      "status": "missing_evidence",
      "details": "probe_degenerate.json uses 4 seeds (7000/7017/7034/7051) but that is the separate stability probe, not the search; does not contradict, but does not confirm either."
    },
    {
      "id": 10,
      "location": "subsec:nas",
      "paper_text": "A validation batch contributes to the sum only if its loss is finite and below a fixed threshold; otherwise it is skipped.",
      "paper_value": "finite+threshold batch-rejection rule",
      "evidence_file": "proxy_diag/probe_degenerate.json",
      "evidence_value": "schema exposes 'rejected_batches' per row (all 0 in probe; val_batches=2)",
      "status": "ambiguous_mapping",
      "details": "The existence of a batch-rejection counter is consistent with the described mechanism, but the 'finite and below a fixed threshold' rule itself is not shown in any supplied artefact."
    },
    {
      "id": 11,
      "location": "subsec:nas",
      "paper_text": "the average is formed as total / max(count, 1), so if every batch of a trial is skipped the signal returns exactly zero",
      "paper_value": "max(count,1) normalizer; zero on all-skip",
      "evidence_file": "proxy_diag/probe_degenerate.json",
      "evidence_value": "schema exposes boolean 'degenerate' field (false in all 12 probe rows)",
      "status": "missing_evidence",
      "details": "A degenerate-flag mechanism exists in the code schema, but the exact normalizer and zero-return behaviour are not verifiable from supplied outputs."
    },
    {
      "id": 12,
      "location": "subsec:nas",
      "paper_text": "a candidate whose projection dimension differs from 128 is additionally given a learned linear layer ... this alignment layer exists only during search",
      "paper_value": "alignment layer iff dim != 128",
      "evidence_file": "proxy_diag/probe_degenerate.json",
      "evidence_value": "align_layer: false for all 4 dim=128 rows; true for all 4 dim=256 rows and all 4 dim=512 rows",
      "status": "exact_match",
      "details": "Flag pattern matches the claim exactly (12/12 rows)."
    },
    {
      "id": 13,
      "location": "subsec:nas, eq:argmin",
      "paper_text": "The search draws candidates uniformly at random from the 42 cells and returns the one with the smallest score",
      "paper_value": "uniform random sampling + argmin selection",
      "evidence_file": "none",
      "evidence_value": "no recorded search log supplied",
      "status": "missing_evidence",
      "details": "Synthetic simulation models this behaviour but does not evidence the real search's sampling."
    },
    {
      "id": 14,
      "location": "subsec:nas",
      "paper_text": "the number of draws is a protocol parameter (the recorded runs used 100, of which a variable number complete successfully)",
      "paper_value": "K = 100 draws per recorded search run",
      "evidence_file": "synthetic_mechanism/synthetic_mechanism.json",
      "evidence_value": "keys 'no_collapse__K100' and 'more_trials.K100' exist (alongside K10/K30/K300/K1000/K5000); n_broken_draws varies 1-200 across synthetic conditions",
      "status": "ambiguous_mapping",
      "details": "K=100 is confirmed as an experimental parameter in the synthetic mechanism, but this is a simulation parameter; no raw log of the actual recorded search runs is supplied. 'Variable number complete successfully' is consistent in spirit with varying broken-draw counts, but only within simulation."
    },
    {
      "id": 15,
      "location": "subsec:nas + subsec:implementation",
      "paper_text": "Stage 2 searches ... The search uses three quick training epochs per candidate with batch size 16 on the wood category, and is repeated 100 times per run.",
      "paper_value": "3 proxy epochs, batch 16, search category = wood, 100 repetitions",
      "evidence_file": "synthetic_mechanism/synthetic_mechanism.json",
      "evidence_value": "K100 key present; epochs/batch-size/category not in any supplied file",
      "status": "missing_evidence",
      "details": "'Repeated 100 times' maps ambiguously to K100; '3 epochs', 'batch size 16' and 'wood' have no artefact in the evidence set."
    },
    {
      "id": 16,
      "location": "sec:method intro",
      "paper_text": "The search runs on a single category and its result is applied to all categories of the dataset.",
      "paper_value": "single-category search, all-category deployment",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "Scope statement; no per-category search logs supplied."
    },
    {
      "id": 17,
      "location": "subsec:rd_detection",
      "paper_text": "adaptively pools it to 14x14 ... a first block produces the 14x14 stage at 256 channels ... 28x28 stage at 128 channels ... 56x56 stage at 64 channels",
      "paper_value": "decoder channel plan 256/128/64 at 14/28/56",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "Architecture description; no model dump or config file supplied."
    },
    {
      "id": 18,
      "location": "subsec:rd_detection",
      "paper_text": "the bottleneck width is fixed at B = 256 in the deployment path, independently of the feature dimension that the search may have selected",
      "paper_value": "B = 256 fixed at deployment",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "Key deployment-path constant; no config/model artefact supplied to confirm."
    },
    {
      "id": 19,
      "location": "subsec:rd_detection",
      "paper_text": "trained ... with AdamW at a learning rate of 10^-3 and weight decay 10^-4, a cosine schedule, and random horizontal flipping with probability 0.5",
      "paper_value": "AdamW, lr 1e-3, wd 1e-4, cosine, flip p=0.5",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "Optimizer/schedule/augmentation config not traceable in evidence set."
    },
    {
      "id": 20,
      "location": "subsubsec:scoring, eq:policy",
      "paper_text": "The published pipeline uses w = (1/3, 1/3, 1/3) and rho = mean ... rho in {mean, top-k, max}",
      "paper_value": "published policy: uniform weights, mean pooling; policy space of 3 pooling operators",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "No scoring-policy sweep or config artefact supplied in this segment."
    },
    {
      "id": 21,
      "location": "subsec:implementation",
      "paper_text": "the detector above is trained for 500 epochs with batch size 64 and seed 42, with a ResNet-18 teacher, on the MVTec-AD dataset",
      "paper_value": "500 epochs, batch 64, seed 42, ResNet-18 teacher, MVTec-AD",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "No detector training logs in evidence set."
    },
    {
      "id": 22,
      "location": "subsec:implementation",
      "paper_text": "Inputs are resized to 224x224 and normalised with ImageNet statistics.",
      "paper_value": "224x224 input, ImageNet normalisation",
      "evidence_file": "proxy_diag/backbone_params.json",
      "evidence_value": "backbone keys vit_tiny_patch16_224, deit_tiny_patch16_224, swin_tiny_patch4_window7_224 encode 224 input",
      "status": "ambiguous_mapping",
      "details": "224 input indirectly consistent with timm model names and the 56/28/14 feature-grid claims; ImageNet normalisation statistics not verifiable from supplied files."
    },
    {
      "id": 23,
      "location": "subsec:implementation",
      "paper_text": "All figures report image-level AUROC averaged over the 15 categories.",
      "paper_value": "AUROC averaged over 15 categories",
      "evidence_file": "none",
      "evidence_value": "no category-level results file supplied in this segment",
      "status": "missing_evidence",
      "details": "15 is the MVTec-AD category count (dataset fact) but no results artefact here confirms the aggregation actually used."
    },
    {
      "id": 24,
      "location": "subsec:implementation",
      "paper_text": "Where a backbone's memory footprint exceeds the device at batch size 64, training uses gradient accumulation with a smaller micro-batch, keeping the effective batch size at 64; the micro-batch used is recorded per run and this affects only the largest backbones.",
      "paper_value": "gradient accumulation, effective batch 64, micro-batch recorded per run",
      "evidence_file": "none",
      "evidence_value": "not present in supplied files",
      "status": "missing_evidence",
      "details": "Claim that micro-batch is 'recorded per run' is unverifiable; no run manifests supplied."
    },
    {
      "id": 25,
      "location": "subsec:genai",
      "paper_text": "the figures are direct plots of recorded measurements, produced by the released script make_figures_v2.py. All code, logs and derived artefacts are released so that every number can be traced to the run that produced it",
      "paper_value": "full traceability of all numbers to runs; figures from make_figures_v2.py",
      "evidence_file": "none",
      "evidence_value": "script and logs not in supplied evidence set; multiple protocol numbers in this section (epochs, batch sizes, seeds, lr, B=256, 0.3 weight, policy weights) have no corresponding artefact",
      "status": "unsupported_claim",
      "details": "Scope overclaim relative to the supplied evidence: within this very section, numerous quantitative protocol values cannot be traced to any provided file. The 14-backbone, 3-dim, alignment-layer and K=100 items are traceable; the rest are not."
    }
  ],
  "segment_verdict": "WARN",
  "notes": "Failure-mode scan: (1) number inflation - none detected; (2) best-seed cherry-pick - none in this segment; (3) config mismatch - none detectable, nothing in the text contradicts the supplied files; (4) aggregation mismatch - none; (5) delta arithmetic - 14x3=42 is exact; (6) caption/table mismatch - not applicable here; (7) scope overclaim - one instance: the traceability statement in subsec:genai ('every number can be traced to the run that produced it') exceeds what the supplied artefacts support, since many protocol constants in this methods section (100 teacher epochs, 500 detector epochs, batch sizes 64/16, seed 42, AdamW lr/wd, cosine, flip 0.5, B=256, 0.3 loss weight, scoring weights 1/3, '15 categories' aggregation) have no corresponding file. Verified items: 14 backbones and their family names (backbone_params.json), 3 feature dims {128,256,512} (probe_degenerate.json), the alignment-layer rule align_layer == (dim != 128) with 12/12 rows matching exactly, and K=100 as an experimental parameter (synthetic_mechanism.json, ambiguous: simulation parameter, not a recorded search log). Minor wording note: 'ViT and DeiT variants' is plural but only one variant of each exists in the search space. Verdict WARN rather than FAIL because no checked number conflicts with evidence; WARN rather than PASS because a substantial fraction of the section's quantitative protocol claims are unverifiable from the supplied evidence set."
}
```
