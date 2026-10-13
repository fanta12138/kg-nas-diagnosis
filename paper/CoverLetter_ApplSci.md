# Cover Letter — *Applied Sciences* (MDPI)

**To:** The Editorial Office, *Applied Sciences*
**Date:** 2026-10-13
**Article type:** Article
**Manuscript title:** The Architecture Axis Is Not the Axis: A Diagnostic Study of Neural Architecture Search and Score Fusion in Unsupervised Industrial Anomaly Detection

**Authors:** Hong Li, Yuanwu Zhou, Peng Deng\* (\*corresponding author, dengpeng@gxust.edu.cn)
School of Civil and Architectural Engineering, Guangxi University of Science and Technology, Liuzhou 545006, China

---

Dear Editor,

We are pleased to submit our manuscript for consideration as an **Article** in *Applied Sciences*. The manuscript is submitted in **free format**, as the journal permits at first submission; LaTeX sources and all figures are provided, and we will reformat to the MDPI template at the revision stage if required.

## 1. The practical problem, and why it belongs in *Applied Sciences*

Automated optical inspection is deployed on real production lines, and the detectors that run there are trained without defect samples. Companies commissioning such a system must make two engineering decisions: which network to deploy, and how the per-pixel discrepancy map is reduced to the accept/reject score. **Only the second decision is free once a model is trained, and the industry standard for both is undocumented.**

This paper makes that decision visible and quantifies it on a real benchmark. The practical outcome for anyone running a reverse-distillation inspector is concrete: **on an already-trained model, with no retraining and no labelled defects, changing only the reduction step raises mean AUROC from 82.36% to 90.28%** and repairs a category that currently fails below chance (from 42% to 84%). The reduction can even be selected without any anomaly label, by separating synthetic pseudo-anomalies from clean images — a usable commissioning procedure, not a laboratory proof of concept.

Alongside this, the paper reports a measurement that matters to every engineer who has to trust a published comparison: **the standard evaluation design cannot resolve the margins it is used to report.** On MVTec-AD the contested margin between successive state-of-the-art generations is 0.5 AUROC points, while a single-category comparison resolves 2.98 points. This is an applied measurement-reliability result with direct consequences for inspection-system procurement, and to our knowledge it has not been reported for image anomaly detection.

## 2. Summary of contributions

1. **Resolvability of the standard benchmark design.** From a 210-detector matrix we estimate the task's noise floor and anchor it to the externally contested margin: a single-category, single-run comparison resolves 2.98 points against a 0.5-point margin; 240 independent runs per backbone would be needed on one category, or 532 categories by pairing. We also show the requirement is method-specific (a PatchCore-style baseline needs only 17).
2. **A diagnosis of the architecture axis.** Two thirds of a standard 42-cell search space is deployment-inert — the searched feature dimension never reaches the deployed detector because the encoder hard-codes the bottleneck width. The distillation signal that ranks candidates separates the twelve stable backbones by 5.7% of its median and returns exactly zero for two of them, which the minimising rule then prefers. Twelve independent searches all returned a collapse-prone backbone, performing **worse than uniform random selection** (80.91% vs 81.65%, at the 1.2th percentile).
3. **A controlled comparison of a complete search space.** 14 backbones × 15 categories under one protocol (210 detectors), decomposing to **0.54%** of the accuracy variance for the architecture against **95.32%** for the category, with the backbone ordering not transferring between categories.
4. **A label-free procedure for the axis that does matter.** Top-*k* pooling, with the policy chosen by separation of synthetic pseudo-anomalies, recovering 61% of the oracle gap (mean over three training seeds), without retraining and without any real defect label.

## 3. Positioning

We do not claim state of the art. The pipeline reaches 82.47% mean AUROC against published values of 95.6–99.6%, and the manuscript reports that gap rather than obscuring it. Our contribution is a quantitative account of which design choices produce the number, how much of the number the benchmark can measure at all, and where the leverage actually lies. All code, logs and derived artefacts are released so that every number traces to the run that produced it; the figure generator reproduces the published figures byte for byte from the frozen records.

## 4. Required statements

> **We confirm that neither the manuscript nor any parts of its content are currently under consideration for publication with or published in another journal.**
>
> **All authors have approved the manuscript and agree with its submission to *Applied Sciences*.**

## 5. Declarations

- **Authorship.** All authors listed qualify for authorship and have approved the submitted version. Author contributions are stated in the manuscript (initial contributions are not equal; the corresponding author is P.D.).
- **Competing interests.** The authors declare no conflict of interest.
- **Funding.** This work was supported by the Guangxi Natural Science Foundation (Grant No. `2025GXNSFHA069076`) and the Guangxi University of Science and Technology Doctoral Fund (Grant Nos. `26Z15`, `26Z25`). The funders had no role in the design of the study, the analysis, or the decision to publish.
- **Data availability.** This work uses only the publicly available MVTec-AD benchmark; no new image data were generated. All derived artefacts are released with the code.
- **Code availability.** The complete source code, raw logs and frozen results are publicly released; the repository link is given in the manuscript and will be confirmed to the Editorial Office.
- **Ethics.** Not applicable: this study did not involve human participants or animals.
- **Generative AI.** Disclosed in full in Section 3.5 of the manuscript. An AI assistant wrote the experimental code and produced drafts of the text under the authors' direction; it did not determine the research questions, the experimental design or the interpretation. The authors verified every reported number against the recorded outputs and take full responsibility for the content.
- **Prior submissions to MDPI.** None.
- **Reviewers.** Suggested and opposed reviewers are provided in the submission system, as the journal instructs, and are deliberately omitted here.

We thank you for considering our manuscript and look forward to the reviewers' comments.

Yours sincerely,

**Peng Deng**, on behalf of all authors
School of Civil and Architectural Engineering
Guangxi University of Science and Technology
No. 2 Wenchang Road, Liuzhou, 545006, Guangxi, China
Email: dengpeng@gxust.edu.cn
