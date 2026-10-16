# Reviewer report — P0_preamble

_model: glm-5.3-flash; prompt 10470 chars; 62s; usage {"completion_tokens": 7173, "completion_tokens_details": {"reasoning_tokens": 6494}, "prompt_tokens": 2650, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 9823}_

**1. Errors / contradictions**

- **Blocking — top-k numbers disagree with the study's headline figures.** Abstract: 82.36→90.28; stated headline: 82.80→90.21. A third near-identical mean, 82.47, is then attributed to "the pipeline." Three baselines within 0.44 points, none defined. The abstract must match the body exactly, and 82.47 vs 82.36 must be distinguished (default scoring vs pre-pooling mean?).
- **Blocking — the inertness claim contradicts itself.** "Two thirds of its space proves deployment-inert" matches "2 of 3 feature dimensions never reach the deployed model," but the gloss "the searched feature dimension never reaches the detector" asserts the singular searched axis is wholly disconnected — a stronger, different claim. One of the two is wrong.
- **Major — variance attribution vs seed design.** 210 detectors = 14×15, i.e., one seed per cell, while the design allows up to 3 seeds. With one run per cell, run noise is not separable and the 95.32% category share may absorb seed variance. State how seeds enter the decomposition. Also reconcile 0.54/95.32 with the reported ranges (backbone 0.54–9.1%, category 53–98%): the abstract presents range endpoints as point estimates without saying so.
- Minor: 61% oracle-gap recovery checks out (7.09/11.65). "Twelve stable backbones"/"twelve searches" are internally consistent.

**2. Overstatement / missing controls**

- **Major:** "larger than the whole spread of the architectures it ranks" — no architecture-spread figure appears in this segment; unsupported as written.
- **Major:** the pipeline's 82.47 vs published 95.6–99.6 is a ~13-point deficit. A reviewer will ask whether the RD reimplementation is misconfigured and whether the diagnostics (variance shares, MDD, inertness) transfer to a correctly configured pipeline. This needs explicit justification, not just disclosure.

**3. Citations / definitions**

- The 0.5-point SOTA margin needs citations at first body use.
- Terminology drift: "score fusion" (title/keywords) vs "reduction step" vs "top-k pooling" — unify. "Accuracy variance" should be "AUROC variance" (AUROC is not accuracy).
- "5.7% of its median": define "span" and the referent of "its." Define "collapse-prone" (criterion).
- Reverse-distillation citation expected at first mention.

**4. Severity ranking**

- **Blocking:** top-k/baseline numeric mismatch; inert-space wording.
- **Major:** seed/variance confound; unsupported "whole spread" claim; transferability given the 82.47-vs-published gap.
- **Minor:** terminology drift; dangling modifier ("Instrumenting a reverse-distillation pipeline, two thirds…"); title tautology ("Axis Is Not the Axis"); file title differs from the submitted title — confirm canonical; abstract is ~197/200 words, verify under MDPI counting; ORCID placeholders.
