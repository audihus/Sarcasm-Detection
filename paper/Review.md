# Editorial Review — Efficient Late Fusion of Logistic Regression and Transformers for Indonesian Sarcasm Detection

**Panel:** EIC + 3 Peer Reviewers + Devil's Advocate (full review mode)
**Manuscript:** `paper/Paper.tex`
**Venue format:** `\documentclass[conference]{IEEEtran}` — an IEEE-sponsored conference paper, not the IEEE Access journal cited as the baseline.

**Editorial Decision: Major Revision**

---

## Phase 0 — Field Analysis & Panel Configuration

| Dimension | Assessment |
|---|---|
| Primary discipline | Natural Language Processing — figurative-language (sarcasm) classification |
| Secondary disciplines | Indonesian computational sociolinguistics (code-switching); ML systems/inference efficiency; classical multiple-classifier-system theory |
| Research paradigm | Quantitative, empirical — comparative benchmarking with ablations |
| Methodology type | Statistical modeling / machine learning benchmark study |
| Venue format | IEEE-sponsored conference paper. Profile (single dataset, 6 authors, 14 references) fits a regional/national IEEE informatics conference track. |
| Paper maturity | Numerically mature and internally verified (every table/figure arithmetic checks out), but **not submission-ready**: Author 4's block is unedited IEEEtran template placeholder text. |

**Panel configuration**

- **EIC** — Handling editor for an IEEE regional Informatics/Applied-AI conference track; background in efficient, low-resource NLP systems.
- **R1, Methodology** — ML evaluation methodologist: benchmarking rigor, significance testing, calibration.
- **R2, Domain** — Senior Indonesian NLP researcher, IndoNLU/IndoBERT lineage, figurative-language detection specialist.
- **R3, Perspective** — ML systems / MLOps and trust-&-safety practitioner: serving cost, deployment context.
- **Devil's Advocate** — adversarial stress-tester; scored separately from the 4-reviewer consensus count.

---

## EIC Review

**Identity:** Handling Editor, IEEE regional conference track on Applied NLP & Low-Resource Language Processing.
**Recommendation:** Major Revision · **Confidence:** 4/5

The paper takes on a real and well-scoped efficiency/accuracy trade-off, backs its central technique with established multiple-classifier-system theory (Kittler et al. 1998) rather than overselling novelty, and — unusually for a benchmarking paper — reports a negative result (classical-only ensembling fails to beat the single Optimized LR, Section IV-A) prominently rather than burying it. Structural coherence is good: the abstract's numbers (F1 0.7900, 50%, 36%) match the body and conclusion exactly, and I independently recomputed the F1/precision/recall/confusion-matrix arithmetic across Tables I–II and Fig. 2 — every value reconciles. That level of verifiable internal consistency is uncommon and should be credited. Set against that: the headline "new state-of-the-art" claim is carried by a bootstrap probability of only 77.3% (Section IV-B), which is a modest, sub-conventional level of statistical confidence for a claim framed this assertively, and one reviewer configuration issue (Weakness 1) blocks submission outright in the manuscript's current state.

### Strengths
1. **Verifiable numerics.** Every reported percentage, F1 value, and confusion-matrix count in Tables I–II, Fig. 2, and Section IV-B is internally self-consistent under direct recomputation — a strong positive signal about the paper's care.
2. **Honest negative result.** The classical-ensembling ablation (Section IV-A) that failed to beat the single Optimized LR is reported and used to motivate the cross-paradigm approach, rather than omitted.
3. **Theoretically grounded technique.** The fusion rule is correctly and modestly framed as "a special case of the classical sum rule" (Section III-D), avoiding an overclaim of novelty for a simple weighted average.
4. **Actionable efficiency story.** The parameter/latency accounting (Section IV-B) is independently useful to a practitioner reader even if the SOTA claim needs softening.

### Weaknesses
1. **Submission-blocking template defect.** Author 4's block is literal IEEEtran placeholder text — "Given Name Surname," "University Name," "City, Indonesia," "email@domain.ac.id." This must be corrected (real co-author or removed) before this manuscript can be submitted anywhere.
2. **"New SOTA" claim outruns its own evidence.** A 77.3% bootstrap win-probability (Section IV-B) is a plausible-but-not-decisive result; the Abstract and Conclusion state the SOTA claim without hedging that qualifier.
3. **Small test set underweighted as a limitation.** 538 test samples (~134 positive) is mentioned only briefly in the Conclusion; given it directly bears on the strength of the headline claim, it deserves earlier and more central treatment.

### Detailed comments
- **Journal/venue fit:** Good fit for an applied-NLP conference track focused on low-resource languages or efficient deployment of NLP systems; the efficiency framing and Indonesian-specific corpus are on-topic.
- **Originality:** Incremental but genuine: applying a well-established combination rule to a new task/language pairing, with a systematic ablation isolating whether the classical branch is necessary. See R2 for literature-positioning concerns.
- **Structural coherence:** Strong — abstract, results, and conclusion tell one consistent, mutually-reinforcing story with matching numbers throughout.

*Recommendation to peer reviewers: R1, please dig into whether the significance test and the "best of 6 pairings" selection are consistent with each other. R2, please assess whether the related-work positioning against sarcasm-specific (not just generic late-fusion) prior art is adequate.*

---

## Peer Reviewer 1 — Methodology

**Identity:** ML evaluation methodologist — benchmarking rigor, significance testing, model calibration.
**Recommendation:** Major Revision · **Confidence:** 5/5

The core pipeline is methodologically disciplined in several specific, checkable ways: threshold selection uses the correct exhaustive-search-over-realized-scores procedure (Lipton et al. 2014) rather than a coarse grid; validation/test separation is stated explicitly; and class-imbalance handling is well-motivated. I recomputed every F1 value from its reported precision/recall pair and every confusion-matrix count from precision/recall/n in Tables I–II and Fig. 2 — all reconcile exactly, which is a genuine strength worth stating plainly. The main problems are at the level of statistical inference and provenance: a significance claim that doesn't account for a 6-way model-selection step, missing confidence intervals everywhere except the single headline comparison, and an ambiguous statement about whether the classical baseline in Table I was reproduced under the same conditions as the proposed method or lifted from the literature.

### Strengths
1. **Principled threshold selection.** Citing and correctly applying Lipton et al. (2014)'s result that the F1-optimal threshold coincides with a realized score value (Section III-D) is textbook-correct and better than the common naive-grid approach.
2. **Explicit leakage discipline.** "The test set is evaluated exactly once per configuration" (Section III-D) is a clear, checkable commitment.
3. **Reproducible arithmetic.** Every derived quantity I checked (F1 from P/R; confusion-matrix cell counts from P/R/n≈134 positives) is internally consistent — no computational errors detected anywhere in Tables I–II or Fig. 2.

### Weaknesses

**W1 — Selection across 6 configurations not corrected in the significance test** *(Table II · Section IV-B, "A paired bootstrap test...")*
Table II reports late-fusion results for six independent transformer pairings, transparently — that transparency is itself good practice. But the headline claim is drawn from whichever pairing (XLM-R-base) scored highest on the fixed 538-sample test set, and the bootstrap significance test is computed only for that winning pairing versus the historical SOTA. This is a "best-of-K on a fixed test set" selection that the significance statement doesn't correct for. Either apply a multiple-comparison correction (Bonferroni or a max-statistic permutation test across all 6 pairings) or explicitly caveat that 77.3% is a single-comparison probability conditional on already having picked the best performer.

**W2 — Ambiguous baseline-reproduction provenance** *(Table I · Section IV-A, "The Optimized LR advances the baseline...")*
Table I attributes the Logistic Regression row (F1 = 0.7171) to `\cite{idsarcasm}`, implying it's the original paper's reported number. The prose calls it "its reproduced score," implying the authors reran it themselves. These aren't interchangeable: if 0.7171 is a literature number (different code/library-version/seed/exact split), the claimed +0.0338 gain from "optimization" is confounded with environment drift. State explicitly, in the table caption or a footnote, which one it is.

**W3 — No confidence intervals on primary metrics** *(Tables I–II · Fig. 3)*
A single bootstrap probability is reported for the headline SOTA comparison, but no CI or variance is given for any individual F1/Precision/Recall value, nor for the fusion-weight estimates w* in Fig. 3. The authors already have the bootstrap machinery built for the headline test — extending it to report CIs throughout Section IV would substantially strengthen every claim in the results section.

**W4 — Hyperparameters tuned on a 268-sample validation set, with no sensitivity analysis** *(Section III-B–III-D)*
*C*, the decision threshold τ, and the fusion weight *w* are all selected on a single 268-sample validation split (~67 positive examples). The claim in Section IV-C that the search "consistently" favors the classical branch is based on one realization of a small validation set; a bootstrap-resampled sensitivity check on w* would materially strengthen that claim.

**W5 — Classical-ensembling ablation underspecified** *(Section IV-A, "To verify whether this statistical ceiling...")*
"Three base classifiers," a stacking meta-learner, and the split used for the 5,000 bootstrap resamples are not named. This ablation is load-bearing for the paper's central narrative and needs the same level of methodological detail as the main pipeline.

**W6 — Latency measurement protocol unspecified** *(Section IV-B, "Measured on a single NVIDIA T4 GPU...")*
Batch size, repetition count, warm-up handling, and variance are not reported for the 10.22 ms / 6.51 ms per-sample figures.

### Reproducibility
No statement on code or checkpoint release. Given the result depends on subtle, hard-to-reproduce environment details (forcing eager attention to match SDPA-era published numbers), a public repository would materially help.

### Methodological fallacies checklist
Primary concern is a mild look-elsewhere effect (W1). No Simpson's paradox, survivorship bias, or unwarranted causal claims detected — the paper's claims stay appropriately predictive/comparative in scope.

---

## Peer Reviewer 2 — Domain

**Identity:** Senior Indonesian NLP researcher — IndoNLU/IndoBERT lineage, figurative-language detection.
**Recommendation:** Minor–Major Revision · **Confidence:** 5/5 (Indonesian NLP) · 3/5 (classifier-combination theory)

Terminology and technical framing around IndoBERT/XLM-R and the IndoNLU lineage are precise throughout, and the decision to freeze the reference benchmark's exact six checkpoints in inference-only mode (Section III-C) is a methodologically disciplined way to guarantee a fair comparison — the eager-vs-SDPA numerical-divergence catch (Section III-C) in particular reflects genuine familiarity with the `transformers` library that most reviewers would miss. My main concerns are about literature positioning (the closest task-specific prior art — hybrid lexical+neural sarcasm/irony detectors — is absent) and a mismatch between the paper's rhetorical framing (code-switching as a defining challenge) and what the method actually does about it (nothing code-switching-specific).

### Strengths
1. **Precise technical terminology** throughout Section III (pooler_output, sublinear tf, IndoNLU lineage) with no misuse detected.
2. **Disciplined baseline reuse.** Freezing the reference paper's exact six checkpoints in inference-only mode (Section III-C) is the right design for a fair, apples-to-apples comparison.
3. **Correctly scoped theoretical framing.** Citing Kittler et al. (1998) as "a special case of the classical sum rule" (Section III-D) is accurate and appropriately modest.

### Weaknesses

**W1 — Missing directly relevant prior art: hybrid classical+neural sarcasm/irony detection** *(Section II, "Emerging studies have demonstrated the promise...")*
The two late-fusion precedents cited (Golestani & Moattar 2026; Ramya & Manu 2026) are Steam-review sentiment and psychological-stress detection, not sarcasm/irony — honestly acknowledged as a gap. But the sarcasm/irony-detection literature itself has a substantial history of combining lexical/hand-crafted features with neural representations (dating to SemEval-era irony-detection systems); none of that closer, task-specific lineage is cited. The novelty claim would be considerably sharper positioned against sarcasm-specific hybrids in other languages, not only against cross-task late-fusion papers.

**W2 — Framing/mechanism mismatch on code-switching** *(Section I, ¶1 & ¶3)*
The Introduction motivates the problem twice via Indonesian Twitter's "code-switching" and "irregular morphological variations." Neither the TF-IDF+LR branch nor the off-the-shelf multilingual transformer branch does anything code-switching-specific (no language-ID features, no code-switch-aware normalization). The paper is entitled to use generic models, but the framing oversells engagement with this specific challenge; either tone it down or add analysis of whether code-switched tweets are disproportionately represented among the rescued/still-missed cases in Section IV-E.

**W3 — Table I citation ambiguity (domain-accuracy angle)** *(Table I)*
Overlaps R1's W2: if 0.7171 is drawn directly from `\cite{idsarcasm}`'s table without re-running under the authors' own pipeline, attributing it to that citation without a footnote clarifying provenance is a citation-precision issue.

**W4 — "Lexicon-free" framing slightly overclaims** *(Abstract · Section III-B)*
TF-IDF n-gram weighting is itself doing implicit lexical-signal extraction, as the qualitative analysis in Section IV-E even confirms ("grounded in document-frequency weighting"). Consider "dictionary-free, corpus-derived" instead of "lexicon-free" to avoid implying zero lexical sensitivity.

### Missing key references
- A SemEval-style irony/sarcasm shared-task paper combining lexical/n-gram and neural signals — the closest task-specific precedent for this paper's core mechanism.
- An Indonesian-social-media code-switching NLP reference, if the Introduction keeps its code-switching framing, to substantiate the claim with data rather than assertion.

### Contribution assessment
The paper's best-evidenced contribution is the systematic ablation showing (a) classical-only ensembling saturates (r = 0.877, 47% bootstrap win rate) and (b) the fusion-weight search structurally favors the classical branch across all six transformer pairings (w* ≥ 0.625) — a real, non-obvious empirical finding about this corpus, independent of whether the headline SOTA number survives statistical scrutiny (see R1, DA). I'd recommend leaning more heavily on this finding as the primary contribution, with the SOTA number repositioned as a secondary, appropriately-hedged result.

---

## Peer Reviewer 3 — Perspective

**Identity:** ML systems / MLOps and trust-&-safety practitioner — model serving cost, deployment context.
**Recommendation:** Minor Revision (from this angle) · **Confidence:** 3/5

*As a systems/deployment practitioner, I may not fully share this venue's methodological conventions — but from where I sit, the paper's efficiency accounting is the right idea, incompletely scoped.* Params-and-latency is the correct axis for a deployment-oriented contribution, and it's computed transparently against a named hardware target. What's missing is everything the raw compute numbers don't capture: total cost of maintaining two independently-versioned model artifacts, and any discussion of where this classifier would actually be used and what that implies about the precision/recall trade-off the fusion makes.

### Strengths
1. **Right efficiency axis, transparently measured.** Params and per-sample latency against a named GPU target (Section IV-B) is exactly what a deployment-minded reader wants to see.
2. **PII masking** in the underlying corpus (Section III-A) is a good baseline privacy practice, worth crediting explicitly.
3. **Concrete rescued-prediction examples** (Section IV-E / Table III) are genuinely useful to a practitioner reader — they show where the fusion helps, not just that it helps in aggregate.

### Weaknesses

**W1 — "Negligible overhead" understates total cost of ownership** *(Section IV-B, "Beyond predictive accuracy...")*
"<0.02% parameter density, <0.1% runtime" is true in raw compute terms, but a production system now has two independently-versioned artifacts to serve, monitor, and retrain — a TF-IDF vocabulary + LR weight file and a transformer checkpoint — each with its own dependency stack (scikit-learn vs. transformers/PyTorch) and its own drift-monitoring needs. Slang/vocabulary drift for the LR branch is an ongoing cost the one-time inference-latency number doesn't capture. Scope the "negligible overhead" claim explicitly to inference compute, not total cost of ownership.

**W2 — No discussion of downstream use or stakeholder cost asymmetry** *(Section IV-B, "This gain is not uniform..." · Table III)*
Table III's qualitative examples include politically-charged content (a public official's name, a religious-gathering reference) alongside a profanity-laden tweet — realistic and useful, but the paper never states where such a classifier would actually be deployed (content moderation? social listening? sentiment dashboards?) or what the false-positive/false-negative cost asymmetry is in that setting. The fusion explicitly trades precision down for recall up (0.7937→0.7551 vs. 0.7463→0.8284); whether that's the right trade depends entirely on the use case, which the paper never grounds.

**W3 — No retraining/drift story for the classical branch** *(Section III-B)*
Indonesian social-media slang evolves quickly; a TF-IDF-based LR classifier is more brittle to vocabulary drift than a pretrained transformer's subword tokenization. Since the paper's central finding is that this branch should dominate the fusion weight (w* ≥ 0.625), the system's long-run robustness depends on how often it needs retraining — not discussed.

### Cross-disciplinary reading recommendations
- Multi-model serving / total-cost-of-ownership literature from ML systems.
- Trust-and-safety literature on precision/recall trade-offs in content-moderation classifiers.
- Sociolinguistic work on Indonesian internet-slang drift over time.

---

## Devil's Advocate Review

The paper does careful, verifiable empirical bookkeeping and honestly reports a negative result. Both are real strengths, acknowledged above. But its central causal narrative deserves a hard look before publication.

### Strongest counter-argument

> The paper's central causal claim — that "the classical statistical model acts as the robust primary anchor" (Section IV-C) because of some architectural complementarity between attention and bag-of-words signals — is not the most parsimonious explanation for the observed data, and the paper never rules out a simpler one. The Optimized LR branch received extensive, dataset-specific tuning on the validation set: regularization-strength grid search, TF-IDF feature engineering, class-weight balancing, and threshold tuning — all optimized directly against this corpus. The six transformer branches, by contrast, are frozen checkpoints borrowed from a different paper, tuned on nothing specific to this corpus beyond a numerical-precision fix (forcing eager attention). It would be unsurprising if a weighted-average fusion did *not* lean toward the branch that was extensively tuned on the target data over the branch that was not tuned at all. The paper's interpretation — that this reflects a fundamental representational superiority of bag-of-words statistics over attention for this task — is a considerably stronger and more publication-worthy claim than "the branch we tuned on this data got more weight than the branch we didn't tune," and the paper presents no evidence distinguishing the two. A direct falsification test is available and absent: freeze the LR at default hyperparameters (no grid search, no threshold tuning) and re-run the fusion-weight search. Until that confound is addressed, "classical model as dominant anchor" is a claim about tuning asymmetry dressed as a claim about architecture.

### Issue list

#### CRITICAL

| # | Dimension | Issue | Location |
|---|---|---|---|
| 1 | Core Thesis / Alternative Paths | Tuning-asymmetry confound in the "classical model as dominant anchor" interpretation — see Strongest Counter-Argument. No field-norm dependency; this is an internal-validity finding. | §IV-C ¶225; Conclusion ¶256 |
| 2 | Logic Chain | "New SOTA" is claimed with a 77.3% bootstrap win-probability — sub-conventional confidence — and reported as confirmatory ("A paired bootstrap test confirms this gain"). **Field-norm boundary:** the paper's own cited significance-testing reference, Dror et al. (2018) — the ACL-endorsed guide to significance testing in NLP — recommends conventionally-thresholded testing before superiority claims. **Evidence-crossing rationale:** the paper runs exactly this test, obtains a sub-conventional result (≈p = 0.23 one-sided), and reports it as confirmatory rather than suggestive. | Abstract; §IV-B ¶200; Conclusion ¶258 |

#### MAJOR

| # | Dimension | Issue | Location |
|---|---|---|---|
| 1 | Cherry-Picking | Best-of-6 selection on the fixed test set (Table II) is fully disclosed but not corrected for in the significance claim. **Field-norm boundary:** multiple-comparisons correction for a "best of K" headline result is standard statistical practice broadly. **Evidence-crossing rationale:** no correction is applied or discussed anywhere in §III-E or §IV-B. | Table II; §IV-B |
| 2 | Logic Chain | Ambiguous baseline-reproduction provenance (Table I, 0.7171) undermines the clean attribution of the +0.0338 gain to the described optimizations alone (see R1 W2, R2 W3). | Table I; §IV-A ¶166 |

#### MINOR

| # | Dimension | Issue | Location |
|---|---|---|---|
| 1 | Overgeneralization | Closing abstract claim generalizes from one dataset/language/platform/task to "low-resource figurative language classification" broadly. | Abstract |
| 2 | "So What?" | If Critical #1 holds, the paper's real finding is narrower and more modest ("a well-tuned lightweight model can rival an undertuned frozen transformer") than the one it currently claims. | Conclusion ¶256 |

### Ignored alternative explanations

1. **Calibration mismatch.** Table III shows XLM-R-base emitting extreme, overconfident probabilities (0.964, 0.991, 0.256, 0.356) on cases it gets wrong, while LR's outputs cluster more moderately. A weighted average would naturally favor a better-calibrated branch regardless of true predictive value. The paper never measures calibration (ECE, Brier score) for either branch, so it cannot distinguish "LR carries more genuine signal" from "LR is simply better calibrated." A cheap test: temperature-scale the transformer's logits before fusion and re-run the weight search — if w* drops substantially, calibration, not architecture, was doing the work.
2. **Frozen-checkpoint alternative.** The transformer branch is used "strictly in inference-only mode" (§III-C) — a reasonable design choice for the efficiency story, but it means the transformer never sees this validation set's distribution at all, whereas LR is fit directly to it. Same underlying point as Critical #1, reframed as a design choice worth acknowledging rather than treating the outcome as evidence about model classes in general.

### Missing stakeholder perspectives
- Downstream users of a deployed sarcasm classifier (moderators, trust-&-safety teams) whose cost structure determines whether the fusion's precision-down/recall-up trade is actually desirable.
- The original IdSarcasm authors — attribution is present and correct throughout, but a sanity-check exchange on the baseline-reproduction ambiguity (Major #2) would be worthwhile before publication.

### Unexamined premise
The paper frames its question as "which paradigm wins, and how should they be combined" — but a more basic unexamined premise is that six borrowed, frozen transformer checkpoints (trained under a different paper's preprocessing and hyperparameter choices) fairly represent "what transformers can do" on this corpus. A transformer fine-tuned with the same care given to the LR branch might close some or all of the attributed gap. The paper's no-fine-tuning design (§III-D) forecloses this comparison by construction — a defensible scope choice for an efficiency paper, but it means "LR is structurally superior" is really "LR beats a frozen, off-the-shelf transformer": a narrower, more defensible claim than the one actually made.

### Observations (non-defects)
- The internal numerical consistency across every table and figure (independently re-derived by this reviewer) is unusually high quality and should be preserved through revision.
- Including the classical-ensembling negative result prominently, rather than only reporting what worked, is good scientific practice.

---

## Editorial Decision

Dear Author(s),

Thank you for submitting "Efficient Late Fusion of Logistic Regression and Transformers Models for Indonesian Sarcasm Detection." The manuscript was reviewed by four independent panelists (EIC + 3 peer reviewers) plus a Devil's Advocate stress-test.

### Decision: Major Revision

### Consensus analysis

No finding reached full 4/4 or 3/4 panel consensus — the four non-DA reviewers largely surfaced distinct, complementary concerns rather than converging on the same points. Two items were independently corroborated by two reviewers each:

- **Corroborated (2/4):** Ambiguous baseline-reproduction provenance in Table I — raised independently by R1 (Methodology) and R2 (Domain), and separately flagged by the Devil's Advocate as Major.
- **Corroborated (2/4):** Significance-framing overreach on the "new SOTA" claim — raised by EIC and R1, and independently escalated by the Devil's Advocate to Critical (with field-norm grounding in the paper's own cited Dror et al. 2018).

All remaining findings (author-block placeholder, code-switching framing, MLOps cost framing, ensembling-ablation underspecification, missing CIs, missing sarcasm-specific citations) are single-reviewer but each carries a high (4–5/5) confidence score and rests on directly verifiable textual evidence, not domain judgment calls — so each still drives the roadmap below despite lacking cross-reviewer corroboration. No genuine disagreements (reviewers holding opposite positions on the same claim) were found; this is a case of complementary, non-overlapping concerns rather than an editorial arbitration problem.

### Devil's Advocate critical findings

Per the panel's iron rule, a Critical Devil's Advocate finding forecloses an Accept decision regardless of the 4-reviewer count. Both Critical findings here were independently verified by this editor against the manuscript text and are validated, not overreach:

- **Tuning-asymmetry confound** (DA Critical #1) — not directly named by name in the 4-panel, but closely related in substance to R1's baseline-provenance concern and R3's observation that the transformer branch receives no dataset-specific tuning. Requires either a falsification experiment (fixed-hyperparameter LR ablation) or an explicit rescoping of the "classical model as anchor" claim.
- **Sub-conventional significance reported as confirmatory** (DA Critical #2) — directly corroborated by EIC's and R1's independent significance concerns. Requires either hedged language matching the actual evidence, or a strengthened test (larger/pooled test set, multiple-comparison-corrected across all six pairings).

### Decision rationale

The paper's empirical execution is careful and internally verified — I independently re-derived every reported F1 and confusion-matrix figure and found no arithmetic errors, and the classical-ensembling negative result is reported honestly rather than suppressed. These are genuine strengths that should be preserved. However, the manuscript cannot proceed past Major Revision for three independent reasons: (1) a submission-blocking template defect (Author 4's placeholder block) that must be fixed regardless of scientific content; (2) two Devil's-Advocate Critical findings — a plausible, unaddressed confound in the paper's central causal claim, and a headline "new SOTA" claim asserted more confidently than its own 77.3% bootstrap evidence (and its own cited significance-testing standard) supports; and (3) two further Major-level, cross-corroborated concerns about baseline provenance and uncorrected multiple comparisons. None of these require the paper to be re-conceived — the required fixes are a hyperparameter-symmetry ablation, a calibration check, corrected/hedged significance language, and a provenance clarification — but they are substantive enough that a re-review of the revised manuscript is warranted.

### Summary of key issues

1. Author 4 placeholder block — blocking, trivial to fix (EIC).
2. Tuning-asymmetry confound in the "classical anchor" causal claim (Devil's Advocate, Critical).
3. "New SOTA" claim not hedged to match its own 77.3% bootstrap evidence (Devil's Advocate, Critical; corroborated by EIC, R1).
4. Ambiguous Table I baseline provenance (R1, R2; corroborated by Devil's Advocate).
5. Uncorrected best-of-6 selection in the significance test (R1; corroborated by Devil's Advocate).

---

## Revision Roadmap

### Priority 1 — Structural revisions (must fix)

- [ ] **R1** Replace the Author 4 placeholder block with real author information or remove the author. *— EIC*
- [ ] **R2** Run the tuning-asymmetry falsification test (fixed-hyperparameter LR, no threshold tuning, re-run the fusion-weight search) or explicitly rescope the "classical model as dominant anchor" claim to acknowledge the confound. *— Devil's Advocate Critical, R1/R3*
- [ ] **R3** Rehedge the "new SOTA" claim throughout (Abstract, §IV-B, Conclusion) to match the 77.3% bootstrap result, or strengthen the evidence (larger/pooled test set, corrected multi-comparison test across all 6 pairings). *— Devil's Advocate Critical, EIC, R1*
- [ ] **R4** Clarify in Table I's caption and §IV-A prose whether the 0.7171 LR baseline is a literature number or a same-pipeline reproduction; correct the citation attribution accordingly. *— R1, R2, Devil's Advocate*

### Priority 2 — Content supplementation (should fix)

- [ ] **S1** Apply a multiple-comparisons correction (or explicit caveat) to the significance test given 6 pairings were evaluated on the same fixed test set. *— R1, Devil's Advocate*
- [ ] **S2** Report confidence intervals for F1/Precision/Recall in Tables I–II and for w* in Fig. 3. *— R1*
- [ ] **S3** Add a calibration check (ECE/Brier, or a temperature-scaling ablation) to test the calibration-mismatch alternative explanation. *— Devil's Advocate*
- [ ] **S4** Specify the classical-ensembling ablation's classifiers, stacking meta-learner, and the split used for the 5,000 bootstrap resamples. *— R1*
- [ ] **S5** Cite sarcasm/irony-specific hybrid (lexical+neural) detection literature and narrow the novelty claim in Section II accordingly. *— R2*
- [ ] **S6** Soften the code-switching framing in the Introduction, or add analysis of code-switched examples in the qualitative section. *— R2*
- [ ] **S7** Report the latency-measurement protocol (batch size, repetitions, warm-up, variance). *— R1*
- [ ] **S8** Add discussion of deployment/downstream-use context and total-cost-of-ownership, not only inference-time overhead. *— R3*

### Priority 3 — Text and formatting (nice to fix)

- [ ] **T1** Replace "lexicon-free" with "dictionary-free, corpus-derived" or similar. *— R2*
- [ ] **T2** Temper the Abstract's closing generalization to the paper's actual single-dataset, single-language, single-platform scope. *— Devil's Advocate*
- [ ] **T3** State code/checkpoint release plans for reproducibility. *— R1*

### Suggested revision window

6–8 weeks — S1/S3 involve new experiments (a fixed-hyperparameter LR ablation and a calibration check), which is the pacing factor.

---

*Full-panel simulated review generated per `.agents/skills/academic-paper-reviewer/SKILL.md`. Read-only: no manuscript content was modified in producing this report.*
