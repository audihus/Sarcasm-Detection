# Citation candidates (researched, not currently in Paper.tex)

Found via `literature-search-arxiv` skill + WebSearch during citation research for
Introduction/Literature Review. All 7 "used" entries were added to `Paper.tex` and
then reverted at the user's request (validity not yet independently confirmed) —
kept here so they don't need to be re-researched if wanted later.

**Before re-adding any of these: verify the DOI/venue independently** (e.g. open the
DOI link, check it resolves to the exact title/authors below). These were found via
AI web search, not manually cross-checked page-by-page.

## Used (added then reverted)

1. **Kabra et al. (2023)** — "Multi-lingual and multi-cultural figurative language
   understanding," in *Findings of the Association for Computational Linguistics:
   ACL 2023*, Toronto, Canada, pp. 8269–8284, doi: 10.18653/v1/2023.findings-acl.525.
   Authors: A. Kabra, E. Liu, S. Khanuja, A. F. Aji, G. Winata, S. Cahyawijaya,
   A. Aremu, P. Ogayo, G. Neubig.
   Suggested placement: Introduction, tied to "figurative language" / confounds
   standard classifiers claim.

2. **Chen, Lin, Li, & Liu (2024)** — "A survey of automatic sarcasm detection:
   Fundamental theories, formulation, datasets, detection methods, and
   opportunities," *Neurocomputing*, vol. 578, art. 127428,
   doi: 10.1016/j.neucom.2024.127428.
   Suggested placement: Literature Review opening paragraph, companion survey to
   Joshi2017 (already cited in Paper.tex).

3. **Ghanem, Karoui, Benamara, Rosso, & Moriceau (2020)** — "Irony detection in a
   multilingual context," in *Advances in Information Retrieval (ECIR 2020)*,
   Lecture Notes in Computer Science, vol. 12036, Springer, Cham, pp. 141–149,
   doi: 10.1007/978-3-030-45442-5_18.
   Suggested placement: Literature Review, cross-lingual/cross-cultural variability
   clause. Note: uses monolingual word representations, NOT cross-lingual
   transformer embeddings — do not attach it to a XLM-R/mBERT-specific sentence.

4. **Potamias, Siolas, & Stafylopatis (2020)** — "A transformer-based approach to
   irony and sarcasm detection," *Neural Computing and Applications*, vol. 32,
   pp. 17309–17320, doi: 10.1007/s00521-020-05102-3.
   Suggested placement: Literature Review, transformer-paradigm paragraph. Note:
   general transformer architecture for English irony/sarcasm — not IndoBERT/XLM-R
   specific, don't imply otherwise.

5. **Gupta, Mittal, & Jain (2025)** — "Multimodal sarcasm detection: A survey of
   methods, fusion techniques, dataset analysis, and open issues," in *Innovative
   Computing and Communications (ICICC 2025)*, Lecture Notes in Networks and
   Systems, vol. 1438, Springer, Singapore, doi: 10.1007/978-981-96-7707-8_23.
   Suggested placement: Literature Review fusion paragraph — supports the novelty
   claim that fusion research *within sarcasm detection* has concentrated on
   multimodal (not text-only) inputs.

6. **Gole, Nwadiugwu, & Miranskyy (2024)** — "On sarcasm detection with OpenAI
   GPT-based models," in *Proc. 34th Int. Conf. Collaborative Advances in Software
   and COmputiNg (CASCON)*, IEEE, pp. 1–6,
   doi: 10.1109/CASCON62161.2024.10837875.
   Suggested placement: Introduction, grounds the "zero-shot large language models
   (LLMs)" mention (previously an orphan/uncited claim).

7. **Dakwah, Firdaus, Furizal, & Faresta (2024)** — "Sentiment analysis on
   marketplace in Indonesia using support vector machine and Naïve Bayes method,"
   *Jurnal Ilmiah Teknik Elektro Komputer dan Informatika (JITEKI)*, vol. 10, no. 1,
   pp. 39–53, doi: 10.26555/jiteki.v10i1.28070.
   Suggested placement: Literature Review classical-models paragraph, alongside
   Wijaya2013/Wibowo2020 as a more recent (2024) example. Note: JITEKI is a national
   Indonesian journal, not a top-tier international venue — legitimate/peer-reviewed
   but modest.

## Considered and rejected (do not re-add without a new supporting clause)

8. **Cahyawijaya et al. (2024) — "Cendol"** — "Open Instruction-tuned Generative
   Large Language Models for Indonesian Languages," ACL 2024 (Vol. 1: Long Papers),
   Bangkok, pp. 14899–14914.
   Rejected: generative-LLM instruction-tuning paper, not classification/sarcasm —
   only tangentially "recent Indonesian NLP research," not methodologically
   relevant to this paper's late-fusion classifier approach.

9. **Talaat (2023)** — "Sentiment analysis classification system using hybrid BERT
   models," *Journal of Big Data*, vol. 10, art. 110,
   doi: 10.1186/s40537-023-00781-w.
   Rejected: architectural hybrid (BERT + BiGRU/BiLSTM layers stacked), NOT
   late-fusion of two independent classifiers' probabilities like this paper (or
   like Golestani2026/Ramya2026, already cited) — placing it next to those would
   misleadingly imply the same fusion mechanism.

## Notes on venues already upgraded from arXiv preprint → published

These upgrades (found via WebSearch) are still valid facts even though the
citations themselves were reverted:
- Ghanem2020: arXiv:2002.02427 → published ECIR 2020 (see #3 above).
- Chen2024survey: replaces an earlier arXiv-only "Was that Sarcasm?" survey by
  Bagga et al. — Neurocomputing is the stronger, published alternative.
- Gupta2025multimodalsurvey: replaces an earlier arXiv-only, not-yet-published
  Gao/Nayak/Coler review (that one was explicitly "submitted to IEEE Trans.
  Affective Computing," still unpublished as of this research) — ICICC 2025 is the
  published alternative covering the same narrative point.
