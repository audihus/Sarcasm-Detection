# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**IdSarcasm** is a benchmarking research project (IEEE Access 2024, DOI: 10.1109/ACCESS.2024.3416955) evaluating three classes of models for Indonesian sarcasm detection: classical ML, fine-tuned transformers, and zero-shot LLMs.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies include: `scikit-learn`, `transformers`, `datasets`, `evaluate`, `accelerate`, `bitsandbytes`, `nltk`.

## Running Experiments

### Classical ML (Logistic Regression, Naive Bayes, SVM)
```bash
bash train_classical.sh
# or directly:
python scripts/run_classical_classification.py \
    --dataset_name w11wo/reddit_indonesia_sarcastic \
    --text_column_name text \
    --output_folder results
```

### Fine-tuned Transformers (IndoBERT, mBERT, XLM-R)
```bash
bash train_reddit.sh     # Reddit dataset, all 18 model-recipe combos
bash train_twitter.sh    # Twitter dataset, all 18 model-recipe combos
# or run a single recipe:
bash recipes/reddit/baseline/xlmr_large_reddit.sh
```

### Zero-shot LLMs (BLOOMZ, mT0)
```bash
bash zero_shot_classification.sh
```

## Architecture

Three independent pipelines share the same two datasets:

```
Datasets (HuggingFace Hub)
  ├── w11wo/reddit_indonesia_sarcastic  (14.1K, text column)
  └── w11wo/twitter_indonesia_sarcastic (12.9K, tweet column)
      │
      ├── scripts/run_classical_classification.py
      │     GridSearchCV over LR/NB/SVM × BoW/TF-IDF → results/classical/
      │
      ├── scripts/run_classification.py
      │     HuggingFace Trainer fine-tuning (128 tok, bs=32, lr=1e-5, cosine)
      │     Variants in recipes/{reddit,twitter}/{baseline,augment,weighted,multidataset}/
      │     augment: adds w11wo/isarcasm_id sarcastic samples to training
      │     weighted: class-weighted cross-entropy for imbalanced data
      │     → checkpoints + eval_results_*.json + predictions.txt
      │
      └── scripts/run_zero_shot_classification.py
            Log-prob over 5 prompt templates, 8-bit quantized inference
            → results/{bloomz-*,mt0-*}/eval_results_*.json
```

## Key Implementation Details

**run_classification.py**: Early stopping (patience=3, threshold=0.01) on F1; best model saved by F1; fp16 mixed precision; eval + save every epoch; seed=42.

**run_zero_shot_classification.py**: Classification is log-probability based (not generation). Averages metrics across 5 fixed prompt templates. Supports both encoder-decoder (mT0) and decoder-only (BLOOMZ) with automatic device mapping.

**Recipes structure**: `recipes/{dataset}/{variant}/{model_name}.sh` — each `.sh` calls `run_classification.py` with model-specific HuggingFace hub IDs and hyperparameters.

**Results format**: All outputs are `eval_results_<dataset_name>.json` containing accuracy, precision, recall, and F1-score.

## Models Used

| Category | Models |
|----------|--------|
| Classical | Logistic Regression, Naive Bayes, SVM |
| Fine-tuned | `indobenchmark/indobert-{base,large}-p1`, `indobenchmark/indobert-base-uncased`, `bert-base-multilingual-cased`, `xlm-roberta-{base,large}` |
| Zero-shot | `bigscience/bloomz-{560m,1b1,1b7,3b,7b1}`, `bigscience/mt0-{small,base,large,xl}` |

## Datasets

Both datasets are publicly available on HuggingFace Hub and loaded automatically at runtime. Local copies in `data/` mirror the hub versions. Reddit data uses `text` column; Twitter uses `tweet` column.
