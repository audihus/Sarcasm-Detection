#!/usr/bin/env bash
# MVSC — lexicon-free classical baseline (Logistic Regression) for Twitter.
# Word TF-IDF(1,2)+sublinear + class_weight=balanced + F1 model-selection + threshold tuning.
# Run from the id_sarcasm/ directory:
#   bash recipes/twitter/mvsc/mvsc_lr_twitter.sh
set -e

# Local offline copy (data/ lives at the repo root, one level above id_sarcasm/):
python scripts/run_mvsc_classification.py \
    --dataset_name ../data/twitter_indonesia_sarcastic \
    --text_column_name tweet \
    --output_folder results \
    --seed 42

# Online alternative (HuggingFace Hub — needs internet):
# python scripts/run_mvsc_classification.py \
#     --dataset_name w11wo/twitter_indonesia_sarcastic \
#     --text_column_name tweet \
#     --output_folder results \
#     --seed 42
