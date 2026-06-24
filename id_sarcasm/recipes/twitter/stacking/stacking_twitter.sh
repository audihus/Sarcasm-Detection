#!/usr/bin/env bash
# Classical stacking/ensemble study (lexicon-free) for Twitter.
# Views: word TF-IDF (1,2)+(1,3) + SentencePiece subword; strategies E0-E3.
# Run from the id_sarcasm/ directory:
#   bash recipes/twitter/stacking/stacking_twitter.sh
set -e

python scripts/run_stacking_classification.py \
    --dataset_name ../data/twitter_indonesia_sarcastic \
    --text_column_name tweet \
    --output_folder results \
    --seed 42

# Online alternative (HuggingFace Hub):
# python scripts/run_stacking_classification.py \
#     --dataset_name w11wo/twitter_indonesia_sarcastic \
#     --text_column_name tweet --output_folder results --seed 42
#
# Without the subword view:  add  --no_subword
