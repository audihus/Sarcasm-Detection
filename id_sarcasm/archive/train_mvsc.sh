# MVSC lexicon-free classical method (final). Run from the id_sarcasm/ directory.
# Mirrors train_classical.sh but uses the improved, honest pipeline.
python scripts/run_mvsc_classification.py --dataset_name ../data/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results --seed 42
