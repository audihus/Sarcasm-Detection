#!/usr/bin/env bash
# Ablation matrix rows 1-4 dari model.md
# Jalankan dari root id_sarcasm/:
#   bash recipes/twitter/marker_aux/run_ablation.sh

set -e

TRAIN=real_data/twitter/train.csv
VAL=real_data/twitter/validation.csv
TEST=real_data/twitter/test.csv
INSET_POS=real_data/twitter/positive.tsv
INSET_NEG=real_data/twitter/negative.tsv
SEEDS="42,1,2"

BASE="python scripts/run_sarcasm_marker_aux.py \
    --train_file $TRAIN --val_file $VAL --test_file $TEST \
    --seeds $SEEDS --fp16"

echo "=== Row 1: Vanilla baseline (mentah, CLS, lambda=0) ==="
$BASE \
    --use_raw_input --pooling cls --lam_aux 0 \
    --output_dir outputs/twitter-row1-vanilla

echo "=== Row 2: Marker-aware, CLS, lambda=0 ==="
$BASE \
    --pooling cls --lam_aux 0 \
    --output_dir outputs/twitter-row2-markers

echo "=== Row 3: Marker-aware, CLS, lambda=0.05 ==="
$BASE \
    --pooling cls --lam_aux 0.05 \
    --inset_pos_path $INSET_POS --inset_neg_path $INSET_NEG \
    --output_dir outputs/twitter-row3-aux-lam005

echo "=== Row 3: Marker-aware, CLS, lambda=0.1 ==="
$BASE \
    --pooling cls --lam_aux 0.1 \
    --inset_pos_path $INSET_POS --inset_neg_path $INSET_NEG \
    --output_dir outputs/twitter-row3-aux-lam010

echo "=== Row 3: Marker-aware, CLS, lambda=0.2 ==="
$BASE \
    --pooling cls --lam_aux 0.2 \
    --inset_pos_path $INSET_POS --inset_neg_path $INSET_NEG \
    --output_dir outputs/twitter-row3-aux-lam020

echo "=== Row 4: Ablation pooling (markers, mean-pool, lambda=0) ==="
$BASE \
    --pooling mean --lam_aux 0 \
    --output_dir outputs/twitter-row4-meanpool

echo ""
echo "=== SELESAI ==="
echo "Threshold tuning (contoh row 3 lam=0.1):"
echo "  python 'scripts/modifiied script/threshold_tuning.py' \\"
echo "      --output_dir outputs/twitter-row3-aux-lam010 --seeds $SEEDS"
