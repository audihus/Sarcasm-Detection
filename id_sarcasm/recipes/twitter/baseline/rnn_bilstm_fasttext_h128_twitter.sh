# Bi-LSTM + fastText cc.id.300.bin + hidden_size=128 — Twitter IdSarcasm.
# Motivasi: baseline hidden=32 menghasilkan 64-dim context vector (terlalu kecil).
# hidden=128 -> 256-dim context, kapasitas RNN lebih besar untuk tangkap pola sarkasme.
# Dropout diturunkan 0.4->0.3 karena model tidak terbukti overfit secara berbahaya.
# Baseline (hidden=32, unfrozen, 3 seed): F1=0.710 (P=0.659, R=0.771). Target: F1>=0.76.
FASTTEXT_PATH="${FASTTEXT_PATH:-embeddings/cc.id.300.bin}"

python scripts/run_rnn_classification.py \
    --model_type bilstm \
    --embedding fasttext \
    --fasttext_path "${FASTTEXT_PATH}" \
    --train_file real_data/twitter/train.csv \
    --validation_file real_data/twitter/validation.csv \
    --test_file real_data/twitter/test.csv \
    --text_column_name content \
    --label_column_name label \
    --max_seq_length 128 \
    --hidden_size 128 \
    --num_layers 1 \
    --dropout 0.3 \
    --gamma 2.0 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_train_epochs 100 \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.01 \
    --metric_for_best_model f1 \
    --seeds 42,1,2 \
    --output_dir outputs/rnn-bilstm-ft-h128-twitter
