# Baseline RNN Bi-GRU + random embedding (trainable from scratch) — Twitter IdSarcasm.
# Apple-to-apple dgn run_classification.py: split sama, max_seq_length 128, F1 binary
# kelas positif, pemilihan model di val, early stopping patience 3, test sekali di akhir.
# DEVIASI (RNN from scratch): Adam lr 1e-3 (bukan 1e-5), tanpa cosine/fp16, whitespace tokenisasi.
# Ekuivalen HF: ganti 3 baris --*_file dengan: --use_hf_dataset --text_column_name tweet
python scripts/run_rnn_classification.py \
    --model_type bigru \
    --embedding random \
    --train_file real_data/twitter/train.csv \
    --validation_file real_data/twitter/validation.csv \
    --test_file real_data/twitter/test.csv \
    --text_column_name content \
    --label_column_name label \
    --max_seq_length 128 \
    --embedding_dim 300 \
    --hidden_size 128 \
    --num_layers 1 \
    --pooling max \
    --dropout 0.3 \
    --learning_rate 1e-3 \
    --batch_size 32 \
    --num_train_epochs 100 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.01 \
    --metric_for_best_model f1 \
    --seeds 42,1,2 \
    --output_dir outputs/rnn-bigru-random-twitter
