# Config TERBAIK sejauh ini (Bi-LSTM + fastText cc.id.300.bin + self-attention + focal loss),
# dijalankan MULTI-SEED utk dapat angka robust (mean +/- std), bukan satu seed beruntung.
# Seed 42 tunggal sebelumnya: TEST@0.50 F1 0.7108 (val 0.7402). Multi-seed = angka jujur.
#
# Apple-to-apple invariants tetap: split 1878/268/538, max_seq_length 128, F1 binary kelas
# positif, pemilihan model di val, early stopping, test sekali. Sisi MODEL (focal/attention/
# scheduler/hyperparam) = variabel bebas; deskripsikan apa adanya di tesis (bukan 'vanilla BiLSTM').
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
    --hidden_size 32 \
    --num_layers 1 \
    --dropout 0.4 \
    --gamma 2.0 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_train_epochs 100 \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.01 \
    --metric_for_best_model f1 \
    --seeds 42,1,2 \
    --output_dir outputs/rnn-bilstm-ft-tuned-multiseed
