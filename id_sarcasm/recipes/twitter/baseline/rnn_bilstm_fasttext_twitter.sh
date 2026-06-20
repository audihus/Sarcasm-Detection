# Baseline RNN Bi-LSTM + pretrained fastText cc.id.300 (300-dim) — Twitter IdSarcasm.
# Apple-to-apple dgn run_classification.py: split sama, max_seq_length 128, F1 binary
# kelas positif, pemilihan model di val, early stopping patience 3, test sekali di akhir.
# DEVIASI (RNN from scratch): Adam lr 1e-3 (bukan 1e-5), tanpa cosine/fp16, whitespace tokenisasi.
#
# UNDUH fastText Indonesia dulu (salah satu):
#   .bin (OOV via subword, ~7GB): https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz
#   .vec (OOV=nol, ~4GB)        : https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz
# lalu set FASTTEXT_PATH ke file hasil ekstrak (default: embeddings/cc.id.300.bin).
FASTTEXT_PATH="${FASTTEXT_PATH:-embeddings/cc.id.300.bin}"

# Ekuivalen HF: ganti 3 baris --*_file dengan: --use_hf_dataset --text_column_name tweet
# Catatan: embedding fastText saat ini DI-FINE-TUNE (script tak punya flag freeze).
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
    --learning_rate 1e-3 \
    --batch_size 32 \
    --num_train_epochs 100 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.01 \
    --metric_for_best_model f1 \
    --seeds 42,1,2 \
    --output_dir outputs/rnn-bilstm-fasttext-twitter
