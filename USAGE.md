# Panduan Penggunaan — Multi-Channel Sarcasm Detection

## Prasyarat

- Conda environment `id_sarcasm` sudah aktif
- Dataset Arrow tersedia di `data/reddit_indonesia_sarcastic/` dan `data/twitter_indonesia_sarcastic/`
- Lexikon InSet tersedia di `data/lexicon/positive.tsv` dan `data/lexicon/negative.tsv`

---

## Step 1 — Siapkan data (jalankan sekali)

Konversi dataset Arrow ke CSV dan generate dependency/sentic graph.
Proses ini membutuhkan Stanza Indonesian (~500MB, diunduh otomatis pada run pertama).

```bash
python prepare_real_data.py --dataset reddit  --output_dir ./real_data/reddit
python prepare_real_data.py --dataset twitter --output_dir ./real_data/twitter
```

Output yang dihasilkan per split (train / validation / test):
- `{split}.csv` — teks + label
- `{split}.csv.graph.new` — dependency adjacency matrix
- `{split}.csv.sentic` — sentiment adjacency matrix

Opsi tambahan:
```bash
--skip_graphs        # skip generate graph jika file sudah ada
--csv_only           # hanya export CSV, skip generate graph
--splits train,test  # pilih split tertentu saja
```

---

## Step 2 — Training

### Reddit
```bash
python train_multichannel.py \
  --train_data       "E:/Luaran Kelulusan/Sarcasm Detection/Code/multi_channel_method/real_data/reddit/train.csv" \
  --val_data         "E:/Luaran Kelulusan/Sarcasm Detection/Code/multi_channel_method/real_data/reddit/validation.csv" \
  --test_data        "E:/Luaran Kelulusan/Sarcasm Detection/Code/multi_channel_method/real_data/reddit/test.csv" \
  --graph_dir        "E:/Luaran Kelulusan/Sarcasm Detection/Code/multi_channel_method/real_data/reddit" \
  --train_split_name train.csv \
  --val_split_name   validation.csv \
  --test_split_name  test.csv \
  --checkpoint_dir   ./ckpt_reddit \
  --dataset_name     reddit
```

### Twitter
```bash
python train_multichannel.py \
  --train_data       "E:/Luaran Kelulusan/Sarcasm Detection/Code/multi_channel_method/real_data/twitter/train.csv" \
  --val_data         "E:/Luaran Kelulusan/Sarcasm Detection/Code/multi_channel_method/real_data/twitter/validation.csv" \
  --test_data        "E:/Luaran Kelulusan/Sarcasm Detection/Code/multi_channel_method/real_data/twitter/test.csv" \
  --graph_dir        "E:/Luaran Kelulusan/Sarcasm Detection/Code/multi_channel_method/real_data/twitter" \
  --train_split_name train.csv \
  --val_split_name   validation.csv \
  --test_split_name  test.csv \
  --checkpoint_dir   ./ckpt_twitter \
  --dataset_name     twitter
```

Hyperparameter default sudah disesuaikan dengan id_sarcasm:
`--epochs 100`, `--batch_size 32`, `--weight_decay 0.03`, `--seed 42`, early stopping patience=3.

### Quick test (1 epoch)
Tambah `--epochs 1` ke command di atas untuk verifikasi pipeline tanpa training penuh.

---

## Step 3 — Resume training dari checkpoint

Tambah `--resume` ke command training:

```bash
python train_multichannel.py \
  ... \
  --checkpoint_dir ./ckpt_twitter \
  --resume
```

---

## Catatan Penting

- `--dataset_name` digunakan sebagai nama file cache embedding (`300_{dataset_name}_embedding_matrix.pkl`).
  Pastikan berbeda untuk tiap dataset agar cache tidak tercampur.
- Jika muncul error size mismatch pada embedding, hapus file `.pkl` di direktori kerja lalu jalankan ulang.
- Log training tersimpan di `log/` dan TensorBoard summary di `summary/`.
- Checkpoint terbaik (berdasarkan Val F1-Macro) tersimpan di `--checkpoint_dir`.
