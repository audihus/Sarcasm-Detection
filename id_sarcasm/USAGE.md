# Panduan Penggunaan IdSarcasm

## Persiapan Awal

### 1. Buat dan aktifkan environment conda

```bash
conda create -n id_sarcasm python=3.10
conda activate id_sarcasm
```

### 2. Install dependencies

```bash
pip install transformers datasets evaluate accelerate bitsandbytes scikit-learn nltk
pip install git+https://github.com/boriswinner/LSH
pip install git+https://github.com/zafercavdar/fasttext-langdetect.git
```

> `bitsandbytes` hanya dibutuhkan untuk pipeline Zero-shot LLM. Bisa dilewati jika hanya ingin menjalankan Classical ML atau Fine-tuning.

### 3. Download dataset

Jalankan dari folder `id_sarcasm`:

```bash
python scripts/download_data.py
```

Dataset akan tersimpan ke:
- `data/reddit_indonesia_sarcastic/`
- `data/twitter_indonesia_sarcastic/`

---

## Menjalankan Eksperimen

Pastikan selalu berada di folder `id_sarcasm` sebelum menjalankan perintah apapun:

```bash
cd id_sarcasm
```

---

### Pipeline 1 — Classical ML

Model yang dilatih: Logistic Regression, Naive Bayes, SVM, dengan fitur BoW dan TF-IDF.

**Tidak membutuhkan GPU.**

```bash
# Dataset Reddit
python scripts/run_classical_classification.py \
    --dataset_name data/reddit_indonesia_sarcastic \
    --text_column_name text \
    --output_folder results

# Dataset Twitter
python scripts/run_classical_classification.py \
    --dataset_name data/twitter_indonesia_sarcastic \
    --text_column_name tweet \
    --output_folder results
```

Hasil tersimpan di `results/classical/eval_results_<nama_dataset>.json`.

---

### Pipeline 2 — Fine-tuning Transformer

Model yang didukung: IndoBERT (Base/Large), mBERT, XLM-R (Base/Large).

**Membutuhkan GPU.**

#### Cara cepat — jalankan recipe yang sudah ada

Untuk menjalankan satu model saja (disarankan untuk percobaan pertama):

```bash
# Contoh: XLM-R Large, dataset Reddit
bash recipes/reddit/baseline/xlmr_large_reddit.sh

# Contoh: XLM-R Large, dataset Twitter
bash recipes/twitter/baseline/xlmr_large_twitter.sh
```

Untuk menjalankan semua model sekaligus:

```bash
bash train_reddit.sh    # semua model, dataset Reddit
bash train_twitter.sh   # semua model, dataset Twitter
```

Daftar recipe yang tersedia di `recipes/{reddit,twitter}/{baseline,augment,weighted}/`:

| File | Model |
|------|-------|
| `indobert_indonlu_base_*.sh` | IndoBERT Base (IndoNLU) |
| `indobert_indonlu_large_*.sh` | IndoBERT Large (IndoNLU) |
| `indobert_indolem_base_*.sh` | IndoBERT Base (IndoLEM) |
| `mbert_base_*.sh` | mBERT Base |
| `xlmr_base_*.sh` | XLM-R Base |
| `xlmr_large_*.sh` | XLM-R Large |

Varian training:
- `baseline/` — training standar
- `augment/` — tambahan data dari dataset iSarcasm
- `weighted/` — class-weighted loss untuk data tidak seimbang

Hasil tersimpan di `outputs/<nama_model>/`.

> Secara default, script akan mencoba upload model ke HuggingFace Hub (`--push_to_hub`). Hapus atau komentari argumen tersebut di file `.sh` jika tidak ingin upload.

#### Cara manual — jalankan langsung dengan argumen kustom

```bash
python scripts/run_classification.py \
    --model_name_or_path xlm-roberta-base \
    --dataset_name data/reddit_indonesia_sarcastic \
    --text_column_name text \
    --label_column_name label \
    --output_dir outputs/my-experiment \
    --do_train --do_eval --do_predict \
    --num_train_epochs 100 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.03 \
    --max_seq_length 128 \
    --metric_name f1 \
    --shuffle_train_dataset \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --seed 42 \
    --fp16
```

#### Argumen penting `run_classification.py`

| Argumen | Keterangan |
|---------|-----------|
| `--model_name_or_path` | Nama model di HuggingFace Hub atau path lokal |
| `--dataset_name` | Nama dataset di Hub atau path lokal (hasil `save_to_disk`) |
| `--text_column_name` | Kolom teks: `text` (Reddit) atau `tweet` (Twitter) |
| `--label_column_name` | Kolom label, default: `label` |
| `--output_dir` | Folder untuk menyimpan checkpoint dan hasil |
| `--num_train_epochs` | Jumlah epoch maksimum (ada early stopping, biasanya berhenti lebih cepat) |
| `--per_device_train_batch_size` | Batch size training per GPU |
| `--learning_rate` | Learning rate optimizer |
| `--max_seq_length` | Panjang token maksimum (default: 128) |
| `--metric_name` | Metrik untuk memilih model terbaik, gunakan `f1` |
| `--do_augment` | Tambahkan data dari dataset iSarcasm ke training |
| `--do_weighted_loss` | Gunakan class-weighted loss (cocok untuk data tidak seimbang) |
| `--weight_multiplier` | Pengali bobot loss (default: 1.0) |
| `--fp16` | Aktifkan mixed precision (mempercepat training di GPU) |
| `--push_to_hub` | Upload model ke HuggingFace Hub setelah training |
| `--hub_model_id` | Nama repo di HuggingFace Hub untuk upload |
| `--max_train_samples` | Batasi jumlah data training (berguna untuk debugging) |

#### Early stopping

Script secara otomatis menghentikan training jika F1-score pada validation set tidak meningkat selama **3 epoch berturut-turut** (threshold: 0.01). Karena itu `--num_train_epochs 100` aman digunakan — training tidak akan benar-benar berjalan 100 epoch.

---

### Pipeline 3 — Zero-shot LLM

Model yang diuji: BLOOMZ (560M–7.1B), mT0 (Small–XL).

**Membutuhkan GPU. Untuk model ≥3B, butuh GPU dengan VRAM ≥16GB.**

```bash
# Satu model saja (disarankan untuk percobaan pertama)
python scripts/run_zero_shot_classification.py \
    --base_model bigscience/bloomz-560m \
    --dataset_name data/reddit_indonesia_sarcastic \
    --output_folder results

# Semua model sekaligus
bash zero_shot_classification.sh
```

Hasil tersimpan di `results/<nama_model>/eval_results_<nama_dataset>.json`.

---

## Format Hasil

Semua pipeline menghasilkan file JSON dengan metrik berikut:

```json
{
    "accuracy": 0.866,
    "f1": 0.714,
    "precision": 0.763,
    "recall": 0.672
}
```

Untuk Classical ML, metrik diberi prefix nama model dan fitur, contoh:
```json
{
    "lr_TFIDF_f1": 0.714,
    "svm_BoW_f1": 0.685,
    ...
}
```

---

## Ringkasan Perbandingan Hasil (F1-score)

| Model | Reddit | Twitter |
|-------|:------:|:-------:|
| Logistic Regression | 0.4887 | 0.7142 |
| Naive Bayes | 0.4591 | 0.6721 |
| SVM | 0.4467 | 0.6782 |
| IndoBERT Base (IndoNLU) | 0.6100 | 0.7273 |
| IndoBERT Large (IndoNLU) | 0.6184 | 0.7160 |
| XLM-R Base | 0.5690 | 0.7386 |
| **XLM-R Large** | **0.6274** | **0.7692** |
| BLOOMZ-560M | 0.3870 | 0.3916 |
| mT0 XL | 0.4001 | 0.3988 |
