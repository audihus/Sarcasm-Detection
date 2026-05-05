# Hyperparameter IndoBERT Base — Reproduksi Paper IdSarcasm

**Paper:** "Benchmarking Indonesian Sarcasm Detection" — IEEE Access 2024  
**DOI:** 10.1109/ACCESS.2024.3416955  
**Repo asli:** `id_sarcasm/` (w11wo)  
**Model yang direproduksi:** `indobenchmark/indobert-base-p1`

---

## Core Hyperparameters

| Parameter | Nilai |
|-----------|-------|
| `model_name_or_path` | `indobenchmark/indobert-base-p1` |
| `learning_rate` | `1e-5` |
| `lr_scheduler_type` | `cosine` |
| `weight_decay` | `0.03` |
| `per_device_train_batch_size` | `32` |
| `per_device_eval_batch_size` | `64` |
| `num_train_epochs` | `100` (efektif berhenti lebih awal karena early stopping) |
| `max_seq_length` | `128` |
| `label_smoothing_factor` | `0.0` |
| `seed` | `42` |
| `fp16` | `True` (mixed precision training) |

## Optimizer

- **Optimizer:** AdamW (default HuggingFace Trainer)
- **Warmup steps:** 0 (tidak diset, default Trainer)
- **Warmup ratio:** tidak diset (default 0.0)

## Early Stopping

| Parameter | Nilai |
|-----------|-------|
| `early_stopping_patience` | `3` |
| `early_stopping_threshold` | `0.01` |
| `metric_for_best_model` | `f1` |
| `load_best_model_at_end` | `True` |

## Evaluation & Checkpoint Strategy

| Parameter | Nilai |
|-----------|-------|
| `save_strategy` | `epoch` |
| `evaluation_strategy` | `epoch` |
| `logging_strategy` | `epoch` |
| `shuffle_train_dataset` | `True` |
| `shuffle_seed` | `42` |

## Tokenisasi & Padding

| Parameter | Nilai |
|-----------|-------|
| `pad_to_max_length` | `True` (semua sequence dipad ke 128) |
| `padding` | `max_length` |
| `truncation` | `True` |
| `use_fast_tokenizer` | `True` |

## Loss Function

- **Baseline & Augment:** `CrossEntropyLoss` tanpa bobot
- **Weighted & Augment+Weighted:** `CrossEntropyLoss` dengan class weight
  - Bobot dihitung dengan `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`
  - Bobot dikalikan `weight_multiplier = 2.0`

## Variant Eksperimen

| Variant | Flag Tambahan |
|---------|---------------|
| `baseline` | — |
| `augment` | `--do_augment` |
| `weighted` | `--do_weighted_loss --weight_multiplier 2.0` |
| `augment+weighted` | `--do_augment --do_weighted_loss --weight_multiplier 2.0` |

**Augment:** menambahkan sampel **sarcastic saja** dari dataset `w11wo/isarcasm_id` ke training set (concat ke split train).

## Dataset

| Dataset | HuggingFace Hub | `text_column_name` | `label_column_name` |
|---------|-----------------|-------------------|---------------------|
| Reddit | `w11wo/reddit_indonesia_sarcastic` | `text` | `label` |
| Twitter | `w11wo/twitter_indonesia_sarcastic` | `tweet` | `label` |

- `dataset_config_name`: `default`
- Label: binary (`0` = non-sarcastic, `1` = sarcastic)
- Metric utama: **F1** (binary, average default)

## Metrik yang Dihitung

- `accuracy`
- `f1`
- `precision`
- `recall`

Dihitung menggunakan library `evaluate` dari HuggingFace. Prediksi diambil dari `argmax(logits)`.

## Contoh Command Lengkap (Baseline Reddit)

```bash
python scripts/run_classification.py \
    --model_name_or_path indobenchmark/indobert-base-p1 \
    --dataset_name w11wo/reddit_indonesia_sarcastic \
    --dataset_config_name default \
    --shuffle_train_dataset \
    --metric_name f1 \
    --text_column_name text \
    --label_column_name label \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.03 \
    --label_smoothing_factor 0.0 \
    --num_train_epochs 100 \
    --do_train --do_eval --do_predict \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --seed 42 \
    --fp16
```

## Catatan Fairness untuk Paper

- Semua setting di atas harus **sama persis** di kode reproduksi agar perbandingan adil.
- Jangan ubah `seed`, `batch_size`, `learning_rate`, atau strategi early stopping.
- Pastikan menggunakan model ID yang sama: `indobenchmark/indobert-base-p1` (bukan `indobert-base-p2` atau varian lain).
- `fp16` digunakan di paper — aktifkan jika GPU mendukung, nonaktifkan hanya jika tidak ada GPU (bisa sedikit mempengaruhi hasil numerik).
- Evaluasi dilakukan pada **test set** (`do_predict`), bukan validation set.
