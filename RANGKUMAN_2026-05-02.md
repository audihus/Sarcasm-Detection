# Rangkuman Sesi 2026-05-02 + Plan 2026-05-03

## Yang sudah dikerjakan hari ini

### 1. Audit bug ablation study — 4 cek lengkap

| Cek | Status | Temuan |
|---|---|---|
| Isolasi channel antar variant | ✓ AMAN | Tidak ada kebocoran sinyal; `bert_only`/`adgcn_only` betul-betul terpisah |
| Sumber majority class collapse | ⚠ **BUG** | Focal Loss `alpha=0.25` mendownweight minority class (kebalikan dari yang seharusnya) |
| Konsistensi evaluasi | ✓ AMAN | AUC dari softmax probability (valid), threshold konsisten, test set sama |
| Arsitektur variant | ✓ AMAN | Concat 768+512=1280 benar, dimensi dense layer cocok |

### 2. Perbaikan kode (4 file)

| File | Perubahan |
|---|---|
| `multichannel-sarcasm-detection/bridgeModel.py:132` | `alpha=0.25` → `alpha=0.75` |
| `train_ablation.py` | Alpha 0.75; patience default 3→8; +2 variant baru (`gcn_dep_only`, `gcn_sentic_only`); reset seed per-variant; `NO_BERT_VARIANTS` set |
| `train_multichannel.py:729` | Patience default 3→8 |
| `colab_train_ablation.ipynb` | Header 7 variant; Cell 4 mkdir update; +4 cell baru (Reddit & Twitter `gcn_*_only`) |

Patch `id_sarcasm/scripts/run_classification.py` (untuk Colab compatibility):
- Line 50: `send_example_telemetry` di-stub via try/except
- Line 691: `classes` di-wrap `np.array()`
- Line 698: `compute_loss` terima `**kwargs`

### 3. Verifikasi metric convention

- Paper IdSarcasm pakai **F1-binary** (rumus `2PR/(P+R)` untuk kelas positif), bukan F1-macro
- Codebase `id_sarcasm/scripts/run_classification.py:665` confirmed pakai F1-binary (HuggingFace `f1.compute()` default `average='binary'`)
- Multi-channel Anda harus dilapor pakai F1-binary di kolom `f1_binary` CSV

### 4. Eksperimen yang sudah dijalankan

**`train_multichannel.py` di Reddit (setelah focal loss fix):**
```
F1-binary  : 0.487
Recall     : 0.507
Precision  : 0.469
Accuracy   : 0.733
```

**Baseline reproduksi `id_sarcasm/scripts/run_classification.py` IndoBERT-base di Reddit:**
```
F1 (TEST set, 2824 sampel) : 0.6135   ← match paper 0.6100 (deviasi 0.5%)
Accuracy                   : 0.7925
Precision                  : 0.5741
Recall                     : 0.6586
```

**Quirk script**: `run_classification.py:745` memanggil `trainer.evaluate(eval_dataset=predict_dataset)` — jadi blok `***** eval metrics *****` sebenarnya hitung di TEST set, bukan val set. Field `eval_samples=1411` di output menyesatkan (itu val size), tapi metric-nya beneran test. Snippet manual user juga keluar 0.6135 karena memang test set yang sama.

**Comparison real** (apples-to-apples test set):
```
Multi-channel TEST F1   : 0.487
IndoBERT-base TEST F1   : 0.6135
Gap                     : -0.123
```

Environment + dataset Anda valid. Gap multi-channel vs baseline real, hipotesis utama tetap: indobert-lite vs indobert-base mismatch.

## Temuan kunci yang belum dieksekusi

**Multi-channel pakai `indobert-lite-base-p1` (~12M params), paper baseline pakai `indobert-base-p1` (124M params).** 10× perbedaan kapasitas. Kemungkinan besar penyebab gap 0.487 vs 0.6135.

Hardcoded di:
- `multichannel-sarcasm-detection/Model.py:33-35` (encoder)
- `multichannel-sarcasm-detection/bridgeModel.py:119-122` (tokenizer)
- `train_ablation.py:211-213` (tokenizer di AblationBridgeModel)

## Plan besok (2026-05-03)

### Step 0 — DONE (2026-05-02 malam)

- IndoBERT-base TEST F1 Reddit = **0.6135** (match paper 0.6100, deviasi 0.5%) ✓
- Quirk `run_classification.py:745` confirmed: trainer.evaluate() dipanggil dengan predict_dataset, jadi `eval_*` metrics sebenarnya test metrics.

### Step 1 — Ganti BERT ke indobert-base-p1 (15 menit)

Edit 3 file di atas, ganti string `indobert-lite-base-p1` → `indobert-base-p1`. (Bisa minta Claude lakukan edit-nya.)

### Step 2 — Bersihkan artefak training lama (5 menit)

Di Drive Colab atau lokal:
```bash
rm -rf checkpoints/
rm -f 300_*_embedding_matrix.pkl  # cache embedding lama
rm -f ablation_results_*.csv      # CSV hasil lama
```

Alasan: dimensi BERT berubah, checkpoint lama tidak compatible. Embedding cache mungkin tidak compatible kalau vocab tokenizer berubah.

### Step 3 — Re-upload file ke Drive (10 menit)

Upload ulang 3 file yang diedit ke Drive Colab.

### Step 4 — Re-run `train_multichannel.py` di Reddit (60-90 menit)

Pakai notebook `colab_train_multichannel.ipynb`. BERT-base 10× lebih besar dari lite → training 2-3× lebih lama. Kalau training awal 30 menit, sekarang ~60-90 menit per dataset.

**Target hasil**: F1-binary minimal 0.55-0.60+ (mendekati IndoBERT-base baseline 0.6135). Kalau ADGCN memberi gain valid, di atas itu.

### Step 5 — Run baseline IndoBERT-base di Twitter (parallel cell, 30-50 menit)

Sambil multi-channel Reddit running, di cell lain:
```bash
!python scripts/run_classification.py \
    --model_name_or_path indobenchmark/indobert-base-p1 \
    --dataset_name w11wo/twitter_indonesia_sarcastic \
    --text_column_name tweet --label_column_name label \
    [...sama dengan recipe Reddit, ganti reddit→twitter, text→tweet...]
```

**Target hasil**: F1 ≈ 0.7273 (paper Twitter baseline IndoBERT-base).

### Step 6 — Decision point

Setelah Step 4 selesai, evaluasi:

**Skenario A: Multi-channel Reddit F1-binary > 0.6135**
→ ADGCN memang memberi gain. Lanjut ablation 7 variant penuh untuk memvalidasi kontribusi tiap channel. Run multi-channel di Twitter.

**Skenario B: Multi-channel Reddit F1-binary ≈ 0.6135 (±0.02)**
→ ADGCN tidak menambah/mengurangi (neutral). Paper claim perlu di-reframe — mungkin tentang efisiensi atau interpretability, bukan accuracy.

**Skenario C: Multi-channel Reddit F1-binary < 0.6135 - 0.05**
→ ADGCN menghambat performa. Investigate lebih dalam:
- Audit ulang ADGCN forward (mungkin ada bug saya belum lihat)
- Tune learning rate ADGCN/dense
- Tune jumlah GCN layer
- Tune dropout
- Atau accept dan reframe paper

## Catatan tambahan

1. **Urutan eksekusi optimal**: jalankan step 4 dan step 5 paralel di dua tab Colab terpisah supaya tidak buang waktu.

2. **Multi-seed eksperimen**: belum dijalankan. Setelah skenario di step 6 jelas, baru investasi multi-seed (3-5 seed × N variant). Hari ini focus reproduce satu seed dulu.

3. **CSV checkpoint**: pastikan setiap run punya output CSV terpisah supaya bisa direkonstruksi kalau ada masalah.

4. **Backup checkpoints/ablation/ lama** kalau Anda mau bandingkan post-mortem (multi-channel lite vs base) sebelum dihapus. Boleh skip kalau tidak perlu.

## File yang perlu di-upload ulang ke Drive besok

Setelah edit BERT model name:

| File | Path di Drive |
|---|---|
| `Model.py` | `multi_channel_method/multichannel-sarcasm-detection/Model.py` |
| `bridgeModel.py` | `multi_channel_method/multichannel-sarcasm-detection/bridgeModel.py` |
| `train_ablation.py` | `multi_channel_method/train_ablation.py` |

(File `run_classification.py` yang sudah dipatch hari ini juga harus terus ada di Drive.)

---

**Selamat istirahat. Besok kita lanjut dengan kepala dingin.**
