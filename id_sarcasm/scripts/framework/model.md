# Spec Nomor 2: Perakitan Model (IndoBERT dua-head)

Dokumen ini spesifikasi untuk diimplementasikan (mis. oleh Claude Code). Semua
keputusan di sini sudah dikunci dari analisis sebelumnya; jangan diubah tanpa alasan.

## 0. Konteks singkat
Framework deteksi sarkasme Indonesia di Twitter (dataset IdSarcasm). Kontribusi:
IndoBERT + marker-aware input, plus auxiliary task incongruity (opsional, bobot kecil).
Hasil negatif (ADGCN, interaction head) masuk sebagai ablation, bukan dibuang.

## 1. Data & input
- File: `train.csv`, `valid.csv`, `test.csv`. Kolom: `content`, `label` (0/1).
- Input encoder BUKAN `content` mentah. Pakai `sarcasm_preprocess.to_encoder_text(content)`
  untuk SEMUA split (transform stateless, aman seragam, bukan sumber leakage).
- Label aux `incongruity`: dari `incongruity.py` dengan `STRONG=2, WINDOW=None` (versi longgar,
  phi tertinggi ~0.13). Hitung untuk train (dan valid kalau mau pantau aux loss). Test tidak perlu.
- Imbalance: 1408 non-sarkas vs 470 sarkas (~75/25). Wajib ditangani (lihat 4).

## 2. Tokenizer (KRITIS, sering jadi bug)
```python
from transformers import AutoTokenizer, AutoModel
from sarcasm_preprocess import SPECIAL_TOKENS

tok = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
# SPECIAL_TOKENS = <username> <link> <hashtag> [CAPS] [ELONG] [REPPUNC] [EMOTICON]
# setelah model dibuat: model.encoder.resize_token_embeddings(len(tok))
```
Tanpa resize_token_embeddings, marker dipecah jadi subword dan sinyalnya hilang.

## 3. Model
- Backbone: `indobenchmark/indobert-base-p1` (sama dengan baseline 0.7306/0.7509).
- Pooling: `[CLS]` (pooler_output) untuk hasil utama; mean-pooling (mask padding) sebagai ablation.
- Main head: `Linear(hidden, 2)` di atas representasi pooled.
- Aux head: `Linear(hidden, 2)` di atas representasi pooled YANG SAMA (prediksi clash 0/1).
- Encoder dibagi kedua head (itu inti idenya: aux membentuk encoder bersama).

```python
class SarcasmModel(nn.Module):
    def __init__(self, name, n_tokens):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.enc.resize_token_embeddings(n_tokens)
        h = self.enc.config.hidden_size
        self.main = nn.Linear(h, 2)
        self.aux  = nn.Linear(h, 2)
    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output                      # ganti mean-pool untuk ablation
        return self.main(pooled), self.aux(pooled)
```

## 4. Objective
- `L = L_main + lambda * L_aux`, dua-duanya cross-entropy.
- `L_main`: class-weighted CE. Bobot ~ invers frekuensi kelas, sekitar `[1.0, 3.0]`.
- `L_aux`: CE biasa (clash longgar ~50/50, tidak perlu bobot berat).
- Sweep `lambda` in `{0, 0.05, 0.1, 0.2}`. **`lambda=0` = ablation tanpa-aux.**
- Aux loss cuma dihitung kalau label aux tersedia (train; valid opsional). Test: hanya main head.

## 5. Hyperparameter (default wajar)
- `max_seq_len = 128` (tweet pendek; median 15, max 69 token).
- batch 16 atau 32, LR `2e-5`, AdamW, warmup 10%, weight_decay 0.01.
- epoch maks 5, early stopping di val F1 (patience 2).
- **Seed: minimal 3 run, laporkan mean ± std.** Data kecil + timpang itu berisik; satu run menyesatkan.

## 6. Evaluasi
- Metrik utama: **F1 kelas sarkas (positif), bukan macro.** Laporkan juga precision/recall;
  macro-F1 sebagai sekunder. (Ini menghindari jebakan macro-vs-binary yang dulu kejadian.)
- Pemilihan model + tuning threshold dilakukan di valid; test disentuh sekali di akhir.
- Karena imbalance, threshold 0.5 mungkin bukan optimal; tuning threshold di val untuk maksimalkan F1 positif.

## 7. Baseline (titik banding, harus direproduksi)
| Konfigurasi | Target F1 |
|---|---|
| IndoBERT vanilla (content mentah, tanpa marker, lambda=0) | ~0.7306 |
| IndoBERT + marker-aware (tanpa aux, lambda=0) | ~0.7509 |

## 8. Ablation matrix (tulang punggung paper)
Tiap baris dijalankan lintas seed, laporkan F1 positif mean ± std.

| # | Input | Pooling | lambda | Catatan |
|---|---|---|---|---|
| 1 | mentah | CLS | 0 | baseline vanilla |
| 2 | + markers | CLS | 0 | = konfigurasi 0.7509 |
| 3 | + markers | CLS | sweep | proposed (full) |
| 4 | + markers | mean | 0 | ablation pooling |
| 5 | + markers + ADGCN channel | CLS | 0 | hasil negatif (penguat tesis) |
| 6 | + markers + interaction head | CLS | 0 | hasil negatif (penguat tesis) |

Tambahan: catat efek normalisasi on/off pada KUALITAS LABEL AUX saja (tidak menyentuh input encoder).

## 9. Guardrail
- JANGAN normalisasi input encoder; marker harus selamat (sudah dijaga `to_encoder_text`).
- Kebocoran tag sarkasme sudah gugur (hashtag ter-mask `<hashtag>`); JANGAN tambah fitur hashtag sarkasme.
- Aux diperkirakan lemah (phi ~0.1). Hasil aux null/negatif itu OUTCOME ablation yang sah, bukan bug.
- Pakai `valid.csv` untuk semua keputusan; `test.csv` hanya untuk angka final.