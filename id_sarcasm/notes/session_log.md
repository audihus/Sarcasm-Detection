# Session Log — Enriched Preprocessing Twitter Sarcasm

## Apa yang Sudah Dilakukan

---

### 1. Audit `sarcasm_preprocess.py` — Dua Jalur Preprocessing

File: `scripts/framework/sarcasm_preprocess.py`

Dua jalur yang sengaja berlawanan:

| Jalur | Fungsi | Output | Tujuan |
|-------|--------|--------|--------|
| Encoder | `to_encoder_text` | `str` | Input ke IndoBERT/mBERT/XLM-R |
| Leksikon | `to_lexicon_tokens` | `list[str]` | Lookup kamus InSet (incongruity) |

**Jalur encoder** mempertahankan noise sebagai marker eksplisit.  
**Jalur leksikon** membersihkan teks sampai kata-katanya cocok dengan entri kamus.

---

### 2. EDA — Pattern Frequency (1878 training samples)

File: `scripts/eda_twitter.py`

**Temuan kunci:**

| Pattern | Count | %Sarc | %Non | Catatan |
|---------|-------|-------|------|---------|
| Tanda baca berulang (!!!/???) | 371 (19.8%) | **28.1%** | 17.0% | Paling diskriminatif untuk sarkasme |
| Slang/singkatan | 485 (25.8%) | 13.0% | **30.1%** | Non-sarc lebih banyak pakai slang |
| Negasi informal (gak/ga) | 257 (13.7%) | **2.1%** | **17.5%** | Non-sarc jauh lebih banyak negasi |
| ALL_CAPS | 388 (20.7%) | 20.6% | 20.7% | Tidak diskriminatif |
| Reduplikasi (kata2) | 113 (6.0%) | 8.1% | 5.3% | Lemah, tapi ada |
| Emoticon | 78 (4.2%) | 7.7% | 3.0% | Sedikit diskriminatif |

**Insight penting:** Negasi dan slang lebih sering muncul di tweet **non-sarkastik**. Menormalisasinya di encoder berarti menghapus sinyal distribusional yang berguna.

---

### 3. Bottom-up Slang Discovery

File: `scripts/eda_slang_discovery.py`

Analisis bottom-up dari semua token training (min freq=3, tanpa KBBI). Menemukan kandidat slang yang belum ada di daftar.

**Yang layak ditambahkan ke SLANG:**

| Kategori | Token baru |
|----------|-----------|
| Singkatan baku | `sdh`, `msh`, `skrg`, `sy`, `kl`, `sbg`, `kpd`, `pd`, `bs`, `mrk`, `dlm`, `thn`, `trus`, `jdi`, `emg`, `dll` |
| Kata ganti informal | `gue`, `gw`, `lu`, `lo` |
| Informal lainnya | `tau`, `pake`, `ngerti` |

**Yang sengaja TIDAK dinormalisasi:**
- Partikel diskursus: `kok`, `sih`, `nih`, `tuh`, `lah`, `deh`, `dong` — bukan singkatan, bisa jadi sinyal sarkasme.

---

### 4. Perubahan di `sarcasm_preprocess.py`

**a. Ekspansi NEGATORS** — tambah 2 varian:
```python
"ngak": "tidak",   # freq 9
"ndak": "tidak",   # freq 5
```

**b. Ekspansi SLANG** — tambah 23 entri baru (dari hasil EDA di atas)

**c. Tambah `_surface_normalize_nomarkers()`** — helper untuk versi tanpa marker:
- Elongasi diciutkan tapi tanpa `[ELONG]`
- CAPS dilowercased tapi tanpa `[CAPS]`
- Repeated punct dinormalisasi tapi tanpa `[REPPUNC]`

**d. Tambah `_normalize_encoder_tokens()`** — dipanggil di `to_encoder_text` setelah `surface_markers`:
- **Sekarang hanya normalisasi reduplikasi** (`kata2` → `kata`)
- Negasi & slang **tidak** dinormalisasi di encoder (lihat bagian Hasil Eksperimen)

**e. Tambah `to_encoder_text_nomarkers()`** — versi tanpa surface markers:
```
repair_encoding → emoticon dibuang → emoji→alias → surface_normalize (tanpa marker)
→ _normalize_encoder_tokens → collapse_placeholders → tidy_whitespace
```

**Pipeline encoder (dengan markers):**
```
repair_encoding → emoticon→[EMOTICON] → emoji→alias → surface_markers (CAPS/ELONG/REPPUNC→marker)
→ _normalize_encoder_tokens (hanya redup) → collapse_placeholders → tidy_whitespace
```

---

### 5. Perubahan di `run_classification.py`

Tambah flag `--add_surface_markers` di `DataTrainingArguments`:

```python
add_surface_markers: bool = field(
    default=False,
    metadata={"help": "Register surface marker tokens as special tokens."},
)
```

Dan implementasinya setelah model dimuat:
```python
if data_args.add_surface_markers:
    _markers = ["<username>", "<link>", "<hashtag>",
                "[CAPS]", "[ELONG]", "[REPPUNC]", "[EMOTICON]"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": _markers})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
```

Cara pakai di Kaggle script: tambahkan `"--add_surface_markers"` ke `cmd` list.

---

### 6. Script Preprocessing Baru

| Script | Output | Keterangan |
|--------|--------|-----------|
| `do_preprocess_markers.py` | `twitter_ready/` | Ada markers + redup normalization |
| `do_preprocess_nomarkers.py` | `twitter_ready_nomarkers/` | Tanpa markers + redup normalization |

---

### 7. Folder Dataset Sekarang

```
preprocessed_data/
├── twitter_ready/            ← AKTIF: markers + redup normalization
├── twitter_ready_nomarkers/  ← AKTIF: tanpa markers + redup normalization
└── twitter_ready_old/        ← BACKUP: versi lama (git commit 69b04be)
```

---

### 8. Hasil Eksperimen & Pelajaran

Tiga kondisi yang sudah ditest di Kaggle:

| Kondisi | Best F1 | Catatan |
|---------|---------|---------|
| Baseline (paper reproduction) | 0.7449 | Raw HuggingFace data |
| Old preproc + unregistered tokens | **0.7609** | Sedikit di atas baseline |
| New preproc + registered tokens | 0.6667 | Drop signifikan |

**Kenapa new preproc + registered turun?**
1. Normalisasi `gak`→`tidak`, `lu`→`kamu` menghapus distributional signal yang diskriminatif
2. Registered token embeddings random + dataset kecil (1878) → noise
3. Kombinasi keduanya = drop ~0.09 F1

**Keputusan:** Hapus normalisasi negasi & slang dari jalur encoder. Hanya reduplikasi yang dinormalisasi di encoder. Negasi & slang tetap dinormalisasi di jalur **leksikon** saja.

---

### 9. Todo List Status

| # | Task | Status |
|---|------|--------|
| 1 | Fix `run_classification.py`: daftarkan SPECIAL_TOKENS ke tokenizer | ✅ Done |
| 2 | Ekspansi NEGATORS+SLANG + tambah `_normalize_encoder_tokens` ke `to_encoder_text` | ✅ Done |

---

### 10. Diskusi Novelty untuk Paper (iSemantic)

**iSemantic** = IEEE Xplore proceedings, Universitas Dian Nuswantoro, regional.

**Yang sudah gagal:**
- Multi-channel pipeline (BERT + ADGCN)
- InSet incongruity detection
- IndoBERTweet (turun dari baseline)

**Kandidat novelty yang masih feasible:**
1. **Enriched preprocessing** (sudah diimplementasi) — butuh hasil eksperimen yang positif
2. **Asymmetric Dice Loss** (Li et al., ACL 2020) — handles class imbalance (470 sarc vs 1408 non), berbeda dari weighted CE yang sudah ada
3. **Ensemble** — soft voting dari top-N model, low effort, hampir selalu improve

**Kombinasi yang direkomendasikan:**
> Enriched preprocessing (surface markers + redup) + Asymmetric Dice Loss

Dua angle kontribusi: representasi input (preprocessing) dan objective training (loss function).

---

### 11. Next Steps

- [ ] Upload `twitter_ready/` yang baru ke Kaggle, test dengan `--add_surface_markers`
- [ ] Test `twitter_ready_nomarkers/` tanpa flag (ablation: markers vs no markers)
- [ ] Implementasi Asymmetric Dice Loss di `run_classification.py`
- [ ] Bandingkan semua kondisi untuk paper
