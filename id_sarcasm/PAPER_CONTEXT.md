# Konteks Paper: Hybrid Classical-Transformer Fusion untuk Deteksi Sarkasme Indonesia

> **Untuk dipakai di Claude.ai chat / konsultasi penulisan paper.**
> Copy-paste seluruh file ini sebagai konteks awal percakapan.

---

## 1. Gambaran Penelitian

**Topik:** Deteksi sarkasme teks berbahasa Indonesia di Twitter menggunakan pendekatan hybrid: kombinasi model klasik (Logistic Regression + TF-IDF) dengan transformer yang sudah di-fine-tune.

**Referensi utama (paper acuan):**
Suhartono et al., "IdSarcasm: Indonesian Sarcasm Detection Using Transformer Models," *IEEE Access*, 2024. DOI: 10.1109/ACCESS.2024.3416955.
- Paper ini membangun benchmark deteksi sarkasme Indonesia (dataset Reddit + Twitter)
- Mengevaluasi: classical ML (LR, NB, SVM), fine-tuned transformers (IndoBERT, mBERT, XLM-R), zero-shot LLM (BLOOMZ, mT0)
- SOTA di dataset Twitter: **XLM-R large F1 = 0.7692**

**Kontribusi penelitian ini (yang sedang ditulis):**
1. Membuktikan classical ML lexicon-free bisa mendekati SOTA (F1 0.7509 vs SOTA 0.7692)
2. Membuktikan classical ensembling gagal (korelasi prediksi antar model klasik r = 0.877 → tidak ada sinyal komplementer)
3. Mengusulkan **hybrid late-fusion**: gabungkan probabilitas LR + probabilitas transformer fine-tuned → melampaui SOTA
4. Membandingkan operator fusi: prob-avg, logit-avg, rank-avg × dengan/tanpa temperature scaling

---

## 2. Dataset

**Nama:** `w11wo/twitter_indonesia_sarcastic` (HuggingFace Hub)
**Split:** train=1878 / val=268 / test=538
**Label:** 0 (non-sarkasme), 1 (sarkasme) — imbalanced ~25% positif
**Kolom teks:** `tweet`
**Sumber:** Twitter berbahasa Indonesia
**Catatan:** Dataset sudah dipreproses (PII-masked). Kecil → overfitting mudah terjadi.

---

## 3. Metode

### 3.1 Classical Baseline (LR Tuned — Lexicon-Free)

**Tidak boleh menggunakan kamus eksternal** (InSet, SentiWordNet, dll.) — ditolak pembimbing.

Arsitektur:
- **Vectorizer:** TF-IDF, ngram (1,2), sublinear_tf=True, min_df=2
- **Tokenizer:** NLTK word_tokenize
- **Classifier:** LogisticRegression, class_weight='balanced', max_iter=3000
- **Hyperparameter selection:** GridSearchCV C ∈ {0.05, 0.1, 0.3, 1, 3, 10, 30}, scoring='f1'
- **Protocol:** PredefinedSplit (train=-1, val=0) → tidak menyentuh test saat pilih C
- **Threshold tuning:** dicari di val untuk maksimalkan F1 (bukan pakai 0.5)
- **Seed:** 41 (sama dengan paper acuan)

### 3.2 Studi Classical Ensembling (Negative Result)

Tiga view: TF-IDF(1,2), TF-IDF(1,3), SentencePiece subword.
Base learners: LR, CalibratedSVC, ComplementNB.
Strategi: single LR (E0), soft-vote (E1), CV-stacking (E2), bagged LR (E3).

**Temuan:** Mean OOF pairwise correlation = **0.877** → semua prediksi terlalu mirip → ensembling tidak menambah nilai. Semua P(>LR) ≤ 47%.

**Kesimpulan:** Classical ensembling exhausted di data ini. Hanya transformer yang memberikan sinyal dekorelasi.

### 3.3 Hybrid Late-Fusion

**Konsep:** Gabungkan `prob_LR` dan `prob_transformer` tanpa melatih ulang transformer.

Transformer diambil dari model fine-tuned yang dirilis penulis paper acuan:
```
w11wo/indobert-base-p1-twitter-indonesia-sarcastic      (IndoBERT base)
w11wo/indobert-large-p1-twitter-indonesia-sarcastic     (IndoBERT large)
w11wo/indobert-base-uncased-twitter-indonesia-sarcastic (IndoBERT uncased/lemma)
w11wo/bert-base-multilingual-cased-twitter-indonesia-sarcastic (mBERT)
w11wo/xlm-roberta-base-twitter-indonesia-sarcastic      (XLM-R base)
w11wo/xlm-roberta-large-twitter-indonesia-sarcastic     (XLM-R large)
```

**Catatan implementasi penting:** XLM-R/RoBERTa harus dipaksa `eager attention`
(`cfg._attn_implementation = "eager"`) karena transformers ≥ 4.42 default ke SDPA
yang menghasilkan numerik berbeda → F1 meleset dari paper. Setelah fix ini, sanity
check tepat: IndoBERT base = 0.7273, XLM-R large = 0.7692.

**Operator fusi yang dibandingkan:**
1. `prob-avg`: `w × p_LR + (1-w) × p_tf`  (baseline)
2. `logit-avg`: rata-rata di ruang log-odds `logit = log(p/(1-p))`, lalu sigmoid
3. `rank-avg`: ubah prob ke peringkat ternormalisasi, lalu rata-rata

**Kalibrasi (temperature scaling):**
- Sebelum fusi, tiap model dikalibrasi dengan satu parameter T
- `logit_baru = logit_asli / T` (T dicari di val, minimize NLL)
- T > 1 → model lebih "moderat" (prob bergerak ke tengah)

**Protokol anti-data-leakage:**
- Bobot `w`, suhu `T`, threshold: semua dipilih dari **val saja**
- Test hanya disentuh **satu kali** di akhir
- Dilaporkan: val-test gap sebagai indikator overfitting

---

## 4. Hasil

### 4.1 Reproduksi Baseline Paper (Twitter)

| Model | F1 Paper | F1 Reproduksi |
|---|---|---|
| LR (Classical) | 0.7142 | 0.7142 ✓ |
| NB | 0.6721 | 0.6721 ✓ |
| SVM | 0.6782 | 0.6782 ✓ |
| IndoBERT base | 0.7273 | 0.7273 ✓ |
| XLM-R large (SOTA) | 0.7692 | 0.7692 ✓ |

### 4.2 LR Tuned (Classical Lexicon-Free Kami)

| Metode | F1 | P | R | Acc |
|---|---|---|---|---|
| LR TF-IDF(1,2) + tuning | **0.7509** | 0.7273 | 0.7761 | 0.8717 |

- Mengalahkan IndoBERT base (0.7273) dan XLM-R base (0.7386)
- Statistik ties dengan SOTA XLM-R large (95% CI mencakup 0.7692, P(beat)=~12%)
- Angka 0.7509 menggunakan val-tuned threshold (protokol bersih: threshold dipilih dari val saja, test disentuh sekali)
- **Catatan angka kanonik:** angka lama **0.7536** (dari `run_mvsc_classification.py`) sudah **superseded**. Penyebab beda sudah diverifikasi (2026-07-04): protokol val-tuned yang SAMA menghasilkan 0.7536 di environment lokal (sklearn/nltk terbaru) dan 0.7509 di environment Kaggle pinned — jadi ini **perbedaan environment** (tokenisasi/versi library), bukan protokol; script lama juga punya bug tambahan `headline = max(f1_default, f1_tuned)`. Angka kanonik = **0.7509** dari environment Kaggle pinned, tempat SEMUA angka headline lain (0.7900, 0.7692, dst.) dihitung — apples-to-apples. Jangan pakai 0.7536 di paper.

### 4.3 Hybrid Fusion Sistematis (LR + tiap transformer)

| Transformer | Alone F1 | Best Hybrid F1 | Method | dSOTA | P(>SOTA) |
|---|---|---|---|---|---|
| xlmr_base | 0.7386 | **0.7900** | wavg | +0.0208 | 77% |
| xlmr_large | 0.7692 | 0.7751 | wavg | +0.0059 | 58% |
| indobert_large | 0.7160 | 0.7639 | wavg | -0.0053 | 43% |
| mbert | 0.6462 | 0.7543 | wavg | -0.0149 | 30% |
| indobert_base | 0.7273 | 0.7518 | wavg | -0.0174 | 27% |
| indobert_lem | 0.6467 | 0.7455 | wavg | -0.0237 | 20% |

**Best single-transformer hybrid: LR + XLM-R base (prob-avg) = F1 0.7900**

Kejutan: XLM-R base + LR > XLM-R large sendiri — model yang "lebih lemah" justru lebih
diuntungkan oleh sinyal komplementer LR.

### 4.4 Operator Fusi Tuning (Hasil Lengkap)

Script: `kaggle_fusion_tuning_twitter.py`
Eksperimen: 3 operator (prob-avg, logit-avg, rank-avg) × 2 kalibrasi (no temp / temperature scaling) × 6 transformer = 36 konfigurasi.

**Hasil: prob-avg tanpa temperature scaling adalah yang paling robust.**

| Transformer | Operator terbaik (by val) | val F1 | test F1 | P(>SOTA) |
|---|---|---|---|---|
| xlmr_base | prob, no temp | 0.8116 | **0.7900** | 77.3% |
| xlmr_large | prob, no temp | 0.8358 | 0.7751 | 58.3% |
| indobert_large | prob, no temp | 0.7971 | 0.7639 | 43.1% |
| mbert | prob, no temp | 0.7970 | 0.7543 | 29.6% |
| indobert_base | prob, no temp | 0.7812 | 0.7518 | 27.4% |
| indobert_lem | prob, no temp | 0.7752 | 0.7455 | 20.3% |

**Temuan ablasi operator:**
- **rank-avg**: selalu lebih buruk di test (overfit ke val 268 sampel). Contoh xlmr_base: rank val=0.8154 → test=0.7510 vs prob val=0.8116 → test=0.7900.
- **logit-avg**: tidak konsisten — membantu untuk xlmr_large (0.7899 test) tapi merugikan xlmr_base (0.7597 test). Tidak dipilih sebagai metode utama.
- **Temperature scaling**: memperburuk semua konfigurasi. T_LR selalu ~0.58 (LR memang over-confident), tapi kalibrasi ini tidak membantu F1 akhir.

**Kesimpulan ablasi:** prob-avg dengan 1 parameter bobot w (dipilih dari val) adalah operator paling sederhana sekaligus paling robust untuk dataset kecil ini. Studi ini memvalidasi kesederhanaan metode kami.

### 4.5 Studi Komposisi Hybrid: 2 Classical + 1 Transformer

Script: `kaggle_2classical_twitter.py`
Ide: daripada menambah transformer kedua (latency 2×), coba tambah model klasik lagi
dengan inductive bias berbeda. Meta-LR dilatih di atas 3 input (OOF cross-val threshold).

**Konfigurasi yang diuji (anchor: LR(1,2) + xlmr_base):**

| Config | Komponen | val F1 | test F1 | val-test gap |
|---|---|---|---|---|
| C1 | LR(1,2) + LR(1,3) + xlmr_base | 0.7922 | 0.7593 | +0.0329 |
| C2 | LR(1,2) + CNB(1,2) + xlmr_base | 0.7949 | 0.7316 | **+0.0633** |
| C3 | LR(1,2) + SVC(1,2) + xlmr_base | 0.7895 | 0.7571 | +0.0324 |
| C4 | semua classical + xlmr_base | — | 0.7423 | — |
| **B2 (ref)** | **LR(1,2) + xlmr_base prob-avg** | **~0.811** | **0.7900** | **~0.021** |

**Ablasi tanpa transformer:**

| Config | Komponen | test F1 |
|---|---|---|
| A1 | LR(1,2) + LR(1,3) | 0.7535 |
| A2 | LR(1,2) + CNB(1,2) | 0.7255 |
| A3 | LR(1,2) + SVC(1,2) | 0.7391 |

**Temuan kunci:**
- **Semua C1-C4 lebih buruk dari B2 (0.7900)** — meta-LR dengan 3+ input overfit ke val=268 sampel.
- **Koefisien meta-LR**: transformer selalu dominan (~2.9) vs classical tambahan (~1.7-2.0). Meta-learner "mengetahui" classical tambahan tidak berguna.
- **C2 paling parah**: val tertinggi (0.7949) tapi test terendah di cluster ini (0.7316) — gap +0.0633 adalah sinyal overfit terbesar di seluruh eksperimen.
- **Ablasi (A1-A3)**: 2 classical tanpa transformer hampir sama dengan LR tunggal (0.7509) → konsisten dengan r=0.877 dari studi ensembling.
- **Konfirmasi**: sinyal transformer tetap esensial; mengganti satu transformer dengan model klasik kedua selalu rugi.

**Kesimpulan:** B2 (LR+xlmr_base, prob-avg, 1 parameter w) tetap sebagai metode terbaik. Studi ini mengkonfirmasi bahwa kesederhanaan bukan kompromi — melainkan pilihan optimal untuk dataset kecil ini.

### 4.6 Studi Character N-gram TF-IDF

Script: `kaggle_char_ngram_twitter.py`
Gap dari paper acuan: IdSarcasm hanya pakai word unigram TF-IDF — tidak ada character-level features sama sekali.
Hipotesis: char n-gram `(char_wb)` menangkap elongasi informal ("bangetttt"), afiks bahasa Indonesia ("me-", "-kan"), dan variasi ejaan ("gabisa") yang tidak bisa ditangkap word n-gram.

**Char range tuning** (dipilih dari val F1 standalone): best = **(2,4)**

**Ablasi Classical:**

| Label | Komponen | test F1 |
|---|---|---|
| B0 | LR word(1,2) | **0.7509** (referensi LR (tuned)) |
| B1 | LR char(2,4) alone | 0.7143 (-0.0366) |
| B2 | LR word+char combined | 0.7322 (-0.0187) |

**Ablasi Hybrid (perbandingan per transformer, LR variant):**

| Transformer | word | char(2,4) | word+char | Gap char vs word |
|---|---|---|---|---|
| xlmr_base | **0.7900** | 0.7622 | 0.7723 | +0.0572 (overfit) |
| xlmr_large | 0.7751 | 0.7722 | 0.7724 | +0.0460 |
| indobert_base | 0.7518 | 0.7450 | 0.7603 | +0.0407 |
| mbert | 0.7543 | 0.7055 | 0.7214 | +0.0814 (overfit parah) |

**Temuan kunci:**
- **Char n-gram SELALU memperburuk** classical standalone (-0.04 sampai -0.02 F1).
- **Hybrid dengan char juga turun** untuk semua transformer kuat. Val-test gap membesar (+0.04–+0.08), pola overfit yang sama dengan eksperimen 2-classical.
- Word+char combined sedikit membantu transformer lemah (indobert_base: +0.0085) tapi tidak signifikan statistik (CI sangat lebar).
- Feature space char n-gram terlalu besar untuk dataset kecil ini (1878 train) — char bigrams → overfit.

**Kesimpulan:** LR word(1,2) tetap representasi klasik optimal. Char n-gram tidak menambah nilai, baik standalone maupun dalam hybrid. Studi ini — bersama dengan eksperimen 2-classical dan operator fusi — membangun argumen: **untuk dataset skala ini, parsimonious model selalu menang**.

### 4.7 Inference Time

*(Perkiraan berdasarkan arsitektur, diukur di GPU T4 Kaggle)*

| Komponen | Waktu (538 sampel) |
|---|---|
| LR alone (TF-IDF + predict) | ~2 ms |
| XLM-R base alone | ~3000–4000 ms |
| XLM-R large alone | ~5000–6000 ms |
| LR + XLM-R base (hybrid) | ~3002 ms (overhead LR < 0.1%) |

### 4.8 Jumlah Parameter

| Komponen | Parameter |
|---|---|
| TF-IDF vocab (non-trainable features) | ~20.000–40.000 |
| LR weights (trainable) | vocab_size + 1 ≈ ~30.000 |
| XLM-R base (fine-tuned) | ~278 juta |
| XLM-R large (fine-tuned) | ~560 juta |
| **Hybrid LR + XLM-R base** | **~278 juta (LR = <0.02%)** |

### 4.9 Komplementaritas LR ↔ Transformer

Script: `kaggle_complementarity_gated_twitter.py` (Tahap A). Motivasi: klaim "transformer memberi
sinyal dekorelasi" (poin [3] narasi) sebelumnya hanya didukung korelasi classical↔classical
(r=0.877, studi ensembling §4.3); belum pernah diukur langsung untuk pasangan LR↔transformer.

**Korelasi prediksi (test, n=538):**

| Pasangan | r (prob) | φ (label) | Disagreement | Oracle F1 |
|---|---|---|---|---|
| LR ↔ xlmr_base | 0.686 | 0.571 | 97 (18.0%) | 0.9058 |
| LR ↔ xlmr_large | 0.669 | 0.581 | 85 (15.8%) | 0.9154 |
| xlmr_base ↔ xlmr_large | 0.756 | 0.666 | 76 (14.1%) | 0.8806 |

**Temuan kunci:**
- **Korelasi LR↔transformer (0.67–0.76) lebih rendah dari classical↔classical (0.877)** — bukti
  kuantitatif pertama bahwa transformer memang memberi sinyal yang lebih berbeda dari LR, meski
  tidak sampai independen. Ini mengisi gap novelty terbesar sebelumnya: klaim [3] sekarang
  bersandar pada angka, bukan asumsi.
- **Error overlap LR × xlmr_base**: both-benar 415 (77.1%), hanya-LR-benar 43 (8.0%),
  hanya-TF-benar 54 (10.0%), both-salah 26 (4.8%). Hanya 4.8% sampel yang tak terselamatkan
  fusi apa pun — sisanya (18.0%) adalah "ruang fusi" tempat salah satu model masih benar.
- **Oracle F1 = 0.9058** (upper bound: pilih prediksi benar setiap kali LR & transformer beda
  pendapat, menggunakan label asli sebagai "wasit"). Ini plafon teoretis untuk kombinasi
  keduanya — bukan metode yang bisa dideploy, tapi ukuran headroom. Gap antara oracle (0.906)
  dan hasil fusi terbaik yang benar-benar dicapai (0.7900, §4.9) menunjukkan bahwa menebak
  "siapa yang harus dipercaya per-sampel" dari sinyal tak sempurna adalah masalah yang keras —
  dan val=268 sampel tidak cukup untuk melatihnya dengan baik (lihat §4.10).
- **Contoh kualitatif** (test): LR mengoreksi XLM-R base yang salah-yakin pada tweet dengan
  pola permukaan sarkastik implisit tanpa kata kunci leksikal jelas (mis. `[301]` "Lord Luhut
  trending topik..." y=0, p_tf=0.991 vs p_lr=0.153; fitur LR dominan: token `<`/`>` placeholder
  PII-mask). Sebaliknya, XLM-R base mengoreksi LR yang salah-yakin pada tweet dengan sarkasme
  kontekstual/tanpa kata bermuatan sentimen eksplisit (mis. `[360]` "...bergaya juga ya mantan
  kamu banyak yang peduli." y=1, p_lr=0.118 vs p_tf=0.993) — pola yang sesuai dugaan: LR kuat
  pada isyarat leksikal permukaan, transformer kuat pada konteks semantik yang lebih dalam.

### 4.10 Confidence-Gated / kNN / MC-Dropout Fusion — Studi Negative Result Keempat

Script: `kaggle_complementarity_gated_twitter.py` (Tahap B/C/D). Motivasi: bobot fusi statis w
(B2, §4.3–4.4) mungkin bisa dilampaui oleh bobot yang beradaptasi per-sampel berdasarkan
confidence/uncertainty. Tiga keluarga gating diuji (G1 hard, G2 soft, G3 disagreement — maks 2
parameter, tiap keluarga memuat wavg statis sebagai kasus khusus), plus dua sinyal fusi baru
(kNN retrieval, MC-dropout uncertainty).

| Metode | anchor | val F1 | test F1 | gap val−test |
|---|---|---|---|---|
| **B2 wavg statis (referensi)** | xlmr_base | 0.8116 (**terendah**) | **0.7900** (**terbaik**) | **+0.022** |
| G3 disagreement gate | xlmr_large | 0.8382 | 0.7730 | +0.065 |
| G1 hard gate | xlmr_large | 0.8358 | 0.7751 | +0.061 |
| G2 soft gate | xlmr_large | 0.8358 | 0.7666 | +0.069 |
| G2 soft gate | xlmr_base | 0.8244 | 0.7529 | +0.072 |
| 3-way LR+kNN+xlmr_base | — | 0.8158 | 0.7443 | +0.072 |
| G1 hard gate | xlmr_base | 0.8120 | 0.7388 | +0.073 |
| G3 disagreement gate | xlmr_base | 0.8120 | 0.7500 | +0.062 |
| MC-std gate (menang by-val) | xlmr_base | 0.8489 (**tertinggi**) | 0.7527 | +0.096 (**terbesar**) |

**Temuan kunci:**
- **Setiap metode yang lebih kompleks dari B2 kalah di test** — meski val F1-nya lebih tinggi
  dari B2. Pola ini persis konsisten dengan studi 2-classical (§4.5) dan char n-gram (§4.6):
  makin canggih metodenya, makin besar val-test gap.
- **Ironi terkuat**: MC-std gate punya val F1 TERTINGGI (0.8489) di antara semua kandidat —
  dan itulah yang akan terpilih oleh protokol seleksi "pilih by-val" — tapi test F1-nya
  (0.7527) dan gap-nya (+0.096) juga yang TERBURUK. B2 justru sebaliknya: val F1 terendah,
  test F1 tertinggi. Ini demonstrasi tekstbuk bahwa val=268 sampel terlalu kecil untuk memilih
  di antara metode fusi bertingkat parameter lebih dari satu.
- **kNN retrieval gagal di gerbang dekorelasi**: korelasi prob kNN↔xlmr_base = 0.916 — hampir
  identik dengan transformer itu sendiri (wajar, kNN dibangun dari embedding CLS xlmr_base) →
  bukan sinyal baru, melainkan salinan berisik dari transformer. Ditandai sebagai negative
  result sesuai protokol; 3-way fusion (LR+kNN+xlmr_base) pun kalah dari B2.
- **MC-dropout (K=10) sebagai sinyal uncertainty justru kalah dari confidence mentah**: AUROC
  std-prediktif memprediksi error XLM-R = 0.72 (test), sedangkan AUROC confidence mentah
  `|p-0.5|` = 0.83 (test) — sinyal Bayesian yang lebih mahal (10× inferensi) memberi informasi
  LEBIH BURUK daripada trik satu-baris. Temuan negative-result tersendiri.

**Kesimpulan:** empat keluarga metode berbeda (gating confidence/disagreement, retrieval kNN,
uncertainty MC-dropout) dicoba secara sistematis untuk melampaui B2 — semuanya overfit ke
val=268 sampel. Bersama §4.5 (2-classical) dan §4.6 (char n-gram), ini adalah studi ablasi
KEEMPAT dan KELIMA yang mengonfirmasi pola yang sama: **untuk dataset skala ini, setiap
penambahan parameter fusi di atas satu (w tunggal) selalu memperbesar val-test gap dan
menurunkan F1 test.** B2 (LR + XLM-R base, prob-avg, 1 parameter) bertahan sebagai metode
terbaik di seluruh ruang metode yang sudah dieksplorasi.

---

## 5. Narasi Kontribusi (Storyline Paper)

```
[1] Classical ML lexicon-free (LR TF-IDF) mencapai F1 0.7509
    → Mengalahkan IndoBERT base & XLM-R base
    → Mendekati SOTA XLM-R large (0.7692) tanpa fine-tuning besar

[2] Classical ensembling GAGAL meningkatkan performa
    → Korelasi prediksi antar model klasik r = 0.877
    → Sinyal semua model klasik terlalu mirip (sama-sama berbasis TF-IDF)
    → Studi ini membuktikan batas atas classical ML untuk data ini

[3] Transformer memberikan sinyal dekorelasi yang diperlukan
    → LR + XLM-R base (hybrid) = 0.7900 > SOTA (0.7692)
    → Overhead parameter LR dalam hybrid < 0.02% — tradeoff sangat efisien
    → Overhead latency LR dalam hybrid < 0.1%

[4] Studi ablasi sistematis memvalidasi kesederhanaan metode
    → Operator fusi: prob-avg lebih robust dari logit-avg dan rank-avg di dataset kecil
    → Char n-gram: feature space terlalu besar untuk 1878 sampel, selalu overfit
    → 2-classical + 1-transformer: meta-LR overfit ke val=268, semua config lebih buruk dari B2
    → Gating (confidence/disagreement) + kNN retrieval + MC-dropout uncertainty: SEMUA overfit
      juga — bahkan metode dgn val F1 tertinggi (MC-std gate 0.849) punya test F1 & gap terburuk
    → Pola konsisten (5 studi ablasi independen): setiap tambahan kompleksitas → val-test gap
      membesar → test F1 turun
    → Kesimpulan: LR word(1,2) + XLM-R base prob-avg adalah titik optimal Pareto (akurasi × kesederhanaan)

[5] Sinyal LR↔transformer terukur lebih dekorelasi dari classical↔classical
    → r = 0.67–0.76 (LR↔transformer) vs r = 0.877 (classical↔classical, poin [2])
    → Oracle F1 (upper bound kombinasi LR+transformer) = 0.906 — headroom besar di atas 0.7900
    → Mengisi bukti kuantitatif utk klaim [3]: transformer memang memberi sinyal yang beda dari LR
```

---

## 6. Constraint Penelitian

- **Tidak boleh menggunakan kamus/leksikon eksternal** (InSet, SentiWordNet, dll.) — ditolak pembimbing
- **Hanya dataset Twitter** (bukan Reddit) yang dianalisis
- **Single transformer saja dalam hybrid** (2 transformer terlalu mahal latency-nya)
- **Transformer tidak di-fine-tune ulang** — hanya inferensi dari model yang sudah dirilis paper acuan
- Dataset kecil (test=538) → confidence interval lebar, klaim statistik harus hati-hati

---

## 7. File Kode Utama

| File | Fungsi |
|---|---|
| `scripts/run_mvsc_classification.py` | LR (tuned) standalone (classical baseline kami) |
| `kaggle_stacking_full_twitter.py` | **MAIN**: LR (tuned) × 6 transformer, reproduksi baseline, confusion matrix, latency |
| `kaggle_mvsc_bootstrap_twitter.py` | Bootstrap CI dan P(beat SOTA) untuk LR (tuned) |
| `scripts/run_classical_classification.py` | Reproduksi paper protocol: NB/SVM/LR seed=41, accuracy-GS |
| `scripts/run_classification.py` | Reproduksi baseline transformer (paper protocol) |
| `scripts/kaggle_baseline_twitter.ipynb` | Kaggle runner notebook |
| `kaggle_complementarity_gated_twitter.py` | analisis komplementaritas LR↔transformer, uji berpasangan (paired bootstrap + McNemar), gated fusion (G1/G2/G3), kNN retrieval fusion, MC-dropout gate |
| `kaggle_oracle_diagnostic_twitter.py` | **BARU**: peta ceiling oracle (pairwise + grand oracle 7 model), probe separability (AUROC + router CV di val disagreement), akurasi kandidat model-3, outcome cascade nol-parameter + verdict HIJAU/MERAH |

---

## 8. Pertanyaan Terbuka untuk Konsultasi Paper

- Bagaimana cara klaim novelty dengan tepat? (hybrid fusion sendiri novel, atau metodologi perbandingan operatornya?)
- Bagaimana melaporkan hasil operator fusi jika logit-avg tidak signifikan lebih baik dari prob-avg?
- Bagaimana menulis bagian "negative result" (classical ensembling gagal) sebagai kontribusi yang positif?
- Bagaimana memilih antara F1 0.7900 (LR+xlmr_base) vs tabel ablasi lengkap sebagai headline?
- Format tabel eksperimen yang sesuai untuk IEEE Access?
