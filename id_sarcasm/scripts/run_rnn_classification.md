# Baseline RNN (Bi-LSTM / Bi-GRU) — Twitter IdSarcasm

Baseline ringan (low-parameter) sebagai pembanding protokol yang ADIL terhadap
baseline transformer (`scripts/run_classification.py`). Script: `scripts/run_rnn_classification.py`.
Recipe: `recipes/twitter/baseline/rnn_{bilstm,bigru}_{random,fasttext}_twitter.sh`.

## 1. Apa yang dibuat IDENTIK (apple-to-apple)

| Aspek | Nilai (sama dgn `run_classification.py`) |
|---|---|
| Dataset + split | Twitter IdSarcasm, train 1878 / val 268 / test 538 (imbalance ~75/25) |
| `max_seq_length` | 128 (truncation kanan) |
| Metrik utama | **F1 BINARY kelas positif** (label 1 = sarkas) + accuracy / precision / recall |
| Pemilihan model | **F1 validasi terbaik** (mirror `load_best_model_at_end` + `metric_for_best_model=f1`) |
| Early stopping | patience 3, threshold 0.01 (mirror `EarlyStoppingCallback`) |
| Budget epoch | cap 100 + early stop |
| Evaluasi test | **SEKALI** di akhir dgn model val-terbaik (mirror `trainer.evaluate(predict_dataset)`) |
| Seed | default 42; multi-seed → mean ± std |
| Output | `eval_results.json` kunci `eval_accuracy / eval_f1 / eval_precision / eval_recall` + `predict_results.txt` |

Metrik dihitung dgn `sklearn` `average="binary", pos_label=1` — identik dengan
`evaluate.load("f1")` default yang dipakai baseline transformer.

## 2. Deviasi yang DISENGAJA (RNN dilatih from scratch)

RNN acak BUKAN fine-tuning transformer, jadi LR transformer (1e-5) tidak relevan.
Deviasi ini variabel bebas model dan didokumentasikan eksplisit:

| Hyperparameter | RNN baseline | Transformer baseline | Alasan |
|---|---|---|---|
| Optimizer / LR | **Adam, 1e-3** | AdamW, 1e-5 | bobot acak butuh LR jauh lebih besar utk konvergen |
| Scheduler | none (konstan) | cosine | baseline ringan; cosine tak material di RNN kecil |
| fp16 | off (default) | on | RNN ringan, CPU-friendly; flag `--fp16` tersedia utk GPU |
| Tokenisasi | **whitespace split** | WordPiece | data sudah ber-marker; split spasi menjaga `<username>`/`<link>`/`<hashtag>` UTUH |
| Loss default | CE tanpa bobot | CE tanpa bobot | sama; `--class_weight` menyalakan balanced CE |

`--do_lower_case` default ON (konsisten antar varian embedding).

## 3. Ablation embedding (`--embedding`)

1. **`random`** — `nn.Embedding` init acak, TRAINABLE dari nol; vocab dari train set.
   Baseline "paling murni low-resource".
2. **`fasttext`** — init dari pretrained fastText Indonesia `cc.id.300` (300-dim):
   - **`.bin`** → setiap kata (termasuk OOV) dapat vektor via subword n-gram
     (`pip install fasttext`). Handle OOV penuh — penting utk slang/typo Twitter.
   - **`.vec`** → hanya kata di file yang terisi; OOV = **vektor nol** (lebih ringan,
     tanpa subword). OOV nol tetap bisa belajar bila embedding tidak di-`--freeze_embedding`.

Vocab & tokenisasi konsisten antar varian, jadi perbandingan random vs fastText
mengisolasi efek pretrained embedding.

## 4. Model

`embedding → Bi-LSTM/Bi-GRU (1 layer, hidden 128/arah) → pooling (max default) → Dropout → Linear(→2)`.
Parameter ~0.5M (fastText, embedding di-freeze) s.d. ~2.9M (random 300-dim trainable),
kontras dgn transformer 110M–560M.

## 5. Cara menjalankan

```bash
cd id_sarcasm
# Random (tanpa unduhan apa pun):
bash recipes/twitter/baseline/rnn_bilstm_random_twitter.sh
bash recipes/twitter/baseline/rnn_bigru_random_twitter.sh

# fastText: unduh cc.id.300.bin (atau .vec) dulu, set path:
FASTTEXT_PATH=embeddings/cc.id.300.bin bash recipes/twitter/baseline/rnn_bilstm_fasttext_twitter.sh
FASTTEXT_PATH=embeddings/cc.id.300.vec bash recipes/twitter/baseline/rnn_bigru_fasttext_twitter.sh
```

Sumber fastText: <https://fasttext.cc/docs/en/crawl-vectors.html> (`cc.id.300`).
Ekuivalen HF (alih-alih CSV lokal): ganti `--*_file` dgn `--use_hf_dataset --text_column_name tweet`.

## 6. Titik banding (Twitter, F1 binary kelas positif)

| Model | F1 |
|---|---|
| Klasik (LR/SVM/NB, lihat `results/classical/`) | ~0.68–0.72 |
| IndoBERT-base (paper / reproduksi env ini) | 0.7273 / 0.7306 |
| XLM-R-large (tertinggi) | 0.7692 |

RNN ringan WAJAR lebih rendah; tujuannya pembanding protokol yang adil + menunjukkan
efek pretrained embedding (random vs fastText). Semua keputusan di valid; test sekali di akhir.

## 7. Output

```
outputs/rnn-<model>-<embedding>-twitter/
├── eval_results.json        # mean ± std antar-seed; kunci eval_* + config + per_seed
├── predict_results.txt      # prediksi test (seed pertama), format index<TAB>prediction
└── seed_<s>/                # per-seed: eval_results.json + predict_results.txt
```
