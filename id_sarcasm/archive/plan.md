Konteks proyek: deteksi sarkasme Bahasa Indonesia (dataset IdSarcasm, Suhartono et al. 2024).
Target: beat baseline IndoBERT-base (Reddit F1-binary 0.6135, Twitter F1-binary 0.7306).
Compute: Kaggle Free T4. Encoder: indobenchmark/indobert-base-p1.

Tiga file preprocessing sudah ada di project:
- preprocessing/augment_pipeline.py  (core: emoji expand, polarity clash, emo conflict)
- scripts/sanity_check.py            (sample 100 teks, hitung lift, output markdown report)
- scripts/preprocess_datasets.py     (apply ke full dataset, save ke disk)

InSet lexicon ada di: real_data/{reddit,twitter}/positive.tsv dan negative.tsv
Dataset HuggingFace: w11wo/reddit_indonesia_sarcastic, w11wo/twitter_indonesia_sarcastic
run_classification.py ada di: scripts/run_classification.py (sudah stable, jangan diubah)

TUGAS:

1. Baca ketiga file preprocessing yang ada. Pahami struktur dan interface-nya.

2. Jalankan sanity check untuk Reddit dulu:
   python scripts/sanity_check.py --dataset reddit --n_samples 100 --project_root .
   Baca output report-nya. Laporkan: lift score, pct_clash, pct_emo_conflict, dan verdict GO/NO-GO.

3. Kalau Reddit GO, jalankan untuk Twitter juga.

4. Kalau kedua dataset GO, jalankan preprocessing full_hybrid untuk keduanya:
   python scripts/preprocess_datasets.py --dataset reddit --variant full_hybrid --project_root .
   python scripts/preprocess_datasets.py --dataset twitter --variant full_hybrid --project_root .

5. Cek bagaimana run_classification.py load dataset-nya (cari argumen --dataset_name atau
   load_dataset call). Kalau script itu hanya support HuggingFace hub name dan belum support
   load_from_disk, tambahkan support path lokal. Patch harus minimal, jangan ubah logika lain.

6. Verifikasi end-to-end: pastikan preprocessed dataset bisa di-load dan kolom text/label
   ada dan formatnya benar sebelum training dimulai.

CONSTRAINTS:
- Jangan ubah logika training di run_classification.py selain bagian load dataset
- Vanilla CrossEntropy, jangan tambah class weight
- Seed 42
- Kalau ada error di augment_pipeline.py (misal library emoji tidak tersedia), fix di sana
- Kalau lift < 1.0 di sanity check, STOP dan laporkan ke saya sebelum lanjut