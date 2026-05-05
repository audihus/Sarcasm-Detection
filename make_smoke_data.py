# -*- coding: utf-8 -*-
"""
make_smoke_data.py
==================
Membuat data uji cepat untuk smoke-test train_multichannel.py.

Yang dilakukan:
  1. Baca N baris pertama dari IAC1/spacy/train.txt dan test.txt
  2. Simpan sebagai CSV (kolom: content, label)
  3. Buat file .graph.new dan .sentic berisi adjacency matrix SINTETIS
     (random, ukuran sesuai panjang kata per kalimat) -- tanpa Stanza

Hasilnya disimpan di ./smoke_test_data/

Jalankan:
    python make_smoke_data.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# ── Konfigurasi ──────────────────────────────────────────────────────────────
REPO_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "multichannel-sarcasm-detection")
IAC1_DIR   = os.path.join(REPO_DIR, "IAC1", "spacy")
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "smoke_test_data")

TRAIN_TXT  = os.path.join(IAC1_DIR, "train.txt")
TEST_TXT   = os.path.join(IAC1_DIR, "test.txt")

N_TRAIN    = 80   # baris yang diambil dari train.txt
N_TEST     = 20   # baris yang diambil dari test.txt
SEED       = 42

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def read_jsonlines(path: str, n: int) -> list:
    """Baca n baris pertama dari file JSON-lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_synthetic_graph(text: str) -> np.ndarray:
    """
    Buat adjacency matrix sintetis berukuran (n_words × n_words).
    Hanya untuk keperluan smoke-test — tidak merepresentasikan struktur
    dependensi nyata.
    """
    words  = text.lower().split()
    n      = max(len(words), 1)
    mat    = np.zeros((n, n), dtype=np.float32)
    # Tambahkan self-loop + beberapa koneksi acak agar graph tidak kosong
    np.fill_diagonal(mat, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < 0.3:
                mat[i, j] = 1.0
                mat[j, i] = 1.0
    return mat


def save_split(rows: list, split_name: str):
    """
    Simpan CSV dan file graph sintetis untuk satu split.
    - CSV    : {OUT_DIR}/{split_name}.csv
    - graph  : {OUT_DIR}/{split_name}.csv.graph.new
    - sentic : {OUT_DIR}/{split_name}.csv.sentic
    """
    df = pd.DataFrame([
        {"content": r["content"], "label": r["label"]}
        for r in rows
    ])
    df = df.reset_index(drop=True)

    csv_path = os.path.join(OUT_DIR, f"{split_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  CSV  → {csv_path}  ({len(df)} rows)")

    # Buat graph pickle: dict[int → np.ndarray]
    idx2dep = {}
    idx2sen = {}
    for i, row in df.iterrows():
        idx2dep[i] = make_synthetic_graph(str(row["content"]))
        idx2sen[i] = make_synthetic_graph(str(row["content"]))

    graph_path  = os.path.join(OUT_DIR, f"{split_name}.csv.graph.new")
    sentic_path = os.path.join(OUT_DIR, f"{split_name}.csv.sentic")

    with open(graph_path,  "wb") as f:
        pickle.dump(idx2dep, f)
    with open(sentic_path, "wb") as f:
        pickle.dump(idx2sen, f)

    print(f"  Graph → {graph_path}")
    print(f"  Senti → {sentic_path}")

    # Tampilkan distribusi label
    dist = df["label"].value_counts().to_dict()
    print(f"  Label distribution: {dist}")


# ── Main ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  make_smoke_data.py  –  Membuat data smoke-test")
print("=" * 60)

print(f"\n[Train] Membaca {N_TRAIN} baris dari {TRAIN_TXT} ...")
train_rows = read_jsonlines(TRAIN_TXT, N_TRAIN)
save_split(train_rows, "train")

print(f"\n[Test]  Membaca {N_TEST} baris dari {TEST_TXT} ...")
test_rows = read_jsonlines(TEST_TXT, N_TEST)
save_split(test_rows, "test")

print(f"\n{'=' * 60}")
print("  Selesai! Jalankan smoke test dengan:")
print(f"{'=' * 60}")
print(f"""
python train_multichannel.py \\
  --train_data       {os.path.join(OUT_DIR, 'train.csv')} \\
  --test_data        {os.path.join(OUT_DIR, 'test.csv')} \\
  --graph_dir        {OUT_DIR} \\
  --train_split_name train.csv \\
  --test_split_name  test.csv  \\
  --epochs 1 \\
  --batch_size 4 \\
  --checkpoint_dir   ./smoke_ckpt/
""")
