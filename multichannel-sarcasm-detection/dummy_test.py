# -*- coding: utf-8 -*-
"""
dummy_test.py  –  End-to-End Sanity Check
==========================================
Tujuan  : Membuktikan bahwa dimensi tensor dari IndoBERT (768-dim)
          + ADGCN/BiLSTM (512-dim) = 1280-dim tersambung sempurna hingga
          FocalLoss backward-pass tanpa Shape-Mismatch Error.

Cara    : Jalankan langsung dari root repo:
              python dummy_test.py

Catatan : Skrip ini TIDAK mengubah file arsitektur apapun.
          Semua perubahan hanya ada di sini sebagai pemanggil.

Arsitektur yang diuji:
    IndoBERT (768) + ADGCN BiLSTM hard-coded (256*2=512) → fused 1280
    → Dense(1280→256→2) → FocalLoss(γ=2, α=0.25)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from types import ModuleType
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# MOCK BROKEN IMPORTS & PATCHES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Mock 'models' module
# GarphModel.py has a broken 'from models.affectivegcn import GraphConvolution'
# but it redefines GraphConvolution immediately after. We mock 'models'
# to satisfy the import without needing the actual files.
if 'models' not in sys.modules:
    mock_models = ModuleType("models")
    sys.modules["models"] = mock_models
    mock_affectivegcn = ModuleType("affectivegcn")
    sys.modules["models.affectivegcn"] = mock_affectivegcn
    # Set to None since it's shadowed in GarphModel.py anyway
    mock_affectivegcn.GraphConvolution = None

from transformers import AutoModel


# ─────────────────────────────────────────────────────────────────────────────
# 0.  KONFIGURASI PATH  –  pastikan modul repo bisa diimpor
# ─────────────────────────────────────────────────────────────────────────────
# Tambahkan direktori repo ke sys.path agar modul lokal bisa ditemukan
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("=" * 70)
print("  DUMMY TEST  –  End-to-End Sanity Check")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MOCK DATASET
#     3 teks literal  +  1 teks sarkasme  → simulasi class imbalance (3:1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 1] Membuat Mock Dataset (imbalanced 3 literal : 1 sarkasme)...")

MOCK_TEXTS = [
    "Pelayanan restoran ini sangat memuaskan dan makanannya enak.",       # literal (0)
    "Cuaca hari ini sangat cerah dan menyenangkan untuk beraktivitas.",   # literal (0)
    "Produk ini berkualitas tinggi dan harganya sangat terjangkau.",      # literal (0)
    "Wah bagus sekali, antri dua jam tapi makanannya habis.",             # sarkasme (1)
]
MOCK_LABELS = [0, 0, 0, 1]

df_mock = pd.DataFrame({"text": MOCK_TEXTS, "label": MOCK_LABELS})

print(f"  Jumlah sampel : {len(df_mock)}")
print(f"  Distribusi label :\n{df_mock['label'].value_counts().to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MOCK GRAPH GENERATION
#     Panggil process() dari sentic_graph & dependency_graph
#     Output: file .sentic dan .graph (di-pickle) untuk batch ini
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Membuat Mock Graph Files (.graph & .sentic)...")

# Path output sementara di direktori yang sama
GRAPH_PATH  = os.path.join(REPO_DIR, "dummy_test.graph")
SENTIC_PATH = os.path.join(REPO_DIR, "dummy_test.sentic")

# Import HANYA jika file belum ada (hemat waktu saat re-run)
if not os.path.exists(GRAPH_PATH):
    print("  → Menjalankan dependency_graph.process() …")
    import dependency_graph as dep_graph_module
    dep_graph_module.process(
        data_input      = df_mock,
        text_column     = "text",
        output_filename = GRAPH_PATH,
    )
else:
    print(f"  → Melewati (file sudah ada): {GRAPH_PATH}")

if not os.path.exists(SENTIC_PATH):
    print("  → Menjalankan sentic_graph.process() …")
    import sentic_graph as sentic_graph_module
    sentic_graph_module.process(
        data_input      = df_mock,
        text_column     = "text",
        output_filename = SENTIC_PATH,
    )
else:
    print(f"  → Melewati (file sudah ada): {SENTIC_PATH}")

# Load graph dicts  {row_idx → np.ndarray(seq_len, seq_len)}
with open(GRAPH_PATH,  "rb") as f:
    dep_graphs = pickle.load(f)     # dict[int, np.ndarray]
with open(SENTIC_PATH, "rb") as f:
    sentic_graphs = pickle.load(f)  # dict[int, np.ndarray]

print(f"  Dependency graph keys  : {list(dep_graphs.keys())}")
print(f"  Sentic graph keys      : {list(sentic_graphs.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MOCK VOCAB & EMBEDDING
#     bridgeModel membutuhkan vocab list dan embedding numpy array.
#     Kita buat vocab minimal dari semua token yang muncul di teks,
#     ditambah token khusus <pad> dan <unk>.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Membuat Mock Vocab & Embedding …")

DIM_INPUT  = 100   # dimensi embedding kata ADGCN (bisa disesuaikan)

# Bangun vocab dari teks mock (tokenisasi sederhana berbasis spasi + lower)
all_words = set()
for text in MOCK_TEXTS:
    for w in text.lower().split():
        all_words.add(w)

SPECIAL_TOKENS = ["<pad>", "<unk>"]
vocab = SPECIAL_TOKENS + sorted(all_words)

# Embedding matrix acak (n_vocab × DIM_INPUT) – cukup untuk sanity check
np.random.seed(42)
embed_matrix = np.random.randn(len(vocab), DIM_INPUT).astype(np.float32)

print(f"  Vocab size   : {len(vocab)}  (termasuk <pad>, <unk>)")
print(f"  Embed shape  : {embed_matrix.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MOCK FLAGS (argparse-style namespace)
#     Sesuaikan dengan parameter yang diharapkan oleh bridgeModel & dualModel
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4] Menyiapkan FLAGS dummy …")

class MockFlags:
    """Namespace pengganti argparse.Namespace untuk sanity-check."""
    # ── Umum ──────────────────────────────────────────────────────────────── #
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_class         = 2               # non-sarcasm vs. sarcasm
    batch_size      = 4               # sesuai jumlah sampel mock
    max_length_sen  = 64              # panjang max token IndoBERTlite
    learning_rate   = 1e-3
    weight_decay    = 0.01
    t_sne           = False           # tidak perlu output t-SNE

    # ── ADGCN ─────────────────────────────────────────────────────────────── #
    dim_input       = DIM_INPUT       # dimensi embedding kata (100)
    dim_hidden      = 256             # BiLSTM → output = 256*2 = 512
    n_layers        = 1
    bidirectional   = True            # ← hardcoded True di GarphModel.py juga
    rnn_type        = "LSTM"
    embed_dropout_rate = 0.1

FLAGS = MockFlags()

print(f"  device         : {FLAGS.device}")
print(f"  dim_input      : {FLAGS.dim_input}")
print(f"  dim_hidden     : {FLAGS.dim_hidden}  (BiLSTM → {FLAGS.dim_hidden*2} dim output)")
print(f"  BERT dim       : 768  (IndoBERT base-p1)")
print(f"  Fused dim      : {768 + FLAGS.dim_hidden*2}  (768 + {FLAGS.dim_hidden*2})")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PERSIAPAN batched_data
#     Format yang diharapkan oleh gen_batch_data() di bridgeModel.py:
#       'sentences'         : list[str]        – teks mentah
#       'length_sen'        : list[int]        – jumlah token per kalimat
#       'dependency_graphs' : list[np.ndarray] – matrix graf (sudah dipad)
#       'sentic_graphs'     : list[np.ndarray] – matrix graf (sudah dipad)
#       'sarcasms'          : list[int]        – label 0/1
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Menyusun batched_data …")

# 5a. Sentences – digunakan oleh tokenizer IndoBERT DAN Lang.VariablesFromSentences
#     Lang.VariablesFromSentences(sentences, flag_list=True) mengharapkan
#     sentences = list of list-of-words (bukan raw string).
sentences_raw       = df_mock["text"].tolist()          # list[str] untuk IndoBERT
sentences_tokenized = [s.lower().split() for s in sentences_raw]  # list[list[str]] untuk Lang

# 5b. Panjang tiap kalimat (jumlah kata, digunakan ADGCN)
length_sen = [len(tok) for tok in sentences_tokenized]

# 5c. Padding graf ke ukuran max_len_graph agar bisa di-stack menjadi tensor 3-D
#     [B, max_len_graph, max_len_graph]
max_len_graph = max(
    max(dep_graphs[i].shape[0]    for i in range(len(df_mock))),
    max(sentic_graphs[i].shape[0] for i in range(len(df_mock))),
)
print(f"  max_len_graph (untuk padding graf) : {max_len_graph}")

def pad_graph(matrix: np.ndarray, target_size: int) -> np.ndarray:
    """Pad adjacency matrix ke ukuran [target_size × target_size] dengan zeros."""
    padded = np.zeros((target_size, target_size), dtype=np.float32)
    n = matrix.shape[0]
    padded[:n, :n] = matrix
    return padded

dep_graphs_padded    = [pad_graph(dep_graphs[i],    max_len_graph) for i in range(len(df_mock))]
sentic_graphs_padded = [pad_graph(sentic_graphs[i], max_len_graph) for i in range(len(df_mock))]

# Pad panjang kalimat juga ke max_len_graph agar tensor sens seragam
# dan COCOK dengan dimensi graf [B, max_len_graph, max_len_graph]
sentences_padded = []
for toks in sentences_tokenized:
    # Pad sampai max_len_graph
    padded_toks = toks + ["<pad>"] * (max_len_graph - len(toks))
    sentences_padded.append(padded_toks)

# 5d. Rakitkan batched_data
batched_data = {
    "sentences"         : sentences_padded,   # list[list[str]] – untuk Lang
    "length_sen"        : length_sen,          # list[int]
    "dependency_graphs" : dep_graphs_padded,   # list[np.ndarray]
    "sentic_graphs"     : sentic_graphs_padded,# list[np.ndarray]
    "sarcasms"          : MOCK_LABELS,         # list[int]
}

# Tambahkan kunci 'sentences_raw' untuk patched bridge model
batched_data["sentences_raw"] = sentences_raw   # list[str] – untuk IndoBERT

print("  batched_data keys :", list(batched_data.keys()))
print("  sentences[0] (5 token pertama):", batched_data["sentences"][0][:5], "…")
print("  length_sen             :", batched_data["length_sen"])
print("  dep_graph[0].shape     :", dep_graphs_padded[0].shape)
print("  sentic_graph[0].shape  :", sentic_graphs_padded[0].shape)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  INISIALISASI bridgeModel
#     Kita subclass bridgeModel supaya kita bisa mengirim raw sentences
#     ke tokenizer IndoBERT dan tokenized-lists ke Lang.VariablesFromSentences
#     tanpa mengubah file bridgeModel.py yang asli.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 6] Inisialisasi bridgeModel …")

from bridgeModel import bridgeModel

class PatchedBridgeModel(bridgeModel):
    """
    Subclass tipis yang meng-override gen_batch_data.

    Perbedaan dari versi asli:
      - 'sentences'     di batched_data adalah list[list[str]] (untuk Lang)
      - 'sentences_raw' di batched_data adalah list[str]        (untuk IndoBERT tokenizer)
    Ini memungkinkan kita menguji pipeline END-TO-END tanpa mengubah bridgeModel.py.
    """

    def gen_batch_data(self, batched_data: dict) -> dict:
        dict_data = {}

        # --- IndoBERT tokenization (gunakan sentences_raw) ------------------ #
        encoding = self.tokenizer(
            batched_data["sentences_raw"],          # <── raw string
            padding        = "max_length",
            truncation     = True,
            max_length     = self.max_length_sen,
            return_tensors = "pt",
        )
        dict_data["input_ids"]      = encoding["input_ids"].to(self.device)
        dict_data["attention_mask"] = encoding["attention_mask"].to(self.device)

        # --- ADGCN word-index (gunakan sentences tokenized) ----------------- #
        dict_data["sens"]    = self.lang.VariablesFromSentences(
            batched_data["sentences"],              # <── list[list[str]]
            True,
            self.device,
        )
        dict_data["len_sen"] = batched_data["length_sen"]

        # --- Graph matrices ------------------------------------------------- #
        dict_data["dependency_graph"] = torch.FloatTensor(
            batched_data["dependency_graphs"]
        ).to(self.device)
        dict_data["sentic_graph"] = torch.FloatTensor(
            batched_data["sentic_graphs"]
        ).to(self.device)

        # --- Label ---------------------------------------------------------- #
        dict_data["sarcasms"] = torch.LongTensor(
            batched_data["sarcasms"]
        ).to(self.device)

        return dict_data


# Inisialisasi model dengan vocab & embed dummy
bridge = PatchedBridgeModel(FLAGS, vocab=vocab, embed=embed_matrix)
bridge.to(FLAGS.device)

print(f"\n  Model berhasil diinisialisasi di device : {FLAGS.device}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FORWARD PASS + FOCAL LOSS + BACKWARD (stepTrain)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7] Menjalankan stepTrain() (Forward + FocalLoss + Backward) …")

loss_val, prob_np = bridge.stepTrain(batched_data, inference=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  OUTPUT LOG
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  HASIL SANITY CHECK")
print("=" * 70)

# Input dimensions
b_data_verify = bridge.gen_batch_data(batched_data)   # re-run untuk inspect

print(f"\n  ┌─ INPUT DIMENSIONS ──────────────────────────────────────────────")
print(f"  │  input_ids shape        : {tuple(b_data_verify['input_ids'].shape)}"
      f"  (batch={len(MOCK_TEXTS)}, seq_len={FLAGS.max_length_sen})")
print(f"  │  attention_mask shape   : {tuple(b_data_verify['attention_mask'].shape)}")
print(f"  │  sens (word-idx) shape  : {tuple(b_data_verify['sens'].shape)}")
print(f"  │  dep_graph shape        : {tuple(b_data_verify['dependency_graph'].shape)}")
print(f"  │  sentic_graph shape     : {tuple(b_data_verify['sentic_graph'].shape)}")
print(f"  │  labels shape           : {tuple(b_data_verify['sarcasms'].shape)}")

print(f"\n  ┌─ OUTPUT DIMENSIONS ─────────────────────────────────────────────")
print(f"  │  prob (softmax) shape   : {prob_np.shape}  (batch × n_class)")
print(f"  │  Contoh prob[0]         : {prob_np[0].round(4)}")
print(f"  │  Predicted classes      : {prob_np.argmax(axis=1).tolist()}")
print(f"  │  True labels            : {MOCK_LABELS}")

print(f"\n  ┌─ LOSS ──────────────────────────────────────────────────────────")
print(f"  │  FocalLoss (γ=2, α=0.25)  =  {loss_val:.6f}")
print(f"  │  (Scalar – backward pass berhasil tanpa error!)")

print(f"\n  ┌─ DIMENSI FUSI ──────────────────────────────────────────────────")
print(f"  │  IndoBERT base CLS      : 768  dim")
print(f"  │  ADGCN BiLSTM output    : {FLAGS.dim_hidden * 2}  dim  (dim_hidden={FLAGS.dim_hidden} × 2)")
print(f"  │  Fused (concat)         : {768 + FLAGS.dim_hidden*2}  dim  ✓")
print(f"  │  Dense 1 output         : 256  dim")
print(f"  │  Dense 2 output (logit) :   2  dim  (non-sarcasm | sarcasm)")

print("\n" + "=" * 70)
print("  ✅  SANITY CHECK LULUS – Pipeline E2E tanpa Shape Mismatch Error!")
print("  (Catatan: Menggunakan indobert-base-p1 sebagai fallback)")
print("=" * 70)
