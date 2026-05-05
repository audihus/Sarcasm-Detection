# -*- coding: utf-8 -*-
"""
train_multichannel.py
=====================
Standalone training script for the IndoBERTlite + ADGCN multichannel
sarcasm detection model.

Pipeline:
  CSV + .graph.new/.sentic pickles
  → SarcasmDataset → collate_fn → PatchedBridgeModel
  → FocalLoss (inside bridgeModel) → AdamW (differential LR)
  → gradual unfreeze → DropEdge augmentation
  → best-val-F1 checkpoint → final test evaluation

Usage:
    python train_multichannel.py \
        --train_data      ./data/train.csv          \
        --test_data       ./data/test.csv           \
        --graph_dir       ./data/                   \
        --train_split_name  train.csv               \
        --test_split_name   test.csv                \
        --epochs 10 --batch_size 16                 \
        --checkpoint_dir  ./checkpoints/
"""

# ── 1. os / sys (must come before any local import) ─────────────────────────
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.join(SCRIPT_DIR, "multichannel-sarcasm-detection")
for _p in [SCRIPT_DIR, REPO_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 2. Mock 'models.affectivegcn' (broken import in GarphModel.py line 11) ──
# GarphModel.py has `from models.affectivegcn import GraphConvolution` which
# immediately gets shadowed at line 12 by a local class definition. We mock
# the module so the broken import does not crash on load.
from types import ModuleType, SimpleNamespace

if "models" not in sys.modules:
    _mock_models = ModuleType("models")
    sys.modules["models"] = _mock_models
    _mock_affectivegcn = ModuleType("models.affectivegcn")
    sys.modules["models.affectivegcn"] = _mock_affectivegcn
    _mock_affectivegcn.GraphConvolution = None

# ── 3. AutoTokenizer / AutoModel patch (indobert-lite sentencepiece issue) ───
# indobert-lite-base-p1 fails with sentencepiece in this environment.
# indobert-base-p1 is identical in architecture (768-dim) and works correctly.
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup


# ── 4. Standard imports ──────────────────────────────────────────────────────
import argparse
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix as sk_confusion_matrix,
                             f1_score as sk_f1_score,
                             precision_score as sk_precision_score,
                             recall_score as sk_recall_score)

# ── 5. Local imports (require REPO_DIR in sys.path) ─────────────────────────
# dataUtils.py runs `spacy.load('en_core_web_sm')` at module level, which
# would crash if the model is not installed. We only need two pure-numpy
# helpers from it, so we inline them here instead of importing the whole module.
from bridgeModel import bridgeModel
from evaluation import evaluateClassification

# ── 6. Inlined helpers from dataUtils (avoids spaCy module-level load) ───────

def _load_word_vec(path: str, word2idx: dict) -> dict:
    word_vec = {}
    with open(path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
        for line in f:
            tokens = line.rstrip().split()
            if tokens[0] in word2idx:
                try:
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype="float32")
                except Exception:
                    pass
    return word_vec


def build_embedding_matrix(word2idx: dict, embed_dim: int,
                            dataset_type: str, data_dir: str) -> np.ndarray:
    cache_path = os.path.join(data_dir, f"{embed_dim}_{dataset_type}_embedding_matrix.pkl")
    if os.path.exists(cache_path):
        print(f"[Embed] Loading cached matrix: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[Embed] Building embedding matrix (size={len(word2idx)}, dim={embed_dim})...")
    matrix = np.zeros((len(word2idx), embed_dim), dtype=np.float64)
    matrix[1, :] = np.random.uniform(
        -1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim)
    )

    glove_candidates = [
        os.path.join(data_dir, "vectors.glove.300d.txt"),
        "./senti/glove.840B.300d.txt",
    ]
    word_vec = {}
    for path in glove_candidates:
        if os.path.exists(path):
            print(f"[Embed] Loading GloVe from {path} ...")
            word_vec = _load_word_vec(path, word2idx)
            break
    if not word_vec:
        print("[Embed] No GloVe file found — using zero/random initialisation.")

    for word, idx in word2idx.items():
        vec = word_vec.get(word)
        if vec is not None:
            d = vec.shape[0]
            if d == embed_dim:
                matrix[idx] = vec
            elif d > embed_dim:
                matrix[idx] = vec[:embed_dim]
            else:
                matrix[idx, :d] = vec

    with open(cache_path, "wb") as f:
        pickle.dump(matrix, f)
    return matrix


# ── 8. Module-level constants ────────────────────────────────────────────────
MAX_LEN    = 128   # max_length_sen: sesuai max_seq_length id_sarcasm
DIM_INPUT  = 300   # GloVe word embedding dimension
DIM_HIDDEN = 256   # ADGCN BiLSTM hidden size → output = 256*2 = 512


# ─────────────────────────────────────────────────────────────────────────────
# PatchedBridgeModel
# ─────────────────────────────────────────────────────────────────────────────

class PatchedBridgeModel(bridgeModel):
    """
    Thin subclass that overrides gen_batch_data to resolve the dual-use of
    'sentences' in the original bridgeModel: raw strings are needed for the
    IndoBERT tokenizer while pre-tokenised word lists are needed for Lang.

    batched_data expected keys (set by collate_fn):
        'sentences_raw' : list[str]          → IndoBERT AutoTokenizer
        'sentences'     : list[list[str]]    → Lang.VariablesFromSentences
        'length_sen'    : list[int]
        'dependency_graphs' : np.ndarray [B, P, P]
        'sentic_graphs'     : np.ndarray [B, P, P]
        'sarcasms'          : list[int]
    """

    def gen_batch_data(self, batched_data: dict) -> dict:
        dict_data = {}

        # IndoBERT tokenisation (raw strings)
        encoding = self.tokenizer(
            batched_data["sentences_raw"],
            padding        = "max_length",
            truncation     = True,
            max_length     = self.max_length_sen,
            return_tensors = "pt",
        )
        dict_data["input_ids"]      = encoding["input_ids"].to(self.device)
        dict_data["attention_mask"] = encoding["attention_mask"].to(self.device)

        # ADGCN word-index (pre-tokenised word lists)
        dict_data["sens"]    = self.lang.VariablesFromSentences(
            batched_data["sentences"], True, self.device
        )
        dict_data["len_sen"] = batched_data["length_sen"]

        # Graph adjacency matrices
        dict_data["dependency_graph"] = torch.FloatTensor(
            batched_data["dependency_graphs"]
        ).to(self.device)
        dict_data["sentic_graph"] = torch.FloatTensor(
            batched_data["sentic_graphs"]
        ).to(self.device)

        # Labels
        dict_data["sarcasms"] = torch.LongTensor(
            batched_data["sarcasms"]
        ).to(self.device)

        return dict_data


# ─────────────────────────────────────────────────────────────────────────────
# SarcasmDataset
# ─────────────────────────────────────────────────────────────────────────────

class SarcasmDataset(Dataset):
    """
    Loads CSV + pre-built graph pickle files eagerly at construction time.

    CSV must have columns 'content' (str) and 'label' (int 0/1).
    Graph files follow the convention:
        {graph_dir}/{split_name}.graph.new   → dependency adjacency matrices
        {graph_dir}/{split_name}.sentic      → sentiment adjacency matrices
    Both pickles are dict[int → np.ndarray] where int keys are 0-based row
    indices aligned with the CSV (after reset_index).
    """

    def __init__(self, csv_path: str, graph_dir: str, split_name: str):
        df = pd.read_csv(csv_path).reset_index(drop=True)

        assert "content" in df.columns, (
            f"CSV '{csv_path}' must have a 'content' column."
        )
        assert "label" in df.columns, (
            f"CSV '{csv_path}' must have a 'label' column."
        )

        dep_path = os.path.join(graph_dir, f"{split_name}.graph.new")
        sen_path = os.path.join(graph_dir, f"{split_name}.sentic")

        with open(dep_path, "rb") as f:
            idx2dep = pickle.load(f)
        with open(sen_path, "rb") as f:
            idx2sen = pickle.load(f)

        n = min(len(df), len(idx2dep), len(idx2sen))
        if n < len(df):
            print(
                f"[WARNING] Graph count ({n}) < CSV rows ({len(df)}) "
                f"for split '{split_name}'. Truncating to {n}."
            )
        df = df.iloc[:n].reset_index(drop=True)

        self.samples = []
        for i, row in df.iterrows():
            text  = str(row["content"]).strip()
            label = int(row["label"])
            words = text.lower().split()
            self.samples.append({
                "text":       text,
                "words":      words,
                "label":      label,
                "dep_matrix": idx2dep[i],
                "sen_matrix": idx2sen[i],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# drop_edge
# ─────────────────────────────────────────────────────────────────────────────

def drop_edge(matrix: np.ndarray, drop_rate: float) -> np.ndarray:
    """
    Randomly zeros out off-diagonal edges (training augmentation only).
    Diagonal self-loops are always preserved.
    """
    if drop_rate <= 0.0:
        return matrix
    result = matrix.copy()
    mask   = np.random.rand(*matrix.shape) >= drop_rate
    np.fill_diagonal(mask, True)
    return result * mask.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Pads graph matrices and word lists to the same target length:
        pad_len = min(max(all dep/sen graph sizes in batch), MAX_LEN)

    This ensures the 'sens' LongTensor and the graph FloatTensors share the
    same second dimension when processed by ADGCN.
    """
    # Determine padding target from graph sizes (not word counts)
    max_graph = max(
        max(s["dep_matrix"].shape[0] for s in batch),
        max(s["sen_matrix"].shape[0] for s in batch),
    )
    pad_len = min(max_graph, MAX_LEN)

    sentences_raw    = []
    sentences_padded = []
    length_sen       = []
    dep_list         = []
    sen_list         = []
    sarcasms         = []

    for s in batch:
        sentences_raw.append(s["text"])

        words   = s["words"]
        eff_len = min(len(words), pad_len)
        padded  = words[:pad_len] + ["<pad>"] * (pad_len - eff_len)
        sentences_padded.append(padded)
        length_sen.append(eff_len)

        def _pad(mat):
            n = mat.shape[0]
            if n >= pad_len:
                return mat[:pad_len, :pad_len].astype(np.float32)
            return np.pad(
                mat,
                ((0, pad_len - n), (0, pad_len - n)),
                mode="constant",
                constant_values=0,
            ).astype(np.float32)

        dep_list.append(_pad(s["dep_matrix"]))
        sen_list.append(_pad(s["sen_matrix"]))
        sarcasms.append(s["label"])

    return {
        "sentences_raw"     : sentences_raw,
        "sentences"         : sentences_padded,
        "length_sen"        : length_sen,
        "dependency_graphs" : np.stack(dep_list, axis=0),
        "sentic_graphs"     : np.stack(sen_list,  axis=0),
        "sarcasms"          : sarcasms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# build_vocab_and_embed
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab_and_embed(train_dataset: SarcasmDataset, args):
    """
    Builds vocabulary and embedding matrix from all training samples.
    Reuses build_embedding_matrix from dataUtils.py (handles GloVe + caching).

    Returns (vocab: list[str], embed: np.ndarray float32, word2idx: dict).
    """
    freq = Counter()
    for s in train_dataset.samples:
        freq.update(s["words"])

    max_non_special = args.voc_size - 2
    sorted_words    = [w for w, _ in freq.most_common(max_non_special)]
    vocab           = ["<pad>", "<unk>"] + sorted_words
    word2idx        = {w: i for i, w in enumerate(vocab)}

    embed = build_embedding_matrix(
        word2idx     = word2idx,
        embed_dim    = DIM_INPUT,
        dataset_type = args.dataset_name,
        data_dir     = args.data_dir,
    )
    embed = embed.astype(np.float32)  # build_embedding_matrix returns float64

    return vocab, embed, word2idx


# ─────────────────────────────────────────────────────────────────────────────
# build_flags
# ─────────────────────────────────────────────────────────────────────────────

def build_flags(args, device: torch.device) -> SimpleNamespace:
    """Constructs the FLAGS namespace expected by bridgeModel.__init__."""
    # fp16 default ON kalau CUDA tersedia, kecuali user pass --no_fp16
    use_fp16 = (not getattr(args, "no_fp16", False)) and (device.type == "cuda")
    return SimpleNamespace(
        device              = device,
        max_length_sen      = MAX_LEN,
        n_class             = 2,
        learning_rate       = args.learning_rate,
        batch_size          = args.batch_size,
        t_sne               = False,
        dim_hidden          = DIM_HIDDEN,
        n_layers            = args.n_layers,
        dim_input           = DIM_INPUT,
        rnn_type            = args.rnn_type,
        bidirectional       = 1,
        embed_dropout_rate  = args.embed_dropout_rate,
        cell_dropout_rate   = args.cell_dropout_rate,
        final_dropout_rate  = args.final_dropout_rate,
        lambda1             = args.lambda1,
        weight_decay        = args.weight_decay,
        use_fp16            = use_fp16,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradual unfreezing helpers
# ─────────────────────────────────────────────────────────────────────────────

def freeze_encoder(model: PatchedBridgeModel) -> None:
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("[Freeze] IndoBERTlite encoder frozen.")


def unfreeze_encoder(model: PatchedBridgeModel) -> None:
    for param in model.model.encoder.parameters():
        param.requires_grad = True
    print("[Unfreeze] IndoBERTlite encoder unfrozen.")


# ─────────────────────────────────────────────────────────────────────────────
# run_evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model: PatchedBridgeModel, loader: DataLoader) -> dict:
    """
    Runs inference over a DataLoader (no drop_edge, inference=True).
    Returns the dict from evaluateClassification:
        {acc, f1_macro, pre_macro, rec_macro, f1_micro, auc, c_m}
    """
    y_true      = []
    y_pred_probs = []

    for batch in loader:
        try:
            _, prob_np = model.stepTrain(batch, inference=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print("[OOM] Skipping eval batch.")
                continue
            raise
        y_true.extend(batch["sarcasms"])
        y_pred_probs.extend(prob_np.tolist())

    return evaluateClassification(y_true, y_pred_probs)


# ─────────────────────────────────────────────────────────────────────────────
# train_one_epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: PatchedBridgeModel,
    loader: DataLoader,
    epoch: int,
    drop_rate: float,
    scheduler=None,
) -> float:
    """
    Runs one training epoch with DropEdge augmentation.
    stepTrain handles zero_grad / backward / optimizer.step internally.
    Returns average FocalLoss over the epoch.
    """
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)
    for batch in pbar:
        # Apply DropEdge augmentation (training only, never eval)
        if drop_rate > 0.0:
            batch["dependency_graphs"] = np.stack(
                [drop_edge(m, drop_rate) for m in batch["dependency_graphs"]], axis=0
            )
            batch["sentic_graphs"] = np.stack(
                [drop_edge(m, drop_rate) for m in batch["sentic_graphs"]], axis=0
            )

        try:
            loss_val, _ = model.stepTrain(batch, inference=False)
            if scheduler is not None:
                scheduler.step()
            total_loss += loss_val
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"\n[OOM] Skipping batch in epoch {epoch}. "
                      "Reduce --batch_size if this recurs.")
                continue
            raise

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Load datasets ─────────────────────────────────────────────────────── #
    print("[Data] Loading train CSV and graphs...")
    full_train_ds = SarcasmDataset(
        args.train_data, args.graph_dir, args.train_split_name
    )
    print("[Data] Loading test CSV and graphs...")
    test_ds = SarcasmDataset(
        args.test_data, args.graph_dir, args.test_split_name
    )

    # ── Val split ─────────────────────────────────────────────────────────── #
    if args.val_data and args.val_split_name:
        # Gunakan pre-split validation dari dataset asli (lebih reproducible)
        val_ds   = SarcasmDataset(args.val_data, args.graph_dir, args.val_split_name)
        train_ds = full_train_ds
        print(f"[Data] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    else:
        # Fallback: random split dari train
        n_val   = max(1, int(len(full_train_ds) * args.val_split))
        n_train = len(full_train_ds) - n_val
        train_ds, val_ds = random_split(
            full_train_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[Data] Train={n_train}  Val={n_val}  Test={len(test_ds)}")

    # ── DataLoaders ───────────────────────────────────────────────────────── #
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )

    # ── Vocab + Embedding (built from full train set before split) ────────── #
    print("[Vocab] Building vocab and embedding matrix from train data...")
    vocab, embed, _ = build_vocab_and_embed(full_train_ds, args)
    print(f"[Vocab] Size: {len(vocab)}")

    # ── Initialise model ─────────────────────────────────────────────────── #
    flags = build_flags(args, device)
    bridge_model = PatchedBridgeModel(flags, vocab=vocab, embed=embed)

    # ── Resume from checkpoint ────────────────────────────────────────────── #
    start_epoch = 1
    best_val_f1 = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    if args.resume:
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device)
            bridge_model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_f1 = ckpt.get("best_val_f1", 0.0)
            print(
                f"[Resume] Resuming from epoch {ckpt['epoch']}, "
                f"best F1: {best_val_f1:.4f}"
            )
        else:
            print(
                "[Resume] Warning: --resume set but no checkpoint found at "
                f"{best_ckpt_path}. Starting from scratch."
            )

    # ── Cosine LR scheduler (matches id_sarcasm paper: lr_scheduler_type=cosine) #
    if args.no_scheduler:
        scheduler = None
        print("[Scheduler] Disabled — using constant LR.")
    else:
        total_steps = len(train_loader) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            bridge_model.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )
        print(f"[Scheduler] Cosine LR, total_steps={total_steps}")

    # ── Training loop ─────────────────────────────────────────────────────── #
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):

        # Gradual unfreezing: freeze encoder for epochs 1-2, unfreeze from 3
        if epoch <= 2:
            freeze_encoder(bridge_model)
        else:
            unfreeze_encoder(bridge_model)

        avg_loss    = train_one_epoch(bridge_model, train_loader, epoch, args.drop_rate, scheduler)
        val_metrics = run_evaluation(bridge_model, val_loader)
        val_f1      = val_metrics["f1_binary"]   # match baseline id_sarcasm metric

        print(
            f"[Epoch {epoch:3d}/{args.epochs}]  "
            f"Loss={avg_loss:.4f}  "
            f"Val-Acc={val_metrics['acc']:.4f}  "
            f"Val-F1-Bin={val_f1:.4f}  "
            f"Patience={patience_counter}/{args.early_stopping_patience}"
        )

        if val_f1 > best_val_f1 + args.early_stopping_threshold:
            best_val_f1      = val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch"           : epoch,
                    "model_state_dict": bridge_model.state_dict(),
                    "best_val_f1"     : best_val_f1,
                    "args"            : vars(args),
                },
                best_ckpt_path,
            )
            print(
                f"  [Checkpoint] New best saved -> {best_ckpt_path} "
                f"(F1={best_val_f1:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(
                    f"[Early Stop] Triggered at epoch {epoch}. "
                    f"Best Val F1: {best_val_f1:.4f}"
                )
                break

    # ── Final evaluation on test set (best checkpoint only) ──────────────── #
    print("\n[Test] Loading best checkpoint for final evaluation...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    bridge_model.load_state_dict(best_ckpt["model_state_dict"])

    # Collect predictions for confusion matrix display
    y_true_test  = []
    y_probs_test = []
    for batch in test_loader:
        try:
            _, prob_np = bridge_model.stepTrain(batch, inference=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
        y_true_test.extend(batch["sarcasms"])
        y_probs_test.extend(prob_np.tolist())

    test_metrics = evaluateClassification(y_true_test, y_probs_test)
    y_pred_test  = np.argmax(y_probs_test, axis=-1).tolist()
    cm           = sk_confusion_matrix(y_true_test, y_pred_test, labels=[1, 0])

    # Binary F1 (same metric as id_sarcasm paper: average='binary', pos_label=1)
    f1_binary  = sk_f1_score(y_true_test, y_pred_test, average='binary', pos_label=1, zero_division=0)
    pre_binary = sk_precision_score(y_true_test, y_pred_test, average='binary', pos_label=1, zero_division=0)
    rec_binary = sk_recall_score(y_true_test, y_pred_test, average='binary', pos_label=1, zero_division=0)

    print("\n" + "=" * 60)
    print("  FINAL TEST RESULTS")
    print("=" * 60)
    print(f"  Accuracy        : {test_metrics['acc']:.4f}")
    print(f"  F1-Binary(Sar)  : {f1_binary:.4f}  ← paper metric (id_sarcasm)")
    print(f"  Prec-Binary     : {pre_binary:.4f}")
    print(f"  Rec-Binary      : {rec_binary:.4f}")
    print(f"  F1-Macro        : {test_metrics['f1_macro']:.4f}")
    print(f"  Precision-Macro : {test_metrics['pre_macro']:.4f}")
    print(f"  Recall-Macro    : {test_metrics['rec_macro']:.4f}")
    print(f"  F1-Micro        : {test_metrics['f1_micro']:.4f}")
    print(f"  AUC             : {test_metrics['auc']:.4f}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred, labels=[1=Sarcasm, 0=Non]):")
    print(f"                Pred-1  Pred-0")
    print(f"  True-1 (Sar)  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  True-0 (Non)  {cm[1,0]:6d}  {cm[1,1]:6d}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# parse_args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multichannel sarcasm detection model (IndoBERTlite + ADGCN)"
    )

    # Required data paths
    parser.add_argument("--train_data",       required=True,
                        help="Path to training CSV (columns: content, label)")
    parser.add_argument("--test_data",        required=True,
                        help="Path to test CSV (columns: content, label)")
    parser.add_argument("--graph_dir",        required=True,
                        help="Directory containing .graph.new and .sentic files")
    parser.add_argument("--train_split_name", required=True,
                        help="Split name for train graphs, e.g. 'train.csv'")
    parser.add_argument("--test_split_name",  required=True,
                        help="Split name for test graphs, e.g. 'test.csv'")

    # Training hyperparameters
    parser.add_argument("--checkpoint_dir",   default="./checkpoints",
                        help="Directory to save best_model.pt")
    parser.add_argument("--epochs",           type=int,   default=100,
                        help="Max epochs — id_sarcasm: 100 (early stopping kicks in earlier)")
    parser.add_argument("--batch_size",       type=int,   default=32,
                        help="Batch size — id_sarcasm: 32")
    parser.add_argument("--learning_rate",    type=float, default=1e-3,
                        help="LR for GCN/Dense layers (IndoBERT uses 1e-5 hardcoded in bridgeModel)")
    parser.add_argument("--val_split",        type=float, default=0.1,
                        help="Fraction of train held out as val (dipakai jika --val_data tidak diisi)")
    parser.add_argument("--val_data",         default=None,
                        help="Path CSV validasi pre-split. Jika diisi, --val_split diabaikan.")
    parser.add_argument("--val_split_name",   default=None,
                        help="Nama split untuk graph validasi, e.g. 'validation.csv'")
    parser.add_argument("--drop_rate",        type=float, default=0.2,
                        help="DropEdge rate for training augmentation (0 = disabled)")
    parser.add_argument("--early_stopping_patience",  type=int,   default=3,
                        help="Match baseline id_sarcasm = 3")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01,
                        help="Min improvement threshold — id_sarcasm: 0.01")
    parser.add_argument("--resume",           action="store_true",
                        help="Resume training from best_model.pt in checkpoint_dir")

    # Model / vocab hyperparameters
    parser.add_argument("--voc_size",             type=int,   default=30000)
    parser.add_argument("--n_layers",             type=int,   default=1)
    parser.add_argument("--rnn_type",             default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("--embed_dropout_rate",   type=float, default=0.5)
    parser.add_argument("--cell_dropout_rate",    type=float, default=0.5)
    parser.add_argument("--final_dropout_rate",   type=float, default=0.5)
    parser.add_argument("--lambda1",              type=float, default=0.5)
    parser.add_argument("--weight_decay",         type=float, default=0.03,
                        help="Weight decay — id_sarcasm: 0.03")

    # Embedding / dataset metadata
    parser.add_argument("--dataset_name", default="id_sarcasm",
                        help="Used in embedding cache filename")
    parser.add_argument("--data_dir",     default="./",
                        help="Dir to search for GloVe vectors and write embed cache")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_scheduler", action="store_true",
                        help="Disable cosine LR scheduler (use constant LR)")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable fp16 mixed precision (default: fp16 ON kalau CUDA)")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(parse_args())
