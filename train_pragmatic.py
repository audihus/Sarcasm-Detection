# -*- coding: utf-8 -*-
"""
train_pragmatic.py
==================
Training script untuk arsitektur v2: IndoBERT + DepGCN (dep-only) + Pragmatic Channel.

Perubahan dari train_multichannel.py:
  - SarcasmDataset_v2: load dep graph saja (tanpa sentic), compute pragmatic features.
  - collate_fn_v2: hapus sentic_graphs, tambah pragmatic_features [B,6].
  - PatchedBridgeModelV2: subclass bridgeModelV2, override gen_batch_data.
  - Loss: CrossEntropyLoss(weight=[1.0, class_weight_sarcasm]) default.
  - Arg baru: --loss_type [ce_weighted|focal], --class_weight_sarcasm.
  - Output CSV per-epoch metrics ke checkpoints/.

Usage:
    python train_pragmatic.py \
        --train_data      real_data/reddit/train.csv          \
        --val_data        real_data/reddit/validation.csv     \
        --test_data       real_data/reddit/test.csv           \
        --graph_dir       real_data/reddit/                   \
        --train_split_name  train.csv                         \
        --val_split_name    validation.csv                    \
        --test_split_name   test.csv                          \
        --dataset_name    reddit                              \
        --loss_type       ce_weighted                         \
        --seed            42
"""

# ── sys.path + mock imports (sama dengan train_multichannel.py) ──────────────
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.join(SCRIPT_DIR, "multichannel-sarcasm-detection")
for _p in [SCRIPT_DIR, REPO_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from types import ModuleType, SimpleNamespace

if "models" not in sys.modules:
    _mock_models = ModuleType("models")
    sys.modules["models"] = _mock_models
    _mock_affectivegcn = ModuleType("models.affectivegcn")
    sys.modules["models.affectivegcn"] = _mock_affectivegcn
    _mock_affectivegcn.GraphConvolution = None

# ── standard imports ──────────────────────────────────────────────────────────
import argparse
import csv
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    f1_score         as sk_f1_score,
    precision_score  as sk_precision_score,
    recall_score     as sk_recall_score,
)
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# ── local imports ─────────────────────────────────────────────────────────────
from bridgeModel_v2 import bridgeModelV2
from pragmatic_features import compute_and_cache
from evaluation import evaluateClassification


# ── inlined helpers from dataUtils (avoids spaCy module-level load) ───────────

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
    cache_path = os.path.join(data_dir, f"{embed_dim}_{dataset_type}_v2_embedding_matrix.pkl")
    
    if os.path.exists(cache_path):
        print(f"[Embed] Loading cached matrix: {cache_path}")
        with open(cache_path, "rb") as f:
            cached_matrix = pickle.load(f)
            # Tambahkan validasi shape di sini
            if cached_matrix.shape[0] == len(word2idx):
                return cached_matrix
            else:
                print(f"[Embed] Mismatch! Cache size ({cached_matrix.shape[0]}) vs Vocab ({len(word2idx)}). Rebuilding...")
                # Lanjut ke bawah untuk membuat ulang matriks

    print(f"[Embed] Building embedding matrix (size={len(word2idx)}, dim={embed_dim})...")
    # ... (sisa kode di bawahnya tetap sama) ...

    print(f"[Embed] Building embedding matrix (size={len(word2idx)}, dim={embed_dim})...")
    matrix = np.zeros((len(word2idx), embed_dim), dtype=np.float64)
    matrix[1, :] = np.random.uniform(
        -1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim)
    )

    glove_candidates = [
        os.path.join(data_dir, "vectors.glove.300d.txt"),
        os.path.join(SCRIPT_DIR, "senti", "glove.840B.300d.txt"),
    ]
    word_vec = {}
    for path in glove_candidates:
        if os.path.exists(path):
            print(f"[Embed] Loading GloVe from {path} ...")
            word_vec = _load_word_vec(path, word2idx)
            break
    if not word_vec:
        print("[Embed] No GloVe file found — using zero/random init.")

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


# ── constants (sama dengan train_multichannel.py) ────────────────────────────
MAX_LEN    = 128
DIM_INPUT  = 300
DIM_HIDDEN = 256


# ── PatchedBridgeModelV2 ─────────────────────────────────────────────────────

class PatchedBridgeModelV2(bridgeModelV2):
    """
    Thin subclass yang override gen_batch_data untuk format batch v2:
        'sentences_raw'      : list[str]           -> IndoBERT tokenizer
        'sentences'          : list[list[str]]      -> Lang
        'length_sen'         : list[int]
        'dependency_graphs'  : np.ndarray [B,P,P]
        'pragmatic_features' : np.ndarray [B,6]
        'sarcasms'           : list[int]
    """

    def gen_batch_data(self, batched_data: dict) -> dict:
        dict_data = {}

        encoding = self.tokenizer(
            batched_data["sentences_raw"],
            padding        = "max_length",
            truncation     = True,
            max_length     = self.max_length_sen,
            return_tensors = "pt",
        )
        dict_data["input_ids"]      = encoding["input_ids"].to(self.device)
        dict_data["attention_mask"] = encoding["attention_mask"].to(self.device)

        dict_data["sens"]    = self.lang.VariablesFromSentences(
            batched_data["sentences"], True, self.device
        )
        dict_data["len_sen"] = batched_data["length_sen"]

        dict_data["dependency_graph"] = torch.FloatTensor(
            batched_data["dependency_graphs"]
        ).to(self.device)

        dict_data["pragmatic_features"] = torch.FloatTensor(
            batched_data["pragmatic_features"]
        ).to(self.device)

        dict_data["sarcasms"] = torch.LongTensor(
            batched_data["sarcasms"]
        ).to(self.device)

        return dict_data


# ── SarcasmDataset_v2 ─────────────────────────────────────────────────────────

class SarcasmDataset_v2(Dataset):
    """
    Dataset v2: CSV + dep graph (.graph.new) + pragmatic features (.npy cache).
    Tidak memuat file .sentic.
    """

    def __init__(self, csv_path: str, graph_dir: str, split_name: str,
                 pragmatic_cache_dir: str = None):
        df = pd.read_csv(csv_path).reset_index(drop=True)
        assert "content" in df.columns and "label" in df.columns

        dep_path = os.path.join(graph_dir, f"{split_name}.graph.new")
        with open(dep_path, "rb") as f:
            idx2dep = pickle.load(f)

        n = min(len(df), len(idx2dep))
        if n < len(df):
            print(f"[WARNING] dep count ({n}) < CSV rows ({len(df)}), truncating.")
        df = df.iloc[:n].reset_index(drop=True)

        # Pragmatic features cache
        if pragmatic_cache_dir is None:
            pragmatic_cache_dir = graph_dir
        safe_split = split_name.replace("/", "_").replace("\\", "_")
        cache_path = os.path.join(pragmatic_cache_dir, f"{safe_split}_pragmatic.npy")
        texts = df["content"].astype(str).tolist()
        prag_feats = compute_and_cache(texts, cache_path)  # [N, 6]

        self.samples = []
        for i, row in df.iterrows():
            text  = str(row["content"]).strip()
            label = int(row["label"])
            words = text.lower().split()
            self.samples.append({
                "text":      text,
                "words":     words,
                "label":     label,
                "dep_matrix": idx2dep[i],
                "prag_feat":  prag_feats[i],  # np.ndarray (6,)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── drop_edge ─────────────────────────────────────────────────────────────────

def drop_edge(matrix: np.ndarray, drop_rate: float) -> np.ndarray:
    if drop_rate <= 0.0:
        return matrix
    result = matrix.copy()
    mask   = np.random.rand(*matrix.shape) >= drop_rate
    np.fill_diagonal(mask, True)
    return result * mask.astype(np.float32)


# ── collate_fn_v2 ─────────────────────────────────────────────────────────────

def collate_fn_v2(batch: list) -> dict:
    """Collate untuk SarcasmDataset_v2 (no sentic, tambah pragmatic_features)."""
    max_graph = max(s["dep_matrix"].shape[0] for s in batch)
    pad_len   = min(max_graph, MAX_LEN)

    sentences_raw    = []
    sentences_padded = []
    length_sen       = []
    dep_list         = []
    prag_list        = []
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
        prag_list.append(s["prag_feat"])
        sarcasms.append(s["label"])

    return {
        "sentences_raw"     : sentences_raw,
        "sentences"         : sentences_padded,
        "length_sen"        : length_sen,
        "dependency_graphs" : np.stack(dep_list,  axis=0),
        "pragmatic_features": np.stack(prag_list, axis=0),  # [B, 6]
        "sarcasms"          : sarcasms,
    }


# ── build_vocab_and_embed ─────────────────────────────────────────────────────

def build_vocab_and_embed(train_dataset: SarcasmDataset_v2, args):
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
    embed = embed.astype(np.float32)
    return vocab, embed, word2idx


# ── build_flags ───────────────────────────────────────────────────────────────

def build_flags(args, device: torch.device) -> SimpleNamespace:
    use_fp16   = (not getattr(args, "no_fp16", False)) and (device.type == "cuda")
    use_focal  = (args.loss_type == "focal")
    ce_vanilla = (args.loss_type == "ce_vanilla")
    return SimpleNamespace(
        device               = device,
        max_length_sen       = MAX_LEN,
        n_class              = 2,
        learning_rate        = args.learning_rate,
        batch_size           = args.batch_size,
        t_sne                = False,
        dim_hidden           = DIM_HIDDEN,
        n_layers             = args.n_layers,
        dim_input            = DIM_INPUT,
        rnn_type             = args.rnn_type,
        bidirectional        = 1,
        embed_dropout_rate   = args.embed_dropout_rate,
        cell_dropout_rate    = args.cell_dropout_rate,
        final_dropout_rate   = args.final_dropout_rate,
        lambda1              = args.lambda1,
        weight_decay         = args.weight_decay,
        use_fp16             = use_fp16,
        use_focal            = use_focal,
        ce_vanilla           = ce_vanilla,
        class_weight_sarcasm = args.class_weight_sarcasm,
    )


# ── gradual unfreezing ────────────────────────────────────────────────────────

def freeze_encoder(model: PatchedBridgeModelV2) -> None:
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("[Freeze] IndoBERT encoder frozen.")


def unfreeze_encoder(model: PatchedBridgeModelV2) -> None:
    for param in model.model.encoder.parameters():
        param.requires_grad = True
    print("[Unfreeze] IndoBERT encoder unfrozen.")


# ── run_evaluation ────────────────────────────────────────────────────────────

def run_evaluation(model: PatchedBridgeModelV2, loader: DataLoader) -> dict:
    y_true       = []
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

        # Filter samples whose probabilities are NaN (model overflow artefact)
        finite_mask = np.isfinite(prob_np).all(axis=1)
        if not finite_mask.any():
            print("[WARN] Entire eval batch has NaN probs — skipping.")
            continue
        labels = batch["sarcasms"]
        y_true.extend(lbl for lbl, ok in zip(labels, finite_mask) if ok)
        y_pred_probs.extend(p for p, ok in zip(prob_np.tolist(), finite_mask) if ok)

    if not y_pred_probs:
        # All batches produced NaN: return dummy metrics so training can continue
        print("[WARN] run_evaluation: all predictions are NaN; returning zero metrics.")
        return {"acc": 0.0, "f1_binary": 0.0, "f1_macro": 0.0,
                "f1_micro": 0.0, "pre_macro": 0.0, "rec_macro": 0.0, "auc": 0.5}

    return evaluateClassification(y_true, y_pred_probs)


# ── train_one_epoch ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: PatchedBridgeModelV2,
    loader: DataLoader,
    epoch: int,
    drop_rate: float,
    scheduler=None,
) -> float:
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)
    for batch in pbar:
        if drop_rate > 0.0:
            batch["dependency_graphs"] = np.stack(
                [drop_edge(m, drop_rate) for m in batch["dependency_graphs"]], axis=0
            )

        try:
            loss_val, _ = model.stepTrain(batch, inference=False)
            if scheduler is not None:
                scheduler.step()
            if np.isfinite(loss_val):
                total_loss += loss_val
                n_batches  += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"\n[OOM] Skipping batch in epoch {epoch}.")
                continue
            raise

    return total_loss / max(n_batches, 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Config] loss_type={args.loss_type}  cw_sarcasm={args.class_weight_sarcasm}  seed={args.seed}")

    # ── Load datasets ──────────────────────────────────────────────────────── #
    prag_cache_dir = args.pragmatic_cache_dir or args.graph_dir
    print("[Data] Loading train...")
    full_train_ds = SarcasmDataset_v2(
        args.train_data, args.graph_dir, args.train_split_name,
        pragmatic_cache_dir=prag_cache_dir,
    )
    print("[Data] Loading test...")
    test_ds = SarcasmDataset_v2(
        args.test_data, args.graph_dir, args.test_split_name,
        pragmatic_cache_dir=prag_cache_dir,
    )

    # ── Val split ──────────────────────────────────────────────────────────── #
    if args.val_data and args.val_split_name:
        val_ds   = SarcasmDataset_v2(
            args.val_data, args.graph_dir, args.val_split_name,
            pragmatic_cache_dir=prag_cache_dir,
        )
        train_ds = full_train_ds
        print(f"[Data] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    else:
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
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_v2, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_v2, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_v2, num_workers=0,
    )

    # ── Vocab + embedding ─────────────────────────────────────────────────── #
    print("[Vocab] Building vocab and embedding from train data...")
    vocab, embed, _ = build_vocab_and_embed(full_train_ds, args)
    print(f"[Vocab] Size: {len(vocab)}")

    # ── Model ─────────────────────────────────────────────────────────────── #
    flags       = build_flags(args, device)
    bridge_model = PatchedBridgeModelV2(flags, vocab=vocab, embed=embed)

    # ── Resume ────────────────────────────────────────────────────────────── #
    start_epoch  = 1
    best_val_f1  = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"best_model_pragmatic_{args.dataset_name}_seed{args.seed}.pt")

    if args.resume and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        bridge_model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        print(f"[Resume] Epoch {ckpt['epoch']}, best F1={best_val_f1:.4f}")

    # ── Scheduler ─────────────────────────────────────────────────────────── #
    if args.no_scheduler:
        scheduler = None
    else:
        total_steps = len(train_loader) * args.epochs
        scheduler   = get_cosine_schedule_with_warmup(
            bridge_model.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )
        print(f"[Scheduler] Cosine LR, total_steps={total_steps}")

    # ── Metrics CSV ───────────────────────────────────────────────────────── #
    csv_path    = os.path.join(args.checkpoint_dir, f"{args.dataset_name}_pragmatic_seed{args.seed}_metrics.csv")
    csv_header  = ["epoch", "train_loss", "val_f1_binary", "val_acc", "val_f1_macro", "val_auc"]
    csv_rows    = []

    # ── Training loop ─────────────────────────────────────────────────────── #
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= 2:
            freeze_encoder(bridge_model)
        else:
            unfreeze_encoder(bridge_model)

        avg_loss    = train_one_epoch(bridge_model, train_loader, epoch, args.drop_rate, scheduler)
        val_metrics = run_evaluation(bridge_model, val_loader)
        val_f1      = val_metrics["f1_binary"]

        print(
            f"[Epoch {epoch:3d}/{args.epochs}]  "
            f"Loss={avg_loss:.4f}  "
            f"Val-F1-Bin={val_f1:.4f}  "
            f"Val-Acc={val_metrics['acc']:.4f}  "
            f"Patience={patience_counter}/{args.early_stopping_patience}"
        )

        csv_rows.append([
            epoch, round(avg_loss, 6),
            round(val_f1, 6), round(val_metrics["acc"], 6),
            round(val_metrics["f1_macro"], 6), round(val_metrics["auc"], 6),
        ])

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
            print(f"  [Checkpoint] Saved (F1={best_val_f1:.4f}) -> {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"[Early Stop] Epoch {epoch}. Best Val F1: {best_val_f1:.4f}")
                break

    # Save metrics CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"[CSV] Metrics saved to: {csv_path}")

    # ── Final test evaluation ─────────────────────────────────────────────── #
    print("\n[Test] Loading best checkpoint...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    bridge_model.load_state_dict(best_ckpt["model_state_dict"])

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
    f1_binary    = sk_f1_score(y_true_test, y_pred_test, average="binary", pos_label=1, zero_division=0)
    pre_binary   = sk_precision_score(y_true_test, y_pred_test, average="binary", pos_label=1, zero_division=0)
    rec_binary   = sk_recall_score(y_true_test, y_pred_test, average="binary", pos_label=1, zero_division=0)

    print("\n" + "=" * 60)
    print("  FINAL TEST RESULTS  [dep-only + pragmatic]")
    print("=" * 60)
    print(f"  Dataset         : {args.dataset_name}")
    print(f"  Loss type       : {args.loss_type}  (cw_sarcasm={args.class_weight_sarcasm})")
    print(f"  Seed            : {args.seed}")
    print(f"  Accuracy        : {test_metrics['acc']:.4f}")
    print(f"  F1-Binary(Sar)  : {f1_binary:.4f}  <- paper metric")
    print(f"  Prec-Binary     : {pre_binary:.4f}")
    print(f"  Rec-Binary      : {rec_binary:.4f}")
    print(f"  F1-Macro        : {test_metrics['f1_macro']:.4f}")
    print(f"  Precision-Macro : {test_metrics['pre_macro']:.4f}")
    print(f"  Recall-Macro    : {test_metrics['rec_macro']:.4f}")
    print(f"  F1-Micro        : {test_metrics['f1_micro']:.4f}")
    print(f"  AUC             : {test_metrics['auc']:.4f}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred, labels=[1=Sar, 0=Non]):")
    print(f"                Pred-1  Pred-0")
    print(f"  True-1 (Sar)  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  True-0 (Non)  {cm[1,0]:6d}  {cm[1,1]:6d}")
    print("=" * 60)


# ── parse_args ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train v2: IndoBERT + DepGCN + Pragmatic Channel"
    )

    # Required
    parser.add_argument("--train_data",        required=True)
    parser.add_argument("--test_data",         required=True)
    parser.add_argument("--graph_dir",         required=True,
                        help="Dir containing .graph.new files (sentic NOT required)")
    parser.add_argument("--train_split_name",  required=True)
    parser.add_argument("--test_split_name",   required=True)

    # Training
    parser.add_argument("--checkpoint_dir",    default="./checkpoints")
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--learning_rate",     type=float, default=1e-3)
    parser.add_argument("--val_split",         type=float, default=0.1)
    parser.add_argument("--val_data",          default=None)
    parser.add_argument("--val_split_name",    default=None)
    parser.add_argument("--drop_rate",         type=float, default=0.2,
                        help="DropEdge rate for dep graph (training only)")
    parser.add_argument("--early_stopping_patience",  type=int,   default=8,
                        help="Patience increased to 8 (more stable with CE loss)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01)
    parser.add_argument("--resume",            action="store_true")

    # Loss
    parser.add_argument("--loss_type",             default="ce_weighted",
                        choices=["ce_weighted", "ce_vanilla", "focal"],
                        help="ce_weighted: CE+class_weights; ce_vanilla: CE tanpa weights; focal: FocalLoss")
    parser.add_argument("--class_weight_sarcasm",  type=float, default=3.0,
                        help="Weight for sarcasm class (class 1) in CE loss")

    # Model / vocab
    parser.add_argument("--voc_size",             type=int,   default=30000)
    parser.add_argument("--n_layers",             type=int,   default=1)
    parser.add_argument("--rnn_type",             default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("--embed_dropout_rate",   type=float, default=0.5)
    parser.add_argument("--cell_dropout_rate",    type=float, default=0.5)
    parser.add_argument("--final_dropout_rate",   type=float, default=0.5)
    parser.add_argument("--lambda1",              type=float, default=0.5)
    parser.add_argument("--weight_decay",         type=float, default=0.03)

    # Embedding / dataset
    parser.add_argument("--dataset_name",         default="id_sarcasm")
    parser.add_argument("--data_dir",             default="./",
                        help="Dir to search for GloVe and write embed cache")
    parser.add_argument("--pragmatic_cache_dir",  default=None,
                        help="Dir for .npy pragmatic cache (default: graph_dir)")

    # Misc
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--no_scheduler", action="store_true")
    parser.add_argument("--no_fp16",      action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
