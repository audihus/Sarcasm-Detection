#!/usr/bin/env python
# coding=utf-8
"""
Baseline RNN low-parameter untuk deteksi sarkasme Indonesia (dataset Twitter IdSarcasm).
Terintegrasi dengan Self-Attention, Focal Loss, dan LR Scheduler.
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

# Pastikan module OS sudah diimport dan bisa dipakai untuk setup env
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger("run_rnn_classification")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RNN (Bi-LSTM/Bi-GRU) baseline for Indonesian sarcasm detection.")

    # --- data source ---
    p.add_argument("--use_hf_dataset", action="store_true",
                   help="Muat dari HuggingFace Hub (mirror baseline transformer) alih-alih CSV lokal.")
    p.add_argument("--dataset_name", type=str, default="w11wo/twitter_indonesia_sarcastic")
    p.add_argument("--dataset_config_name", type=str, default="default")
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--text_column_name", type=str, default="content",
                   help="Kolom teks. CSV lokal: 'content'. HF dataset: 'tweet'.")
    p.add_argument("--label_column_name", type=str, default="label")

    # --- tokenisasi / vocab ---
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=30000, help="Cap ukuran vocab (termasuk <pad>,<unk>).")
    p.add_argument("--min_freq", type=int, default=1, help="Frekuensi minimum token agar masuk vocab.")
    p.add_argument("--do_lower_case", type=lambda x: str(x).lower() != "false", default=True,
                   help="Lowercase teks sebelum split spasi (default True).")

    # --- model ---
    p.add_argument("--model_type", type=str, default="bilstm", choices=["bilstm", "bigru"])
    p.add_argument("--embedding", type=str, default="random", choices=["random", "fasttext"])
    p.add_argument("--fasttext_path", type=str, default=None,
                   help="Path cc.id.300.bin (subword OOV) atau cc.id.300.vec (OOV=nol).")
    p.add_argument("--embedding_dim", type=int, default=300,
                   help="Dim embedding utk --embedding random; fasttext mengikuti dim file.")
    p.add_argument("--freeze_embedding", action="store_true",
                   help="Bekukan embedding layer (gradien tidak mengalir ke embedding weight). "
                        "Efektif untuk fastText pretrained: turunkan overfit, kurangi trainable params.")
    p.add_argument("--hidden_size", type=int, default=128, help="Hidden size per arah (output Bi = 2x).")
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=2.0, help="Gamma parameter untuk Focal Loss.")

    # --- optimisasi ---
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--early_stopping_patience", type=int, default=20) # Default udah di-set 20 nih!
    p.add_argument("--early_stopping_threshold", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--class_weight", action="store_true",
                   help="Class-weighted utk menangani imbalance.")
    p.add_argument("--augment_file", type=str, default=None,
                   help="Path ke CSV tambahan (kolom teks & label sama dgn --text/label_column_name) "
                        "yang di-append ke training set. Contoh: real_data/reddit/train.csv")
    p.add_argument("--fp16", action="store_true", help="Mixed precision (hanya efektif di CUDA).")

    # --- protokol ---
    p.add_argument("--seeds", type=str, default="42", help="Daftar seed dipisah koma, mis. '42,1,2'.")
    p.add_argument("--metric_for_best_model", type=str, default="f1",
                   choices=["f1", "accuracy", "precision", "recall"])
    p.add_argument("--output_dir", type=str, required=True)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & tokenisasi
# ---------------------------------------------------------------------------
def load_splits(args) -> Dict[str, Tuple[List[str], List[int]]]:
    splits = {}
    if args.use_hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.dataset_name, args.dataset_config_name)
        text_col = args.text_column_name if args.text_column_name in ds["train"].column_names else "tweet"
        for split in ["train", "validation", "test"]:
            texts = [str(t) for t in ds[split][text_col]]
            labels = [int(l) for l in ds[split][args.label_column_name]]
            splits[split] = (texts, labels)
    else:
        import pandas as pd
        files = {"train": args.train_file, "validation": args.validation_file, "test": args.test_file}
        for split, path in files.items():
            if path is None:
                raise ValueError(f"--{split}_file wajib bila --use_hf_dataset tidak diset.")
            df = pd.read_csv(path)
            texts = [str(t) for t in df[args.text_column_name].tolist()]
            labels = [int(l) for l in df[args.label_column_name].tolist()]
            splits[split] = (texts, labels)
    for split, (texts, labels) in splits.items():
        pos = sum(labels)
        logger.info("Split %s: %d samples (neg %d / pos %d)", split, len(labels), len(labels) - pos, pos)
    return splits


def tokenize(text: str, do_lower: bool) -> List[str]:
    if do_lower:
        text = text.lower()
    return text.split()


def build_vocab(train_texts: List[str], do_lower: bool, min_freq: int, max_size: int) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for t in train_texts:
        counter.update(tokenize(t, do_lower))
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        if word not in vocab:
            vocab[word] = len(vocab)
    logger.info("Vocab: %d token (cap %d, min_freq %d) dari %d token unik di train.",
                len(vocab), max_size, min_freq, len(counter))
    return vocab


def encode(text: str, vocab: Dict[str, int], do_lower: bool, max_len: int) -> List[int]:
    ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(text, do_lower)][:max_len]
    if not ids:
        ids = [UNK_IDX]
    return ids


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, vocab, do_lower, max_len):
        self.samples = [encode(t, vocab, do_lower, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], len(self.samples[idx]), self.labels[idx]


def collate_batch(batch):
    seqs, lengths, labels = zip(*batch)
    max_len = max(lengths)
    padded = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Pretrained embedding (fastText)
# ---------------------------------------------------------------------------
def load_fasttext_matrix(vocab: Dict[str, int], path: str, default_dim: int) -> Tuple[np.ndarray, int]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"fastText file tidak ditemukan: {path}")

    inv_vocab = {idx: word for word, idx in vocab.items()}

    if path.endswith(".bin"):
        try:
            import fasttext  # type: ignore
        except ImportError as e:
            raise ImportError("Memuat .bin butuh `pip install fasttext`.") from e
        ft = fasttext.load_model(path)
        dim = ft.get_dimension()
        matrix = np.zeros((len(vocab), dim), dtype=np.float32)
        covered = 0
        for idx in range(len(vocab)):
            if idx == PAD_IDX:
                continue
            word = inv_vocab[idx]
            matrix[idx] = ft.get_word_vector(word)
            covered += 1
        logger.info("fastText(.bin) dim=%d: %d/%d token diisi via subword (OOV-free).",
                    dim, covered, len(vocab) - 1)
        return matrix, dim

    dim = default_dim
    matrix = None
    found = 0
    want = set(vocab.keys())
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().rstrip().split(" ")
        if len(first) == 2:
            dim = int(first[1])
        else:
            dim = len(first) - 1
            f.seek(0)
        matrix = np.zeros((len(vocab), dim), dtype=np.float32)
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in want and word not in (PAD_TOKEN, UNK_TOKEN):
                idx = vocab[word]
                vec = np.asarray(parts[1: dim + 1], dtype=np.float32)
                if vec.shape[0] == dim:
                    matrix[idx] = vec
                    found += 1
    logger.info("fastText(.vec) dim=%d: %d/%d token tercakup; sisanya VEKTOR NOL (OOV).",
                dim, found, len(vocab) - 1)
    return matrix, dim


# ---------------------------------------------------------------------------
# Custom Modules (Focal Loss & Self Attention)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, mask):
        u_t = torch.tanh(self.W(hidden_states))
        attn_weights = self.u(u_t).squeeze(-1)
        
        # Masking: abaikan token padding
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Context vector via weighted sum
        context_vector = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context_vector


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes,
                 model_type, dropout, pretrained=None, freeze_embedding=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
            with torch.no_grad():
                self.embedding.weight[PAD_IDX].zero_()
        if freeze_embedding:
            self.embedding.weight.requires_grad = False

        rnn_cls = nn.LSTM if model_type == "bilstm" else nn.GRU
        self.rnn = rnn_cls(
            embed_dim, hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        
        rnn_out_dim = hidden_size * 2
        self.attention = SelfAttention(rnn_out_dim)
        self.fc = nn.Linear(rnn_out_dim, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)

        out, _ = pad_packed_sequence(out_packed, batch_first=True)  # (B, L, 2H)
        mask = (input_ids != PAD_IDX)  # Mask untuk Self Attention
        
        context_vector = self.attention(out, mask)
        return self.fc(self.dropout(context_vector))


# ---------------------------------------------------------------------------
# Train / eval satu seed
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, loss_fct) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for input_ids, lengths, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids, lengths)
        loss = loss_fct(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        "precision": precision_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        "loss": total_loss / max(n, 1),
    }
    return metrics, preds


@torch.no_grad()
def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for input_ids, lengths, y in loader:
        input_ids = input_ids.to(device)
        logits = model(input_ids, lengths)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred, average="binary", pos_label=1, zero_division=0),
        "precision": precision_score(labels, pred, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(labels, pred, average="binary", pos_label=1, zero_division=0),
    }


def best_threshold_on_val(val_probs: np.ndarray, val_labels: np.ndarray) -> Tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(val_labels, (val_probs >= t).astype(int),
                     average="binary", pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def run_single_seed(args, seed, vocab, pretrained, embed_dim, datasets_raw, device) -> Dict:
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_texts, train_labels = datasets_raw["train"]
    val_texts, val_labels = datasets_raw["validation"]
    test_texts, test_labels = datasets_raw["test"]

    train_ds = SarcasmDataset(train_texts, train_labels, vocab, args.do_lower_case, args.max_seq_length)
    val_ds = SarcasmDataset(val_texts, val_labels, vocab, args.do_lower_case, args.max_seq_length)
    test_ds = SarcasmDataset(test_texts, test_labels, vocab, args.do_lower_case, args.max_seq_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_batch, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = RNNClassifier(
        vocab_size=len(vocab), embed_dim=embed_dim, hidden_size=args.hidden_size,
        num_layers=args.num_layers, num_classes=2, model_type=args.model_type,
        dropout=args.dropout, pretrained=pretrained,
        freeze_embedding=args.freeze_embedding,
    ).to(device)

    # Inisialisasi bobot kelas jika diminta
    weight = None
    if args.class_weight:
        classes = np.array([0, 1])
        w = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
        weight = torch.tensor(w, dtype=torch.float, device=device)
        logger.info("[seed %d] class weights (balanced): %s", seed, w.tolist())
        
    # Ganti angka 2.0 yang hardcoded dengan args.gamma
    loss_fct = FocalLoss(alpha=weight, gamma=args.gamma)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    
    # Scheduler untuk menurunkan LR jika F1 stuck
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metric = None
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        epoch_loss, n = 0.0, 0
        for input_ids, lengths, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids, lengths)
                loss = loss_fct(logits, labels)
            scaler.scale(loss).backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * labels.size(0)
            n += labels.size(0)

        val_metrics, _ = evaluate(model, val_loader, device, loss_fct)
        cur = val_metrics[args.metric_for_best_model]
        
        # Step scheduler
        scheduler.step(cur)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            "[seed %d] epoch %3d | train_loss %.4f | val_f1 %.4f acc %.4f P %.4f R %.4f | LR: %.6f",
            seed, epoch, epoch_loss / max(n, 1), val_metrics["f1"],
            val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"],
            current_lr
        )

        prev_best = best_metric
        if best_metric is None or cur > best_metric:
            best_metric = cur
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            
        if prev_best is None or (cur > prev_best and (cur - prev_best) > args.early_stopping_threshold):
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info("[seed %d] early stopping di epoch %d (patience %d).",
                            seed, epoch, args.early_stopping_patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, test_preds = evaluate(model, test_loader, device, loss_fct)

    val_probs, val_labels = predict_probs(model, val_loader, device)
    test_probs, test_labels = predict_probs(model, test_loader, device)
    best_t, val_f1_at_t = best_threshold_on_val(val_probs, val_labels)
    test_metrics_tuned = metrics_at(test_probs, test_labels, best_t)
    test_preds_tuned = (test_probs >= best_t).astype(int)

    logger.info(
        "[seed %d] BEST val_%s=%.4f @epoch %d | TEST@0.50 f1 %.4f P %.4f R %.4f | "
        "TEST@%.2f(tuned) f1 %.4f P %.4f R %.4f",
        seed, args.metric_for_best_model, best_metric or 0.0, best_epoch,
        test_metrics["f1"], test_metrics["precision"], test_metrics["recall"],
        best_t, test_metrics_tuned["f1"], test_metrics_tuned["precision"], test_metrics_tuned["recall"],
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return {
        "seed": seed,
        "best_val_metric": float(best_metric or 0.0),
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "test_preds": test_preds.tolist(),
        "best_threshold": best_t,
        "val_f1_at_best_threshold": float(val_f1_at_t),
        "test_metrics_tuned": test_metrics_tuned,
        "test_preds_tuned": test_preds_tuned.tolist(),
        "n_trainable_params": int(n_trainable),
        "n_total_params": int(n_total),
    }


# ---------------------------------------------------------------------------
# Main: multi-seed orchestration
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | model_type=%s embedding=%s",
                device, args.model_type, args.embedding)

    datasets_raw = load_splits(args)

    if args.augment_file:
        import pandas as pd
        aug_df = pd.read_csv(args.augment_file)
        aug_texts = [str(t) for t in aug_df[args.text_column_name].tolist()]
        aug_labels = [int(l) for l in aug_df[args.label_column_name].tolist()]
        orig_texts, orig_labels = datasets_raw["train"]
        datasets_raw["train"] = (orig_texts + aug_texts, orig_labels + aug_labels)
        n_total = len(orig_labels) + len(aug_labels)
        n_pos = sum(orig_labels) + sum(aug_labels)
        logger.info(
            "Augmentasi dari %s: +%d sampel -> train=%d (neg %d / pos %d, %.1f%% positif)",
            args.augment_file, len(aug_labels), n_total, n_total - n_pos, n_pos,
            100.0 * n_pos / n_total,
        )

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]

    vocab = build_vocab(datasets_raw["train"][0], args.do_lower_case, args.min_freq, args.vocab_size)

    pretrained, embed_dim = None, args.embedding_dim
    if args.embedding == "fasttext":
        if not args.fasttext_path:
            raise ValueError("--embedding fasttext butuh --fasttext_path (cc.id.300.bin atau .vec).")
        pretrained, embed_dim = load_fasttext_matrix(vocab, args.fasttext_path, args.embedding_dim)
        if embed_dim != args.embedding_dim:
            logger.info("embedding_dim mengikuti file fastText: %d (override %d).", embed_dim, args.embedding_dim)

    per_seed = []
    for seed in seeds:
        result = run_single_seed(args, seed, vocab, pretrained, embed_dim, datasets_raw, device)
        per_seed.append(result)
        
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        tm = result["test_metrics"]
        tmt = result["test_metrics_tuned"]
        with open(os.path.join(seed_dir, "eval_results.json"), "w") as f:
            json.dump({
                "eval_accuracy": tm["accuracy"], "eval_f1": tm["f1"],
                "eval_precision": tm["precision"], "eval_recall": tm["recall"],
                "eval_loss": tm["loss"], "best_val_metric": result["best_val_metric"],
                "best_epoch": result["best_epoch"], "seed": seed,
                "best_threshold": result["best_threshold"],
                "val_f1_at_best_threshold": result["val_f1_at_best_threshold"],
                "eval_accuracy_tuned": tmt["accuracy"], "eval_f1_tuned": tmt["f1"],
                "eval_precision_tuned": tmt["precision"], "eval_recall_tuned": tmt["recall"],
            }, f, indent=4)
        write_predictions(os.path.join(seed_dir, "predict_results.txt"), result["test_preds"])
        write_predictions(os.path.join(seed_dir, "predict_results_tuned.txt"), result["test_preds_tuned"])

    def agg(key, field="test_metrics"):
        vals = np.array([r[field][key] for r in per_seed], dtype=np.float64)
        return float(vals.mean()), float(vals.std())

    acc_m, acc_s = agg("accuracy")
    f1_m, f1_s = agg("f1")
    p_m, p_s = agg("precision")
    r_m, r_s = agg("recall")
    
    tacc_m, tacc_s = agg("accuracy", "test_metrics_tuned")
    tf1_m, tf1_s = agg("f1", "test_metrics_tuned")
    tp_m, tp_s = agg("precision", "test_metrics_tuned")
    tr_m, tr_s = agg("recall", "test_metrics_tuned")
    thr_vals = np.array([r["best_threshold"] for r in per_seed], dtype=np.float64)

    summary = {
        "eval_accuracy": acc_m, "eval_f1": f1_m, "eval_precision": p_m, "eval_recall": r_m,
        "eval_accuracy_std": acc_s, "eval_f1_std": f1_s,
        "eval_precision_std": p_s, "eval_recall_std": r_s,
        "eval_accuracy_tuned": tacc_m, "eval_f1_tuned": tf1_m,
        "eval_precision_tuned": tp_m, "eval_recall_tuned": tr_m,
        "eval_f1_tuned_std": tf1_s,
        "best_threshold_mean": float(thr_vals.mean()), "best_threshold_std": float(thr_vals.std()),
        "n_seeds": len(seeds), "seeds": seeds,
        "n_trainable_params": per_seed[0]["n_trainable_params"],
        "n_total_params": per_seed[0]["n_total_params"],
        "config": {
            "model_type": args.model_type, "embedding": args.embedding,
            "hidden_size": args.hidden_size, "num_layers": args.num_layers,
            "embedding_dim": embed_dim, "max_seq_length": args.max_seq_length,
            "vocab_size": len(vocab), "learning_rate": args.learning_rate,
            "batch_size": args.batch_size, "class_weight": args.class_weight,
            "do_lower_case": args.do_lower_case, "freeze_embedding": args.freeze_embedding,
            "augment_file": args.augment_file,
        },
        "per_seed": [
            {"seed": r["seed"], "best_val_metric": r["best_val_metric"],
             "best_epoch": r["best_epoch"], "best_threshold": r["best_threshold"],
             **{f"test_{k}": v for k, v in r["test_metrics"].items()},
             **{f"test_{k}_tuned": v for k, v in r["test_metrics_tuned"].items()}}
            for r in per_seed
        ],
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    write_predictions(os.path.join(args.output_dir, "predict_results.txt"), per_seed[0]["test_preds"])

    logger.info("=" * 78)
    logger.info("HASIL TEST (%s, %s) — %d seed %s", args.model_type, args.embedding, len(seeds), seeds)
    logger.info("  -- threshold 0.50 / argmax (apple-to-apple) --")
    logger.info("  F1 (positif)  : %.4f +/- %.4f", f1_m, f1_s)
    logger.info("  Accuracy      : %.4f +/- %.4f", acc_m, acc_s)
    logger.info("  Precision     : %.4f +/- %.4f", p_m, p_s)
    logger.info("  Recall        : %.4f +/- %.4f", r_m, r_s)
    logger.info("  -- threshold tuned @val (add-on; t=%.2f +/- %.2f) --",
                float(thr_vals.mean()), float(thr_vals.std()))
    logger.info("  F1 (positif)  : %.4f +/- %.4f", tf1_m, tf1_s)
    logger.info("  Accuracy      : %.4f +/- %.4f", tacc_m, tacc_s)
    logger.info("  Precision     : %.4f +/- %.4f", tp_m, tp_s)
    logger.info("  Recall        : %.4f +/- %.4f", tr_m, tr_s)
    logger.info("  Trainable params: %s (total %s)",
                f"{per_seed[0]['n_trainable_params']:,}", f"{per_seed[0]['n_total_params']:,}")
    logger.info("  Output -> %s", os.path.join(args.output_dir, "eval_results.json"))
    logger.info("=" * 78)


def write_predictions(path: str, preds: List[int]):
    with open(path, "w") as f:
        f.write("index\tprediction\n")
        for i, item in enumerate(preds):
            f.write(f"{i}\t{int(item)}\n")


if __name__ == "__main__":
    main()