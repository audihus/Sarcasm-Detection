#!/usr/bin/env python3
"""
Late fusion sarcasm detection: numeric features are concatenated directly to the
BERT CLS (last_hidden_state[:, 0, :]) embedding before the classification head.

Architecture:
    text  -> IndoBERT -> CLS hidden state (768)  ─┐
                                                  ├─ cat -> Linear(768+D, 2) -> logits
    features (D dims)  ───────────────────────────┘

Reddit  features (D=3): [word_count, sentence_count, avg_sentence_length]
                         z-score      z-score          z-score
Twitter features (D=3): [is_clash, question_count, has_hyperbole]
                         binary     z-score norm    binary
Twitter features (D=4, --use_contrastive_feature):
                        [is_clash, question_count, has_hyperbole, has_contrastive_conj]

Kaggle commands
---------------
SEL 1 — Install:
    !pip install datasets transformers scikit-learn PySastrawi -q

SEL 2 — Reddit late fusion (identik hyperparameter dengan baseline recipes):
    !python scripts/run_classification_fusion.py \
        --dataset_name reddit \
        --model_name indobenchmark/indobert-base-p1 \
        --output_dir /kaggle/working/outputs/indobert-base-p1-reddit-indonesia-sarcastic-fusion \
        --num_epochs 100 --batch_size 32 --learning_rate 1e-5 \
        --weight_decay 0.03 --lr_scheduler_type cosine \
        --shuffle_train_dataset --seed 42 --fp16

SEL 3 — Twitter late fusion (sesuaikan slug dataset Kaggle untuk path InSet):
    !python scripts/run_classification_fusion.py \
        --dataset_name twitter \
        --model_name indobenchmark/indobert-base-p1 \
        --output_dir /kaggle/working/outputs/indobert-base-p1-twitter-indonesia-sarcastic-fusion \
        --num_epochs 100 --batch_size 32 --learning_rate 1e-5 \
        --weight_decay 0.03 --lr_scheduler_type cosine \
        --shuffle_train_dataset --seed 42 --fp16 \
        --inset_pos_path /kaggle/input/id-sarcasm-data/real_data/twitter/positive.tsv \
        --inset_neg_path /kaggle/input/id-sarcasm-data/real_data/twitter/negative.tsv

SEL 4 — Multi-seed Twitter + fitur konjungsi:
    !python scripts/run_classification_fusion.py \
        --dataset_name twitter \
        --model_name indobenchmark/indobert-base-p1 \
        --output_dir /kaggle/working/outputs/twitter-contrastive-multiseed \
        --num_epochs 100 --batch_size 32 --learning_rate 1e-5 \
        --weight_decay 0.03 --lr_scheduler_type cosine \
        --shuffle_train_dataset --fp16 \
        --seeds "42,1,2,3,4" --use_contrastive_feature \
        --inset_pos_path /kaggle/input/id-sarcasm-data/real_data/twitter/positive.tsv \
        --inset_neg_path /kaggle/input/id-sarcasm-data/real_data/twitter/negative.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, set_seed

# Allow importing preprocessing/ from the project root (one level above scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.augment_pipeline import (
    HYPERBOLE_WORDS,
    detect_polarity_clash,
    load_inset_lexicon,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_CONFIG: Dict[str, Dict] = {
    "reddit": {
        "hub_name": "w11wo/reddit_indonesia_sarcastic",
        "text_col": "text",
    },
    "twitter": {
        "hub_name": "w11wo/twitter_indonesia_sarcastic",
        "text_col": "tweet",
    },
}

_BASELINE_HEAD_PARAMS = 768 * 2 + 2  # Linear(768, 2)

# Word-boundary regex untuk konjungsi pertentangan/konsesif bahasa Indonesia.
# Clash + konjungsi ini cenderung rekonsiliasi tulus, bukan sarkasme → fitur menurunkan
# kepercayaan head pada clash untuk kasus tersebut.
_CONTRASTIVE_CONJ_RE = re.compile(
    r"\b(tapi|tetapi|namun|walau|walaupun|meski|meskipun|kendati|padahal|sekalipun|biarpun)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SarcasmModelWithFeatures(nn.Module):
    """IndoBERT encoder + numeric feature concatenation + MLP classification head."""

    def __init__(self, model_name: str, feature_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # MLP Head untuk Late Fusion yang lebih robust (Anti-Overfit)
        self.classifier = nn.Sequential(
            nn.Linear(768 + feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Stabilisasi distribusi fitur
            nn.ReLU(),                  # Fungsi aktivasi non-linear
            nn.Dropout(dropout_rate),   # Regularisasi anti-overfit
            nn.Linear(hidden_dim, 2)
        )

        # Inisialisasi bobot khusus untuk layer Linear di dalam MLP
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        # Raw CLS token from the last hidden state, shape (B, 768)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]

        # Gabungkan text embedding dengan fitur leksikal
        combined = torch.cat([cls_emb, features], dim=1)  # (B, 768 + feature_dim)

        # Lewatkan ke MLP head
        return self.classifier(combined)                  # (B, 2)


# ---------------------------------------------------------------------------
# Auxiliary-clash model
# ---------------------------------------------------------------------------

class SarcasmModelAuxClash(nn.Module):
    """IndoBERT encoder + dual head: sarcasm (main) + clash (auxiliary).

    Tidak menerima fitur leksikal — input murni teks (768 CLS).
    forward() mengembalikan (logits_sarcasm, logits_clash).
    """

    def __init__(self, model_name: str, hidden_dim: int = 256, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        self.head_main = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )
        self.head_aux = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )
        for module in list(self.head_main) + list(self.head_aux):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_emb = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return self.head_main(cls_emb), self.head_aux(cls_emb)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SarcasmDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: np.ndarray,
        labels: List[int],
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "features": self.features[idx],
            "labels": self.labels[idx],
        }


class SarcasmDatasetAux(Dataset):
    """Dataset untuk mode auxiliary clash — tanpa fitur leksikal, dengan clash_labels."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        clash_labels: List[int],
        labels: List[int],
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.clash_labels = torch.tensor(clash_labels, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids":    self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "clash_labels": self.clash_labels[idx],
            "labels":       self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    texts: List[str],
    dataset_name: str,
    inset_pos: Optional[frozenset] = None,
    inset_neg: Optional[frozenset] = None,
    feature_stats: Optional[dict] = None,
    is_train: bool = False,
    use_contrastive_feature: bool = False,
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Build numeric feature matrix.

    Reddit  -> shape (N, 3): [word_count, sentence_count, avg_sentence_length]
                              all z-score normalized
    Twitter -> shape (N, 3): [is_clash, question_count_zscore, has_hyperbole]
               or (N, 4) with --use_contrastive_feature:
                             [is_clash, question_count_zscore, has_hyperbole, has_contrastive_conj]

    When is_train=True, mean/std are computed from `texts` and returned in
    feature_stats. For val/test, pass the stats computed on the training set.
    Only non-binary features are z-score normalized.
    """
    if dataset_name == "reddit":
        word_counts: List[float] = []
        sentence_counts: List[float] = []
        avg_lengths: List[float] = []

        for text in texts:
            wc = float(len(text.split()))
            # sentence boundaries: '.', '!', '?'
            parts = [p.strip() for p in re.split(r'[.!?]+', text) if p.strip()]
            sc = float(max(len(parts), 1))
            word_counts.append(wc)
            sentence_counts.append(sc)
            avg_lengths.append(wc / sc)

        wc_arr  = np.array(word_counts,    dtype=float)
        sc_arr  = np.array(sentence_counts, dtype=float)
        asl_arr = np.array(avg_lengths,     dtype=float)

        if is_train:
            feature_stats = {
                "word_count_mean":         float(wc_arr.mean()),
                "word_count_std":          float(wc_arr.std()),
                "sentence_count_mean":     float(sc_arr.mean()),
                "sentence_count_std":      float(sc_arr.std()),
                "avg_sentence_length_mean": float(asl_arr.mean()),
                "avg_sentence_length_std":  float(asl_arr.std()),
            }
            wc_n  = (wc_arr  - feature_stats["word_count_mean"])         / (feature_stats["word_count_std"]          + 1e-8)
            sc_n  = (sc_arr  - feature_stats["sentence_count_mean"])     / (feature_stats["sentence_count_std"]      + 1e-8)
            asl_n = (asl_arr - feature_stats["avg_sentence_length_mean"]) / (feature_stats["avg_sentence_length_std"] + 1e-8)
        else:
            wc_n  = (wc_arr  - feature_stats["word_count_mean"])         / (feature_stats["word_count_std"]          + 1e-8)
            sc_n  = (sc_arr  - feature_stats["sentence_count_mean"])     / (feature_stats["sentence_count_std"]      + 1e-8)
            asl_n = (asl_arr - feature_stats["avg_sentence_length_mean"]) / (feature_stats["avg_sentence_length_std"] + 1e-8)

        features = np.stack([wc_n, sc_n, asl_n], axis=1)
        return features, feature_stats

    elif dataset_name == "twitter":
        is_clash_list: List[float] = []
        question_counts: List[float] = []
        has_hyperbole_list: List[float] = []

        for text in texts:
            clash, _ = detect_polarity_clash(text, inset_pos, inset_neg)
            is_clash_list.append(1.0 if clash else 0.0)

            question_counts.append(float(min(text.count("?"), 3)))

            text_lower = text.lower()
            hyper = 1.0 if any(hw in text_lower for hw in HYPERBOLE_WORDS) else 0.0
            has_hyperbole_list.append(hyper)

        q_arr = np.array(question_counts, dtype=float)
        if is_train:
            mean = float(q_arr.mean())
            std = float(q_arr.std())
            feature_stats = {"question_count_mean": mean, "question_count_std": std}
        else:
            mean = feature_stats["question_count_mean"]
            std = feature_stats["question_count_std"]
        q_normalized = (q_arr - mean) / (std + 1e-8)

        cols = [
            np.array(is_clash_list, dtype=float),
            q_normalized,
            np.array(has_hyperbole_list, dtype=float),
        ]

        if use_contrastive_feature:
            conj_flags = np.array(
                [1.0 if _CONTRASTIVE_CONJ_RE.search(t) else 0.0 for t in texts],
                dtype=float,
            )
            cols.append(conj_flags)

        features = np.stack(cols, axis=1)
        return features, feature_stats

    else:
        raise ValueError(f"dataset_name harus 'reddit' atau 'twitter', dapat: {dataset_name!r}")


def compute_clash_labels(
    texts: List[str],
    inset_pos: frozenset,
    inset_neg: frozenset,
) -> List[int]:
    """Turunkan label clash biner (1=clash, 0=tidak) untuk seluruh list teks."""
    return [1 if detect_polarity_clash(t, inset_pos, inset_neg)[0] else 0 for t in texts]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_probs: bool = False,
) -> Tuple:
    """Returns (f1_binary, predictions, true_labels[, probs_class1])."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["features"].to(device),
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
            if return_probs:
                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
    f1 = f1_score(all_labels, all_preds, average="binary")
    if return_probs:
        return f1, all_preds, all_labels, all_probs
    return f1, all_preds, all_labels


def run_eval_aux(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_probs: bool = False,
) -> Tuple:
    """Evaluasi sarcasm head dari SarcasmModelAuxClash. Signature identik run_eval."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for batch in loader:
            logits_sar, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_preds.extend(logits_sar.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
            if return_probs:
                probs = torch.softmax(logits_sar, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
    f1 = f1_score(all_labels, all_preds, average="binary")
    if return_probs:
        return f1, all_preds, all_labels, all_probs
    return f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Per-seed training
# ---------------------------------------------------------------------------

def train_one_seed(
    seed: int,
    args: argparse.Namespace,
    train_ds: SarcasmDataset,
    val_ds: SarcasmDataset,
    test_ds: SarcasmDataset,
    val_texts: List[str],
    test_texts: List[str],
    feature_dim: int,
    device: torch.device,
    seed_output_dir: Path,
) -> dict:
    """Full training + evaluation for one seed. Returns result dict with F1/P/R/FP/FN."""
    # Reset ALL RNG sources at the top of every seed iteration.
    seed_everything(seed)
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    # Recreate train_loader with a freshly seeded generator.
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train_dataset,
        generator=g,
        drop_last=True,  # prevents BatchNorm crash on single-sample last batch
    )
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = SarcasmModelWithFeatures(args.model_name, feature_dim).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    bert_params  = sum(p.numel() for p in model.bert.parameters())
    head_params  = sum(p.numel() for p in model.classifier.parameters())
    print(f"  Total parameters        : {total_params:,}")
    print(f"  BERT parameters         : {bert_params:,}")
    print(f"  Classification head     : {head_params:,} "
          f"(vs baseline {_BASELINE_HEAD_PARAMS} → +{head_params - _BASELINE_HEAD_PARAMS})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_steps = args.num_epochs * len(train_loader)
    if args.lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    else:
        scheduler = None

    use_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None
    if use_fp16:
        print("  fp16 mixed precision enabled")

    best_model_path = seed_output_dir / "best_model.pt"
    best_f1 = 0.0
    patience_counter = 0

    print(f"  Training for up to {args.num_epochs} epochs (patience=3)...")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            feat = batch["features"].to(device)
            lbl  = batch["labels"].to(device)

            if scaler:
                with autocast():
                    logits = model(ids, mask, feat)
                    loss = criterion(logits, lbl)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(ids, mask, feat)
                loss = criterion(logits, lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_f1, val_preds, val_true = run_eval(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_preds)
        val_metric = val_f1 if args.metric_for_best_model == "f1" else val_acc
        print(
            f"  Epoch {epoch:2d}/{args.num_epochs} | loss={avg_loss:.4f} "
            f"| val_{args.metric_name}={val_metric:.4f}",
            end="",
        )

        if val_metric > best_f1:
            best_f1 = val_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(" [best]")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/3)")
            if patience_counter >= 3:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    # Load best checkpoint for final evaluation.
    print(f"  Loading best checkpoint (val_{args.metric_for_best_model}={best_f1:.4f})...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Save validation predictions for offline error analysis.
    _, val_preds_best, val_true_best, val_probs_best = run_eval(
        model, val_loader, device, return_probs=True
    )
    val_pred_data = {
        "texts":  val_texts,
        "preds":  val_preds_best,
        "labels": val_true_best,
        "probs":  [round(p, 6) for p in val_probs_best],
    }
    with open(seed_output_dir / "val_predictions.json", "w", encoding="utf-8") as f:
        json.dump(val_pred_data, f, ensure_ascii=False, indent=2)

    # Test set evaluation.
    test_f1, test_preds, test_true, test_probs_best = run_eval(
        model, test_loader, device, return_probs=True
    )
    test_acc = accuracy_score(test_true, test_preds)
    test_pre = precision_score(test_true, test_preds, average="binary")
    test_rec = recall_score(test_true, test_preds, average="binary")
    fp = sum(1 for p, l in zip(test_preds, test_true) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(test_preds, test_true) if p == 0 and l == 1)

    result = {
        "seed":      seed,
        "f1":        test_f1,
        "accuracy":  test_acc,
        "precision": test_pre,
        "recall":    test_rec,
        "fp":        fp,
        "fn":        fn,
    }

    with open(seed_output_dir / "eval_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save test probabilities — format identik val_predictions.json.
    test_pred_data = {
        "texts":  test_texts,
        "preds":  test_preds,
        "labels": test_true,
        "probs":  [round(p, 6) for p in test_probs_best],
    }
    with open(seed_output_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_pred_data, f, ensure_ascii=False, indent=2)

    with open(seed_output_dir / "predict_results.txt", "w") as f:
        f.write("index\tprediction\n")
        for idx, pred in enumerate(test_preds):
            f.write(f"{idx}\t{pred}\n")

    print(
        f"  [seed={seed}] Test F1={test_f1:.4f} | P={test_pre:.4f} "
        f"| R={test_rec:.4f} | FP={fp} | FN={fn}"
    )
    return result


def train_one_seed_aux(
    seed: int,
    args: argparse.Namespace,
    train_ds: SarcasmDatasetAux,
    val_ds: SarcasmDatasetAux,
    test_ds: SarcasmDatasetAux,
    val_texts: List[str],
    test_texts: List[str],
    device: torch.device,
    seed_output_dir: Path,
) -> dict:
    """Training + evaluasi satu seed, mode auxiliary clash. Returns sama dengan train_one_seed."""
    seed_everything(seed)
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=args.shuffle_train_dataset, generator=g,
        drop_last=True,
    )
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = SarcasmModelAuxClash(args.model_name).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    bert_params  = sum(p.numel() for p in model.bert.parameters())
    head_params  = (sum(p.numel() for p in model.head_main.parameters())
                    + sum(p.numel() for p in model.head_aux.parameters()))
    print(f"  Total parameters     : {total_params:,}")
    print(f"  BERT parameters      : {bert_params:,}")
    print(f"  Head parameters      : {head_params:,} (main + aux)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_steps = args.num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps) if args.lr_scheduler_type == "cosine" else None

    use_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None
    if use_fp16:
        print("  fp16 mixed precision enabled")

    best_model_path = seed_output_dir / "best_model.pt"
    best_f1 = 0.0
    patience_counter = 0

    print(f"  Training for up to {args.num_epochs} epochs (patience=3, aux_lambda={args.aux_lambda})...")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids       = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            lbl       = batch["labels"].to(device)
            clash_lbl = batch["clash_labels"].to(device)

            if scaler:
                with autocast():
                    logits_sar, logits_clash = model(ids, mask)
                    loss = (criterion(logits_sar, lbl)
                            + args.aux_lambda * criterion(logits_clash, clash_lbl))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits_sar, logits_clash = model(ids, mask)
                loss = (criterion(logits_sar, lbl)
                        + args.aux_lambda * criterion(logits_clash, clash_lbl))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_f1, val_preds, val_true = run_eval_aux(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_preds)
        val_metric = val_f1 if args.metric_for_best_model == "f1" else val_acc
        print(
            f"  Epoch {epoch:2d}/{args.num_epochs} | loss={avg_loss:.4f} "
            f"| val_{args.metric_name}={val_metric:.4f}",
            end="",
        )

        if val_metric > best_f1:
            best_f1 = val_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(" [best]")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/3)")
            if patience_counter >= 3:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    print(f"  Loading best checkpoint (val_{args.metric_for_best_model}={best_f1:.4f})...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Save validation predictions.
    _, val_preds_best, val_true_best, val_probs_best = run_eval_aux(
        model, val_loader, device, return_probs=True
    )
    with open(seed_output_dir / "val_predictions.json", "w", encoding="utf-8") as f:
        json.dump(
            {"texts": val_texts, "preds": val_preds_best,
             "labels": val_true_best, "probs": [round(p, 6) for p in val_probs_best]},
            f, ensure_ascii=False, indent=2,
        )

    # Test set evaluation.
    test_f1, test_preds, test_true, test_probs_best = run_eval_aux(
        model, test_loader, device, return_probs=True
    )
    test_acc = accuracy_score(test_true, test_preds)
    test_pre = precision_score(test_true, test_preds, average="binary")
    test_rec = recall_score(test_true, test_preds, average="binary")
    fp = sum(1 for p, l in zip(test_preds, test_true) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(test_preds, test_true) if p == 0 and l == 1)

    result = {
        "seed": seed, "f1": test_f1, "accuracy": test_acc,
        "precision": test_pre, "recall": test_rec, "fp": fp, "fn": fn,
    }
    with open(seed_output_dir / "eval_results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(seed_output_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(
            {"texts": test_texts, "preds": test_preds,
             "labels": test_true, "probs": [round(p, 6) for p in test_probs_best]},
            f, ensure_ascii=False, indent=2,
        )
    with open(seed_output_dir / "predict_results.txt", "w") as f:
        f.write("index\tprediction\n")
        for idx, pred in enumerate(test_preds):
            f.write(f"{idx}\t{pred}\n")

    print(
        f"  [seed={seed}] Test F1={test_f1:.4f} | P={test_pre:.4f} "
        f"| R={test_rec:.4f} | FP={fp} | FN={fn}"
    )
    return result


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Late fusion sarcasm detection")
    parser.add_argument("--dataset_name", required=True, choices=["reddit", "twitter"],
                        help="'reddit' atau 'twitter'")
    parser.add_argument("--model_name", default="indobenchmark/indobert-base-p1",
                        help="HuggingFace model ID untuk encoder")
    parser.add_argument("--output_dir", required=True,
                        help="Direktori untuk menyimpan output")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Panjang token maksimal (default: 128)")
    parser.add_argument("--metric_name", default="f1",
                        choices=["f1", "accuracy"],
                        help="Metrik evaluasi utama (default: f1)")
    parser.add_argument("--metric_for_best_model", default="f1",
                        choices=["f1", "accuracy"],
                        help="Metrik untuk memilih best model (default: f1)")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Jumlah epoch maksimal (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size training (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="AdamW learning rate (default: 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.03,
                        help="AdamW weight decay (default: 0.03)")
    parser.add_argument("--lr_scheduler_type", default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="Tipe LR scheduler (default: cosine)")
    parser.add_argument("--shuffle_train_dataset", action="store_true",
                        help="Shuffle dataset training setiap epoch")
    parser.add_argument("--fp16", action="store_true",
                        help="Mixed precision training (fp16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed untuk single run (default: 42); diabaikan jika --seeds diisi")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds untuk multi-seed run, mis. '42,1,2,3,4'. "
                             "Jika diisi, --seed diabaikan.")
    parser.add_argument("--inset_pos_path", default=None,
                        help="Path ke positive.tsv InSet (wajib untuk Twitter)")
    parser.add_argument("--inset_neg_path", default=None,
                        help="Path ke negative.tsv InSet (wajib untuk Twitter)")
    parser.add_argument("--use_contrastive_feature", action="store_true",
                        help="Tambah fitur biner konjungsi pertentangan (Twitter only). "
                             "Saat aktif, feature_dim Twitter naik dari 3 → 4.")
    parser.add_argument("--no_features", action="store_true",
                        help="Jalankan plain IndoBERT tanpa fitur leksikal (feature_dim=0). "
                             "InSet tidak diperlukan. Digunakan sebagai baseline tandingan.")
    parser.add_argument("--auxiliary_clash", action="store_true",
                        help="Mode auxiliary task: encoder dilatih sekaligus memprediksi sarkasme "
                             "dan clash. Hanya Twitter. Tidak kompatibel dengan --no_features.")
    parser.add_argument("--aux_lambda", type=float, default=0.3,
                        help="Bobot loss auxiliary clash (default: 0.3). "
                             "loss_total = loss_sarcasm + aux_lambda * loss_clash.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine seeds list. --seeds overrides --seed.
    if args.seeds is not None:
        seeds_list = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds_list = [args.seed]
    print(f"Seeds: {seeds_list}")

    cfg = DATASET_CONFIG[args.dataset_name]
    text_col: str = cfg["text_col"]

    # ------------------------------------------------------------------
    # 1. Load dataset  (once, shared across all seeds)
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading dataset: {args.dataset_name}")
    hub_name = cfg["hub_name"]
    if os.path.isdir(hub_name):
        raw = load_from_disk(hub_name)
    else:
        raw = load_dataset(hub_name)

    def get_texts(split: str) -> List[str]:
        return [str(x) for x in raw[split][text_col]]

    def get_labels(split: str) -> List[int]:
        col = "label" if "label" in raw[split].column_names else raw[split].column_names[-1]
        return [int(x) for x in raw[split][col]]

    train_texts = get_texts("train");      train_labels = get_labels("train")
    val_texts   = get_texts("validation"); val_labels   = get_labels("validation")
    test_texts  = get_texts("test");       test_labels  = get_labels("test")

    print(f"  Train: {len(train_texts):,}  Val: {len(val_texts):,}  Test: {len(test_texts):,}")

    # ------------------------------------------------------------------
    # 2. Load InSet lexicon (Twitter only, once)
    # ------------------------------------------------------------------
    if args.auxiliary_clash and args.dataset_name != "twitter":
        raise ValueError("--auxiliary_clash hanya didukung untuk --dataset_name twitter.")
    if args.auxiliary_clash and args.no_features:
        raise ValueError("--auxiliary_clash dan --no_features tidak bisa digunakan bersamaan.")

    inset_pos: Optional[frozenset] = None
    inset_neg: Optional[frozenset] = None
    if args.dataset_name == "twitter" and (not args.no_features or args.auxiliary_clash):
        if not args.inset_pos_path or not args.inset_neg_path:
            raise ValueError(
                "Twitter membutuhkan --inset_pos_path dan --inset_neg_path"
            )
        print("\n[2/6] Loading InSet lexicon...")
        inset_pos, inset_neg = load_inset_lexicon(args.inset_pos_path, args.inset_neg_path)
    else:
        print("\n[2/6] InSet tidak diperlukan, dilewati.")

    # ------------------------------------------------------------------
    # 3. Feature extraction (once; feature_dim derived dynamically)
    # ------------------------------------------------------------------
    if args.no_features:
        print("\n[3/6] --no_features aktif: melewati ekstraksi fitur (feature_dim=0).")
        train_features = np.zeros((len(train_texts), 0), dtype=float)
        val_features   = np.zeros((len(val_texts),   0), dtype=float)
        test_features  = np.zeros((len(test_texts),  0), dtype=float)
        feature_stats: dict = {}
    elif args.auxiliary_clash:
        print("\n[3/6] --auxiliary_clash: menurunkan label clash (tidak ada fitur input).")
        train_clash_labels = compute_clash_labels(train_texts, inset_pos, inset_neg)
        val_clash_labels   = compute_clash_labels(val_texts,   inset_pos, inset_neg)
        test_clash_labels  = compute_clash_labels(test_texts,  inset_pos, inset_neg)
        train_features = np.zeros((len(train_texts), 0), dtype=float)  # dummy untuk feature_dim
        feature_stats = {}
        print(f"  Clash labels — train: {sum(train_clash_labels):,} clash / {len(train_clash_labels):,} total")
    else:
        print("\n[3/6] Extracting features...")
        train_features, feature_stats = extract_features(
            train_texts, args.dataset_name,
            inset_pos=inset_pos, inset_neg=inset_neg,
            is_train=True,
            use_contrastive_feature=args.use_contrastive_feature,
        )
        val_features, _ = extract_features(
            val_texts, args.dataset_name,
            inset_pos=inset_pos, inset_neg=inset_neg,
            feature_stats=feature_stats,
            use_contrastive_feature=args.use_contrastive_feature,
        )
        test_features, _ = extract_features(
            test_texts, args.dataset_name,
            inset_pos=inset_pos, inset_neg=inset_neg,
            feature_stats=feature_stats,
            use_contrastive_feature=args.use_contrastive_feature,
        )

    # Dynamic feature_dim dari shape aktual (0 untuk aux/no_features mode).
    feature_dim: int = train_features.shape[1]
    print(f"  Feature stats: {feature_stats}")
    if not args.auxiliary_clash:
        print(f"  Train features shape: {train_features.shape}  (feature_dim={feature_dim})")

    with open(output_dir / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)

    # ------------------------------------------------------------------
    # 4. Tokenization (once)
    # ------------------------------------------------------------------
    print(f"\n[4/6] Tokenizing with {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(texts: List[str]) -> Dict:
        return tokenizer(
            texts,
            padding="max_length",
            max_length=args.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

    train_enc = tokenize(train_texts)
    val_enc   = tokenize(val_texts)
    test_enc  = tokenize(test_texts)

    # ------------------------------------------------------------------
    # 5. Build Dataset objects (once; DataLoaders re-created per seed)
    # ------------------------------------------------------------------
    if args.auxiliary_clash:
        train_ds = SarcasmDatasetAux(train_enc["input_ids"], train_enc["attention_mask"], train_clash_labels, train_labels)
        val_ds   = SarcasmDatasetAux(val_enc["input_ids"],   val_enc["attention_mask"],   val_clash_labels,   val_labels)
        test_ds  = SarcasmDatasetAux(test_enc["input_ids"],  test_enc["attention_mask"],  test_clash_labels,  test_labels)
    else:
        train_ds = SarcasmDataset(train_enc["input_ids"], train_enc["attention_mask"], train_features, train_labels)
        val_ds   = SarcasmDataset(val_enc["input_ids"],   val_enc["attention_mask"],   val_features,   val_labels)
        test_ds  = SarcasmDataset(test_enc["input_ids"],  test_enc["attention_mask"],  test_features,  test_labels)

    # ------------------------------------------------------------------
    # 6. Train one run per seed
    # ------------------------------------------------------------------
    multi_seed = len(seeds_list) > 1
    all_results: List[dict] = []

    for seed in seeds_list:
        print(f"\n{'='*60}")
        print(f"[5/6] Initializing model — seed={seed}  feature_dim={feature_dim}")
        print(f"{'='*60}")

        seed_dir = output_dir / f"seed_{seed}" if multi_seed else output_dir
        if args.auxiliary_clash:
            result = train_one_seed_aux(
                seed=seed, args=args,
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                val_texts=val_texts, test_texts=test_texts,
                device=device, seed_output_dir=seed_dir,
            )
        else:
            result = train_one_seed(
                seed=seed, args=args,
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                val_texts=val_texts, test_texts=test_texts,
                feature_dim=feature_dim,
                device=device, seed_output_dir=seed_dir,
            )
        all_results.append(result)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    if multi_seed:
        f1s   = [r["f1"]        for r in all_results]
        precs = [r["precision"] for r in all_results]
        recs  = [r["recall"]    for r in all_results]
        fps   = [r["fp"]        for r in all_results]
        fns   = [r["fn"]        for r in all_results]

        summary = {
            "seeds":    seeds_list,
            "per_seed": all_results,
            "summary": {
                "f1_mean":        float(np.mean(f1s)),
                "f1_std":         float(np.std(f1s)),
                "precision_mean": float(np.mean(precs)),
                "precision_std":  float(np.std(precs)),
                "recall_mean":    float(np.mean(recs)),
                "recall_std":     float(np.std(recs)),
                "fp_mean":        float(np.mean(fps)),
                "fn_mean":        float(np.mean(fns)),
            },
        }

        s = summary["summary"]
        print(f"\n{'='*60}")
        print(f"Multi-seed summary ({len(seeds_list)} seeds: {seeds_list})")
        print(f"{'='*60}")
        print(f"  F1:        {s['f1_mean']:.4f} ± {s['f1_std']:.4f}")
        print(f"  Precision: {s['precision_mean']:.4f} ± {s['precision_std']:.4f}")
        print(f"  Recall:    {s['recall_mean']:.4f} ± {s['recall_std']:.4f}")
        print(f"  FP (mean): {s['fp_mean']:.1f}")
        print(f"  FN (mean): {s['fn_mean']:.1f}")
        print(f"{'='*60}")

        summary_path = output_dir / "multiseed_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved → {summary_path}")
        print(f"Per-seed outputs → {output_dir}/seed_*/")

    else:
        # Single-seed: legacy output format in output_dir directly.
        result = all_results[0]
        eval_results = {
            "f1":        result["f1"],
            "accuracy":  result["accuracy"],
            "precision": result["precision"],
            "recall":    result["recall"],
        }
        print(f"\nTest results: {eval_results}")
        print(f"\nOutput saved to {output_dir}/")
        print("  best_model.pt  eval_results.json  feature_stats.json  predict_results.txt  val_predictions.json")


if __name__ == "__main__":
    main()
