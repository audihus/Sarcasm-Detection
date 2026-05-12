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
        "feature_dim": 3,
    },
    "twitter": {
        "hub_name": "w11wo/twitter_indonesia_sarcastic",
        "text_col": "tweet",
        "feature_dim": 3,
    },
}

_BASELINE_HEAD_PARAMS = 768 * 2 + 2  # Linear(768, 2)


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
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Build numeric feature matrix.

    Reddit  -> shape (N, 3): [word_count, sentence_count, avg_sentence_length]
                              all z-score normalized
    Twitter -> shape (N, 3): [is_clash, question_count_zscore, has_hyperbole]
                              binary     z-score norm             binary

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

        features = np.stack(
            [
                np.array(is_clash_list, dtype=float),
                q_normalized,
                np.array(has_hyperbole_list, dtype=float),
            ],
            axis=1,
        )
        return features, feature_stats

    else:
        raise ValueError(f"dataset_name harus 'reddit' atau 'twitter', dapat: {dataset_name!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """Returns (f1_binary, predictions, true_labels)."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["features"].to(device),
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    f1 = f1_score(all_labels, all_preds, average="binary")
    return f1, all_preds, all_labels


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
                        help="Random seed (default: 42)")
    parser.add_argument("--inset_pos_path", default=None,
                        help="Path ke positive.tsv InSet (wajib untuk Twitter)")
    parser.add_argument("--inset_neg_path", default=None,
                        help="Path ke negative.tsv InSet (wajib untuk Twitter)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = DATASET_CONFIG[args.dataset_name]
    text_col: str = cfg["text_col"]
    feature_dim: int = cfg["feature_dim"]

    # ------------------------------------------------------------------
    # 1. Load dataset
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
    # 2. Load InSet lexicon (Twitter only)
    # ------------------------------------------------------------------
    inset_pos: Optional[frozenset] = None
    inset_neg: Optional[frozenset] = None
    if args.dataset_name == "twitter":
        if not args.inset_pos_path or not args.inset_neg_path:
            raise ValueError(
                "Twitter membutuhkan --inset_pos_path dan --inset_neg_path"
            )
        print("\n[2/6] Loading InSet lexicon...")
        inset_pos, inset_neg = load_inset_lexicon(args.inset_pos_path, args.inset_neg_path)
    else:
        print("\n[2/6] InSet tidak diperlukan untuk Reddit, dilewati.")

    # ------------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------------
    print("\n[3/6] Extracting features...")
    train_features, feature_stats = extract_features(
        train_texts, args.dataset_name,
        inset_pos=inset_pos, inset_neg=inset_neg,
        is_train=True,
    )
    val_features, _ = extract_features(
        val_texts, args.dataset_name,
        inset_pos=inset_pos, inset_neg=inset_neg,
        feature_stats=feature_stats,
    )
    test_features, _ = extract_features(
        test_texts, args.dataset_name,
        inset_pos=inset_pos, inset_neg=inset_neg,
        feature_stats=feature_stats,
    )
    print(f"  Feature stats: {feature_stats}")
    print(f"  Train features shape: {train_features.shape}")

    with open(output_dir / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)

    # ------------------------------------------------------------------
    # 4. Tokenization
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
    # 5. DataLoader
    # ------------------------------------------------------------------
    train_ds = SarcasmDataset(train_enc["input_ids"], train_enc["attention_mask"], train_features, train_labels)
    val_ds   = SarcasmDataset(val_enc["input_ids"],   val_enc["attention_mask"],   val_features,   val_labels)
    test_ds  = SarcasmDataset(test_enc["input_ids"],  test_enc["attention_mask"],  test_features,  test_labels)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=args.shuffle_train_dataset, generator=g,
    )
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    # ------------------------------------------------------------------
    # 6. Model, optimizer, training
    # ------------------------------------------------------------------
    print(f"\n[5/6] Initializing model (feature_dim={feature_dim})...")
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

    # Cosine LR scheduler — steps per batch, matching HF Trainer behaviour
    total_steps = args.num_epochs * len(train_loader)
    if args.lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    else:
        scheduler = None

    use_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None
    if use_fp16:
        print("  fp16 mixed precision enabled")

    best_model_path = output_dir / "best_model.pt"
    best_f1 = 0.0
    patience_counter = 0

    print(f"\n[6/6] Training for up to {args.num_epochs} epochs (patience=3)...")
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

    # ------------------------------------------------------------------
    # Evaluate best model on test set
    # ------------------------------------------------------------------
    print(f"\nLoading best checkpoint (val_{args.metric_for_best_model}={best_f1:.4f})...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_f1, test_preds, test_true = run_eval(model, test_loader, device)
    test_acc = accuracy_score(test_true, test_preds)
    test_pre = precision_score(test_true, test_preds, average="binary")
    test_rec = recall_score(test_true, test_preds, average="binary")

    eval_results = {
        "f1": test_f1,
        "accuracy": test_acc,
        "precision": test_pre,
        "recall": test_rec,
    }
    print(f"\nTest results: {eval_results}")

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # predict_results.txt — format identik dengan run_classification.py
    with open(output_dir / "predict_results.txt", "w") as f:
        f.write("index\tprediction\n")
        for idx, pred in enumerate(test_preds):
            f.write(f"{idx}\t{pred}\n")

    print(f"\nOutput saved to {output_dir}/")
    print("  best_model.pt  eval_results.json  feature_stats.json  predict_results.txt")


if __name__ == "__main__":
    main()
