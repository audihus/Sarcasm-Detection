#!/usr/bin/env python3
"""
Training script: IndoBERT dua-head (main=sarkas, aux=incongruity clash).

Mendukung ablation:
  --use_raw_input        Row 1: konten mentah tanpa marker
  (default)              Row 2-3: to_encoder_text + special tokens
  --pooling mean         Row 4: mean-pool ablation
  --lam_aux 0            Row 1-2: tanpa aux loss
  --lam_aux 0.05|0.1|0.2 Row 3: sweep aux weight

Output per seed:
  <output_dir>/seed_<N>/val_predictions.json
  <output_dir>/seed_<N>/test_predictions.json   (kompatibel threshold_tuning.py)
  <output_dir>/seed_<N>/metrics.json

Contoh penggunaan:
  python scripts/run_sarcasm_marker_aux.py \\
      --train_file real_data/twitter/train.csv \\
      --val_file   real_data/twitter/validation.csv \\
      --test_file  real_data/twitter/test.csv \\
      --pooling cls --lam_aux 0.1 \\
      --inset_pos_path real_data/twitter/positive.tsv \\
      --inset_neg_path real_data/twitter/negative.tsv \\
      --seeds "42,1,2" --fp16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

# Pastikan scripts/framework/ bisa diimport dari manapun script ini dijalankan.
FRAMEWORK = Path(__file__).resolve().parent / "framework"
sys.path.insert(0, str(FRAMEWORK))
from sarcasm_model import SarcasmModel
from sarcasm_preprocess import SPECIAL_TOKENS, to_encoder_text
from incongruity import incongruity_label, load_inset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SarcasmDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_len: int,
        clash_labels: Optional[List[int]] = None,
    ):
        self.enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.clash  = (
            torch.tensor(clash_labels, dtype=torch.long)
            if clash_labels is not None
            else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        if self.clash is not None:
            item["clash"] = self.clash[idx]
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_incongruity(texts: List[str], inset, max_len_inset: int) -> List[int]:
    from sarcasm_preprocess import to_lexicon_tokens
    result = []
    for t in texts:
        toks = to_lexicon_tokens(t)
        clash, _ = incongruity_label(toks, inset, max_len_inset)
        result.append(clash)
    return result


def load_split(path: str, use_raw: bool) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path)
    raw_texts = df["content"].astype(str).tolist()
    labels    = df["label"].astype(int).tolist()
    texts = raw_texts if use_raw else [to_encoder_text(t) for t in raw_texts]
    return texts, labels


def metrics_dict(labels, preds) -> dict:
    return {
        "f1_binary":   float(f1_score(labels, preds, average="binary", zero_division=0)),
        "f1_macro":    float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "precision":   float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall":      float(recall_score(labels, preds, average="binary", zero_division=0)),
        "accuracy":    float(accuracy_score(labels, preds)),
    }


def save_predictions(path: Path, texts, probs, preds, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"texts": texts, "probs": probs, "preds": preds, "labels": labels},
            f, ensure_ascii=False, indent=2,
        )


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, fp16: bool):
    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with autocast("cuda", enabled=fp16):
                main_logits, _ = model(ids, mask)
            probs = torch.softmax(main_logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(main_logits, dim=-1).cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
    labels = loader.dataset.labels.tolist()
    f1 = f1_score(labels, all_preds, average="binary", zero_division=0)
    return f1, all_probs, all_preds


# ---------------------------------------------------------------------------
# Training loop (satu seed)
# ---------------------------------------------------------------------------

def run_one_seed(args, seed: int, inset, max_len_inset: int, class_weights: np.ndarray):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[seed={seed}] device={device}")

    # ----- Tokenizer -----
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if not args.use_raw_input:
        tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # ----- Data -----
    print(f"[seed={seed}] Loading data...")
    train_texts, train_labels = load_split(args.train_file, args.use_raw_input)
    val_texts,   val_labels   = load_split(args.val_file,   args.use_raw_input)
    test_texts,  test_labels  = load_split(args.test_file,  args.use_raw_input)

    train_clash = None
    if args.lam_aux > 0:
        print(f"[seed={seed}] Menghitung incongruity untuk {len(train_texts)} train...")
        train_clash = compute_incongruity(train_texts, inset, max_len_inset)
        clash_rate  = sum(train_clash) / len(train_clash)
        print(f"[seed={seed}] Clash rate train: {clash_rate:.1%}")

    train_ds = SarcasmDataset(train_texts, train_labels, tok, args.max_seq_len, train_clash)
    val_ds   = SarcasmDataset(val_texts,   val_labels,   tok, args.max_seq_len)
    test_ds  = SarcasmDataset(test_texts,  test_labels,  tok, args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=args.shuffle_train, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ----- Model -----
    model = SarcasmModel(args.model_name, len(tok), pooling=args.pooling).to(device)

    # ----- Optimizer & Scheduler -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ----- Loss -----
    w = torch.tensor(class_weights, dtype=torch.float, device=device)
    main_loss_fn = nn.CrossEntropyLoss(weight=w)
    aux_loss_fn  = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=args.fp16)

    # ----- Training -----
    best_val_f1   = -1.0
    patience_count = 0
    best_state    = None

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            with autocast("cuda", enabled=args.fp16):
                main_logits, aux_logits = model(ids, mask)
                loss = main_loss_fn(main_logits, labs)
                if args.lam_aux > 0 and "clash" in batch:
                    clash = batch["clash"].to(device)
                    loss = loss + args.lam_aux * aux_loss_fn(aux_logits, clash)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        val_f1, _, _ = evaluate(model, val_loader, device, args.fp16)
        avg_loss = total_loss / len(train_loader)
        print(f"[seed={seed}] epoch {epoch:3d} | loss {avg_loss:.4f} | val F1 {val_f1:.4f}")

        if val_f1 > best_val_f1 + args.early_stopping_threshold:
            best_val_f1   = val_f1
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[seed={seed}] Early stop @ epoch {epoch} (best val F1={best_val_f1:.4f})")
                break

    # ----- Final eval dengan best model -----
    model.load_state_dict(best_state)

    val_f1,  val_probs,  val_preds  = evaluate(model, val_loader,  device, args.fp16)
    test_f1, test_probs, test_preds = evaluate(model, test_loader, device, args.fp16)

    val_m  = metrics_dict(val_labels,  val_preds)
    test_m = metrics_dict(test_labels, test_preds)

    print(f"[seed={seed}] Best  val F1  = {val_f1:.4f}")
    print(f"[seed={seed}] Final test F1 = {test_f1:.4f}")

    # ----- Simpan -----
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    save_predictions(seed_dir / "val_predictions.json",  val_texts,  val_probs,  val_preds,  val_labels)
    save_predictions(seed_dir / "test_predictions.json", test_texts, test_probs, test_preds, test_labels)

    with open(seed_dir / "metrics.json", "w") as f:
        json.dump({"seed": seed, "val": val_m, "test": test_m}, f, indent=2)

    return test_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SarcasmModel dua-head training")
    parser.add_argument("--model_name",     default="indobenchmark/indobert-base-p1")
    parser.add_argument("--train_file",     required=True)
    parser.add_argument("--val_file",       required=True)
    parser.add_argument("--test_file",      required=True)
    parser.add_argument("--inset_pos_path", default=None,
                        help="Path positive.tsv InSet. Wajib jika lam_aux > 0.")
    parser.add_argument("--inset_neg_path", default=None,
                        help="Path negative.tsv InSet. Wajib jika lam_aux > 0.")
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--pooling",        default="cls", choices=["cls", "mean"])
    parser.add_argument("--lam_aux",        type=float, default=0.0)
    parser.add_argument("--use_raw_input",  action="store_true",
                        help="Pakai content mentah, tanpa to_encoder_text (ablation row 1).")
    parser.add_argument("--seeds",                    default="42,1,2")
    parser.add_argument("--max_seq_len",              type=int,   default=128)
    parser.add_argument("--batch_size",               type=int,   default=32)
    parser.add_argument("--lr",                       type=float, default=1e-5)
    parser.add_argument("--weight_decay",             type=float, default=0.03)
    parser.add_argument("--num_epochs",               type=int,   default=100)
    parser.add_argument("--patience",                 type=int,   default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01)
    parser.add_argument("--shuffle_train",            action="store_true")
    parser.add_argument("--fp16",                     action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # ----- InSet -----
    inset, max_len_inset = None, 1
    if args.lam_aux > 0:
        if args.inset_pos_path is None or args.inset_neg_path is None:
            parser.error("--inset_pos_path dan --inset_neg_path wajib jika --lam_aux > 0")
        inset, max_len_inset = load_inset(args.inset_pos_path, args.inset_neg_path)
        print(f"InSet dimuat: {len(inset)} entri, frasa terpanjang {max_len_inset} kata")

    # ----- Class weights (dari train set) -----
    train_df = pd.read_csv(args.train_file)
    train_labels_all = train_df["label"].astype(int).tolist()
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels_all)
    print(f"Class weights: {class_weights}")

    # ----- Multi-seed run -----
    all_test_f1 = []
    for seed in seeds:
        test_f1 = run_one_seed(args, seed, inset, max_len_inset, class_weights)
        all_test_f1.append(test_f1)

    print(f"\n{'='*50}")
    print(f"Test F1 per seed : {[round(f, 4) for f in all_test_f1]}")
    print(f"Mean +/- std     : {np.mean(all_test_f1):.4f} +/- {np.std(all_test_f1):.4f}")

    summary = {
        "seeds":            seeds,
        "test_f1_per_seed": all_test_f1,
        "test_f1_mean":     float(np.mean(all_test_f1)),
        "test_f1_std":      float(np.std(all_test_f1)),
        "config":           vars(args),
    }
    out_path = Path(args.output_dir) / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary disimpan -> {out_path}")


if __name__ == "__main__":
    main()
