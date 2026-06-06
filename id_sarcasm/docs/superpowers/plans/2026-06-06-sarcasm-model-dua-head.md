# SarcasmModel Dua-Head IndoBERT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementasi `SarcasmModel` (IndoBERT + main head + aux incongruity head) dengan multi-seed training, lambda sweep, threshold tuning, dan ablation matrix rows 1–4 sesuai `scripts/framework/model.md`.

**Architecture:** IndoBERT backbone (`indobenchmark/indobert-base-p1`) + special tokens 7 buah → `[CLS] pooler_output` → dua Linear head (main: sarkas 0/1, aux: clash 0/1). Loss gabungan `L = L_main + λ * L_aux`. Training script mengikuti pola `run_classification_fusion.py` (custom loop, multi-seed, save JSON predictions).

**Tech Stack:** Python 3, PyTorch, HuggingFace Transformers, `datasets`, `scikit-learn`, `ftfy`, `emoji`, `numpy`, `pandas`

---

## File Structure

| File | Status | Tanggung jawab |
|------|--------|----------------|
| `scripts/framework/sarcasm_model.py` | **Buat baru** | Kelas `SarcasmModel` PyTorch — backbone + dua head |
| `scripts/run_sarcasm_marker_aux.py` | **Buat baru** | Training loop multi-seed, data loading, incongruity labeling inline, evaluasi |
| `recipes/twitter/marker_aux/run_ablation.sh` | **Buat baru** | Shell script 4 ablation rows (vanilla, markers, markers+aux sweep, mean-pool) |
| `scripts/framework/incongruity.py` | Read-only | Sudah ada — `incongruity_label()` dan `load_inset()` dipakai langsung |
| `scripts/framework/sarcasm_preprocess.py` | Read-only | Sudah ada — `to_encoder_text()`, `SPECIAL_TOKENS` |
| `scripts/modifiied script/threshold_tuning.py` | Read-only | Sudah ada — kompatibel langsung dengan JSON output training script |

---

## Task 1: Install Dependencies

**Files:** `requirements.txt` (modifikasi)

- [ ] **Step 1: Install paket yang belum ada**

```powershell
pip install ftfy emoji pandas
```

Expected output (tidak ada error):
```
Successfully installed ftfy-6.x.x emoji-2.x.x pandas-2.x.x
```

- [ ] **Step 2: Tambahkan ke requirements.txt**

Tambahkan tiga baris ke `requirements.txt`:
```
ftfy
emoji
pandas
```

- [ ] **Step 3: Verifikasi import**

```powershell
python -c "import ftfy, emoji, pandas, torch, transformers; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add ftfy, emoji, pandas for marker-aware pipeline"
```

---

## Task 2: Buat `scripts/framework/sarcasm_model.py`

**Files:**
- Create: `scripts/framework/sarcasm_model.py`

- [ ] **Step 1: Buat file model**

Buat `scripts/framework/sarcasm_model.py` dengan konten berikut:

```python
"""
SarcasmModel: IndoBERT + main head (sarkas) + aux head (incongruity clash).

Forward mengembalikan (main_logits, aux_logits). Pooling mode dikontrol lewat
argumen `pooling` saat konstruksi: "cls" (default) atau "mean".
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class SarcasmModel(nn.Module):
    def __init__(self, name: str, n_tokens: int, pooling: str = "cls"):
        """
        Args:
            name:     HuggingFace model ID, mis. "indobenchmark/indobert-base-p1"
            n_tokens: len(tokenizer) setelah add_special_tokens — untuk resize embedding
            pooling:  "cls" pakai pooler_output; "mean" pakai mean atas token non-padding
        """
        super().__init__()
        assert pooling in ("cls", "mean"), f"pooling harus 'cls' atau 'mean', dapat: {pooling}"
        self.pooling = pooling
        self.enc = AutoModel.from_pretrained(name)
        self.enc.resize_token_embeddings(n_tokens)
        h = self.enc.config.hidden_size       # 768 untuk indobert-base
        self.main = nn.Linear(h, 2)
        self.aux  = nn.Linear(h, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "cls":
            pooled = out.pooler_output                        # (B, 768)
        else:
            # mean-pool atas token yang bukan padding
            mask = attention_mask.unsqueeze(-1).float()       # (B, L, 1)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.main(pooled), self.aux(pooled)
```

- [ ] **Step 2: Tes cepat import**

```powershell
cd "E:\Folder audi\Code\Sarcasm-Detection\id_sarcasm"
python -c "
import sys; sys.path.insert(0, 'scripts/framework')
from sarcasm_model import SarcasmModel
print('import OK')
"
```

Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/framework/sarcasm_model.py
git commit -m "feat: add SarcasmModel two-head IndoBERT (cls+mean pooling)"
```

---

## Task 3: Buat Training Script `scripts/run_sarcasm_marker_aux.py`

**Files:**
- Create: `scripts/run_sarcasm_marker_aux.py`

Script ini satu file self-contained (mengikuti pola `run_classification_fusion.py`). Mendukung:
- `--pooling cls|mean`
- `--lam_aux 0|0.05|0.1|0.2`
- `--use_raw_input` (ablation row 1: pakai `content` mentah tanpa `to_encoder_text`)
- `--seeds "42,1,2,3,4"` (multi-seed)
- Output JSON kompatibel dengan `threshold_tuning.py`

- [ ] **Step 1: Buat file lengkap**

Buat `scripts/run_sarcasm_marker_aux.py`:

```python
#!/usr/bin/env python3
"""
Training script: IndoBERT dua-head (main=sarkas, aux=incongruity clash).

Mendukung ablation:
  --use_raw_input        Row 1: konten mentah tanpa marker
  (default)              Row 2–3: to_encoder_text + special tokens
  --pooling mean         Row 4: mean-pool ablation
  --lam_aux 0            Row 1–2: tanpa aux loss
  --lam_aux 0.05|0.1|0.2 Row 3: sweep aux weight

Output per seed:
  <output_dir>/seed_<N>/val_predictions.json
  <output_dir>/seed_<N>/test_predictions.json   (kompatibel threshold_tuning.py)
  <output_dir>/seed_<N>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AdamW, set_seed, get_cosine_schedule_with_warmup

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
    """Hitung label clash 0/1 untuk setiap teks."""
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
    if use_raw:
        texts = raw_texts
    else:
        texts = [to_encoder_text(t) for t in raw_texts]
    return texts, labels


def metrics_dict(labels, preds, probs) -> dict:
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ----- Model -----
    model = SarcasmModel(args.model_name, len(tok), pooling=args.pooling).to(device)

    # ----- Optimizer & Scheduler -----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ----- Loss -----
    w = torch.tensor(class_weights, dtype=torch.float, device=device)
    main_loss_fn = nn.CrossEntropyLoss(weight=w)
    aux_loss_fn  = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=args.fp16)

    # ----- Training -----
    best_val_f1 = -1.0
    patience_count = 0
    best_state = None

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            with autocast(enabled=args.fp16):
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

        # ----- Eval on val -----
        val_f1, val_probs, val_preds = evaluate(model, val_loader, device, args.fp16)
        avg_loss = total_loss / len(train_loader)
        print(f"[seed={seed}] epoch {epoch:3d} | loss {avg_loss:.4f} | val F1 {val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
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

    val_m  = metrics_dict(val_labels,  val_preds,  val_probs)
    test_m = metrics_dict(test_labels, test_preds, test_probs)

    print(f"[seed={seed}] Best  val F1 = {val_f1:.4f}")
    print(f"[seed={seed}] Final test F1 = {test_f1:.4f}")

    # ----- Simpan -----
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    save_predictions(seed_dir / "val_predictions.json",  val_texts,  val_probs,  val_preds,  val_labels)
    save_predictions(seed_dir / "test_predictions.json", test_texts, test_probs, test_preds, test_labels)

    result = {"seed": seed, "val": val_m, "test": test_m}
    with open(seed_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return test_f1, test_m


def evaluate(model, loader, device, fp16: bool):
    model.eval()
    all_probs, all_preds, _ = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with autocast(enabled=fp16):
                main_logits, _ = model(ids, mask)
            probs = torch.softmax(main_logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(main_logits, dim=-1).cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
    # ambil labels dari loader.dataset
    labels = loader.dataset.labels.tolist()
    f1 = f1_score(labels, all_preds, average="binary", zero_division=0)
    return f1, all_probs, all_preds


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
    parser.add_argument("--lam_aux",        type=float, default=0.0,
                        help="Bobot aux loss. 0 = tanpa aux (ablation).")
    parser.add_argument("--use_raw_input",  action="store_true",
                        help="Pakai content mentah, tanpa to_encoder_text (ablation row 1).")
    parser.add_argument("--seeds",          default="42,1,2",
                        help="Comma-separated seeds.")
    parser.add_argument("--max_seq_len",    type=int,   default=128)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--num_epochs",     type=int,   default=5)
    parser.add_argument("--patience",       type=int,   default=2)
    parser.add_argument("--fp16",           action="store_true")
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
    classes = np.array([0, 1])
    class_weights = compute_class_weight("balanced", classes=classes, y=train_labels_all)
    print(f"Class weights: {class_weights}")

    # ----- Multi-seed run -----
    all_test_f1 = []
    for seed in seeds:
        test_f1, _ = run_one_seed(args, seed, inset, max_len_inset, class_weights)
        all_test_f1.append(test_f1)

    print(f"\n{'='*50}")
    print(f"Test F1 per seed : {[round(f, 4) for f in all_test_f1]}")
    print(f"Mean ± std       : {np.mean(all_test_f1):.4f} ± {np.std(all_test_f1):.4f}")

    summary = {
        "seeds": seeds,
        "test_f1_per_seed": all_test_f1,
        "test_f1_mean": float(np.mean(all_test_f1)),
        "test_f1_std":  float(np.std(all_test_f1)),
        "config": vars(args),
    }
    out_path = Path(args.output_dir) / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary disimpan → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verifikasi sintaks**

```powershell
python -m py_compile scripts/run_sarcasm_marker_aux.py && echo "syntax OK"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_sarcasm_marker_aux.py
git commit -m "feat: add run_sarcasm_marker_aux.py — two-head training script"
```

---

## Task 4: Buat Shell Recipe Ablation

**Files:**
- Create: `recipes/twitter/marker_aux/run_ablation.sh`

- [ ] **Step 1: Buat direktori dan file**

```bash
mkdir -p recipes/twitter/marker_aux
```

Buat `recipes/twitter/marker_aux/run_ablation.sh`:

```bash
#!/usr/bin/env bash
# Ablation matrix rows 1–4 dari model.md
# Jalankan dari root id_sarcasm/:
#   bash recipes/twitter/marker_aux/run_ablation.sh

TRAIN=real_data/twitter/train.csv
VAL=real_data/twitter/validation.csv
TEST=real_data/twitter/test.csv
INSET_POS=real_data/twitter/positive.tsv
INSET_NEG=real_data/twitter/negative.tsv
SEEDS="42,1,2"

BASE="python scripts/run_sarcasm_marker_aux.py \
    --train_file $TRAIN --val_file $VAL --test_file $TEST \
    --seeds $SEEDS --fp16"

echo "=== Row 1: Vanilla baseline (mentah, CLS, lambda=0) ==="
$BASE \
    --use_raw_input --pooling cls --lam_aux 0 \
    --output_dir outputs/twitter-row1-vanilla

echo "=== Row 2: Marker-aware, CLS, lambda=0 ==="
$BASE \
    --pooling cls --lam_aux 0 \
    --output_dir outputs/twitter-row2-markers

echo "=== Row 3: Marker-aware, CLS, lambda=0.05 ==="
$BASE \
    --pooling cls --lam_aux 0.05 \
    --inset_pos_path $INSET_POS --inset_neg_path $INSET_NEG \
    --output_dir outputs/twitter-row3-aux-lam005

echo "=== Row 3: Marker-aware, CLS, lambda=0.1 ==="
$BASE \
    --pooling cls --lam_aux 0.1 \
    --inset_pos_path $INSET_POS --inset_neg_path $INSET_NEG \
    --output_dir outputs/twitter-row3-aux-lam010

echo "=== Row 3: Marker-aware, CLS, lambda=0.2 ==="
$BASE \
    --pooling cls --lam_aux 0.2 \
    --inset_pos_path $INSET_POS --inset_neg_path $INSET_NEG \
    --output_dir outputs/twitter-row3-aux-lam020

echo "=== Row 4: Ablation pooling (markers, mean-pool, lambda=0) ==="
$BASE \
    --pooling mean --lam_aux 0 \
    --output_dir outputs/twitter-row4-meanpool

echo "=== SELESAI ==="
echo "Threshold tuning: python scripts/modifiied\ script/threshold_tuning.py \
    --output_dir outputs/twitter-row3-aux-lam010 --seeds $SEEDS"
```

- [ ] **Step 2: Commit**

```bash
git add recipes/twitter/marker_aux/run_ablation.sh
git commit -m "feat: add ablation shell recipe rows 1–4"
```

---

## Task 5: Smoke Test Satu Seed (Verifikasi End-to-End)

**Files:** tidak ada perubahan

- [ ] **Step 1: Jalankan smoke test seed 42, 1 epoch, row 2**

```powershell
cd "E:\Folder audi\Code\Sarcasm-Detection\id_sarcasm"
python scripts/run_sarcasm_marker_aux.py `
    --train_file real_data/twitter/train.csv `
    --val_file   real_data/twitter/validation.csv `
    --test_file  real_data/twitter/test.csv `
    --pooling cls --lam_aux 0 `
    --seeds "42" --num_epochs 1 --batch_size 16 `
    --output_dir outputs/smoke-test-row2
```

Expected (output terakhir seperti):
```
[seed=42] epoch   1 | loss x.xxxx | val F1 x.xxxx
[seed=42] Final test F1 = x.xxxx
Test F1 per seed : [x.xxxx]
Mean ± std       : x.xxxx ± 0.0000
Summary disimpan → outputs/smoke-test-row2/summary.json
```

- [ ] **Step 2: Verifikasi file output tersimpan**

```powershell
ls outputs/smoke-test-row2/seed_42/
```

Expected: `val_predictions.json`, `test_predictions.json`, `metrics.json`

- [ ] **Step 3: Jalankan smoke test dengan aux loss (row 3)**

```powershell
python scripts/run_sarcasm_marker_aux.py `
    --train_file real_data/twitter/train.csv `
    --val_file   real_data/twitter/validation.csv `
    --test_file  real_data/twitter/test.csv `
    --pooling cls --lam_aux 0.1 `
    --inset_pos_path real_data/twitter/positive.tsv `
    --inset_neg_path real_data/twitter/negative.tsv `
    --seeds "42" --num_epochs 1 --batch_size 16 `
    --output_dir outputs/smoke-test-row3
```

Expected: output mencetak clash rate train, tidak ada error.

- [ ] **Step 4: Verifikasi threshold tuning kompatibel**

```powershell
python "scripts/modifiied script/threshold_tuning.py" `
    --output_dir outputs/smoke-test-row2 `
    --seeds "42"
```

Expected: tabel threshold tanpa error.

- [ ] **Step 5: Commit hasil smoke test (jika ada perubahan kode)**

```bash
git add -p   # stage hanya perubahan kode, bukan file outputs/
git commit -m "fix: smoke test corrections" # hanya jika ada fix
```

---

## Task 6: Jalankan Ablation Penuh (Opsional, butuh GPU)

**Files:** tidak ada perubahan kode

- [ ] **Step 1: Jalankan ablation lengkap**

```bash
bash recipes/twitter/marker_aux/run_ablation.sh
```

Estimasi waktu per row: 10–30 menit (CPU) atau 2–5 menit (GPU) × 3 seed × 6 config ≈ bisa panjang. Jalankan di background atau Kaggle.

- [ ] **Step 2: Jalankan threshold tuning untuk semua row**

```powershell
foreach ($row in "row1-vanilla","row2-markers","row3-aux-lam005","row3-aux-lam010","row3-aux-lam020","row4-meanpool") {
    python "scripts/modifiied script/threshold_tuning.py" `
        --output_dir "outputs/twitter-$row" `
        --seeds "42,1,2"
}
```

- [ ] **Step 3: Kumpulkan ringkasan**

```powershell
foreach ($row in "row1-vanilla","row2-markers","row3-aux-lam005","row3-aux-lam010","row3-aux-lam020","row4-meanpool") {
    Write-Host "=== $row ==="; cat "outputs/twitter-$row/summary.json"
}
```

---

## Verification

Setelah Task 1–5 selesai, verifikasi:

1. `python -m py_compile scripts/framework/sarcasm_model.py` → OK
2. `python -m py_compile scripts/run_sarcasm_marker_aux.py` → OK
3. Smoke test row 2 (markers, no aux) menghasilkan `val_predictions.json` dan `test_predictions.json`
4. Smoke test row 3 (with aux) mencetak clash rate tanpa error
5. `threshold_tuning.py` berjalan di atas output smoke test tanpa error
6. Git log menunjukkan 4 commit bersih (deps, model, script, recipe)

---

## Catatan: Ablation Rows 5–6

Model.md mencantumkan rows 5 (ADGCN) dan 6 (interaction head) sebagai "hasil negatif". Keduanya memerlukan arsitektur tambahan yang signifikan (ADGCN butuh graph files `.graph.new`, interaction head butuh cross-attention). Rows ini **tidak diimplementasikan dalam plan ini** — pisahkan sebagai plan terpisah jika dibutuhkan.
