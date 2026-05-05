# -*- coding: utf-8 -*-
"""
train_ablation.py
=================
Ablation study untuk model multichannel sarcasm detection.

7 Variant:
  full            : BERT + ADGCN + dependency & sentic graph (baseline)
  bert_only       : hanya BERT CLS, tanpa ADGCN
  adgcn_only      : hanya ADGCN (BiLSTM + dep & sentic alternating), tanpa BERT
  dep_only        : BERT + ADGCN, tapi semua 6 GCN layer pakai dependency graph
  sentic_only     : BERT + ADGCN, tapi semua 6 GCN layer pakai sentic graph
  gcn_dep_only    : ADGCN saja dengan dep graph di semua 6 layer, tanpa BERT
  gcn_sentic_only : ADGCN saja dengan sentic graph di semua 6 layer, tanpa BERT

Usage (satu variant):
    python train_ablation.py \
        --train_data ./data/train.csv --test_data ./data/test.csv \
        --graph_dir ./data/ --train_split_name train.csv --test_split_name test.csv \
        --ablation bert_only

Usage (semua variant sekaligus):
    python train_ablation.py \
        --train_data ./data/train.csv --test_data ./data/test.csv \
        --graph_dir ./data/ --train_split_name train.csv --test_split_name test.csv \
        --ablation all
"""

# ── 1. os / sys ──────────────────────────────────────────────────────────────
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.join(SCRIPT_DIR, "multichannel-sarcasm-detection")
for _p in [SCRIPT_DIR, REPO_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 2. Mock 'models.affectivegcn' (broken import di GarphModel.py baris 11) ─
from types import ModuleType, SimpleNamespace

if "models" not in sys.modules:
    _mock_models = ModuleType("models")
    sys.modules["models"] = _mock_models
    _mock_affectivegcn = ModuleType("models.affectivegcn")
    sys.modules["models.affectivegcn"] = _mock_affectivegcn
    _mock_affectivegcn.GraphConvolution = None

# ── 3. Standard imports ──────────────────────────────────────────────────────
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import (
    f1_score        as sk_f1_score,
    precision_score as sk_precision_score,
    recall_score    as sk_recall_score,
)

# ── 4. Import utilitas dari train_multichannel (file asli tidak dimodifikasi) ─
from train_multichannel import (
    SarcasmDataset,
    collate_fn,
    drop_edge,
    build_vocab_and_embed,
    build_flags,
    run_evaluation,
    train_one_epoch,
    PatchedBridgeModel,
    MAX_LEN,
    DIM_INPUT,
    DIM_HIDDEN,
)

# ── 5. Import dari repo multichannel ─────────────────────────────────────────
from Model       import dualModel
from bridgeModel import bridgeModel, FocalLoss, _split_decay
from basicModel  import Lang
from evaluation  import evaluateClassification
from torch.cuda.amp import GradScaler

# ─────────────────────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_MODES = ["full", "bert_only", "adgcn_only", "dep_only", "sentic_only",
                  "gcn_dep_only", "gcn_sentic_only"]

# Variant yang TIDAK menggunakan BERT (forward skip encoder, optimizer eksklusi
# encoder_params, gradual unfreeze tidak berlaku).
NO_BERT_VARIANTS = {"adgcn_only", "gcn_dep_only", "gcn_sentic_only"}

CSV_COLUMNS = [
    "ablation_variant", "acc",
    "f1_binary", "pre_binary", "rec_binary",
    "f1_macro",  "pre_macro",  "rec_macro",
    "f1_micro",  "auc",
]


# ─────────────────────────────────────────────────────────────────────────────
# AblationDualModel
# ─────────────────────────────────────────────────────────────────────────────

class AblationDualModel(dualModel):
    """
    Subclass dualModel yang mendukung 7 ablation mode.

    Head untuk semua mode = 1-layer (Dropout + Linear) ala HF
    AutoModelForSequenceClassification:
      full / dep_only / sentic_only              : 768 + 512 = 1280  (dari super)
      bert_only                                  : 768
      adgcn_only / gcn_dep_only / gcn_sentic_only: 512
    """

    def __init__(self, opt, n_vocab, embed_list, ablation_mode: str = "full"):
        # Buat encoder, adgcn, dan dense bawaan dualModel
        # (parent sudah pakai pooler_output + Dropout+Linear(1280→2) sejak update head)
        super().__init__(opt, n_vocab, embed_list)
        self.ablation_mode = ablation_mode

        dim_adgcn = opt.dim_hidden * 2   # 256 * 2 = 512

        if ablation_mode == "bert_only":
            dense_input_dim = 768
        elif ablation_mode in ("adgcn_only", "gcn_dep_only", "gcn_sentic_only"):
            dense_input_dim = dim_adgcn
        else:
            # full / dep_only / sentic_only: dense 1280 sudah benar dari super
            return

        # Head 1-layer (Dropout + Linear) — match baseline HF style
        self.dense = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(dense_input_dim, 2),
        )

    def forward(self, dict_inst: dict):
        # Full mode: delegasikan ke parent tanpa perubahan
        if self.ablation_mode == "full":
            return super().forward(dict_inst)

        no_bert = self.ablation_mode in ("adgcn_only", "gcn_dep_only", "gcn_sentic_only")

        # ── BERT encoding ─────────────────────────────────────────────────── #
        if not no_bert:
            bert_out = self.encoder(
                input_ids      = dict_inst["input_ids"],
                attention_mask = dict_inst["attention_mask"],
            )
            # pooler_output = pretrained Linear(768→768)+tanh pada CLS
            # (sama dengan HF AutoModelForSequenceClassification)
            bert_rep = bert_out.pooler_output                # [B, 768]

        # ── ADGCN encoding ────────────────────────────────────────────────── #
        if self.ablation_mode != "bert_only":
            dep    = dict_inst["dependency_graph"]
            sentic = dict_inst["sentic_graph"]

            # Graph substitution: pass graph yang sama ke kedua slot
            if self.ablation_mode in ("dep_only", "gcn_dep_only"):
                sentic = dep         # semua 6 GCN layer pakai dependency graph
            elif self.ablation_mode in ("sentic_only", "gcn_sentic_only"):
                # Sentic graph has continuous float values [0, 2] (abs diff of
                # InSet lexicon scores), unlike binary dep graph {0, 1}.  Using
                # it for all 6 GCN layers without normalization causes larger
                # gradients → fp16 overflow after epoch 1's weight update.
                # Row-normalise (D⁻¹A) so each row sums to 1, matching the
                # effective magnitude of the binary dep graph case.
                row_sum = sentic.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                sentic = sentic / row_sum
                dep = sentic         # semua 6 GCN layer pakai sentic graph
            # adgcn_only: dep dan sentic tetap keduanya (tidak diganti)

            adgcn_rep = self.adgcn(
                dict_inst["sens"],
                dict_inst["len_sen"],
                dep,
                sentic,
            )   # [B, 512]

        # ── Fusion dan klasifikasi ────────────────────────────────────────── #
        if self.ablation_mode == "bert_only":
            dense_input = bert_rep
        elif no_bert:
            # adgcn_only / gcn_dep_only / gcn_sentic_only — single channel ADGCN
            dense_input = adgcn_rep
        else:   # dep_only / sentic_only
            dense_input = torch.cat([bert_rep, adgcn_rep], dim=-1)

        logits = self.dense(dense_input)
        prob   = torch.softmax(logits, dim=-1)
        return prob, logits


# ─────────────────────────────────────────────────────────────────────────────
# AblationBridgeModel
# ─────────────────────────────────────────────────────────────────────────────

class AblationBridgeModel(PatchedBridgeModel):
    """
    Bridge model untuk ablation study.

    Mereplikasi bridgeModel.__init__ secara manual (TIDAK memanggil super.__init__)
    agar hanya satu model yang dimuat ke GPU — menghindari double-load BERT.
    Mewarisi gen_batch_data() dari PatchedBridgeModel tanpa perubahan.
    """

    def __init__(self, FLAGS, vocab: list, embed: np.ndarray,
                 ablation_mode: str = "full"):
        # Lewati PatchedBridgeModel/__init__ agar tidak double-load BERT
        nn.Module.__init__(self)

        self.device         = FLAGS.device
        self.max_length_sen = FLAGS.max_length_sen
        self.n_class        = FLAGS.n_class
        self.learning_rate  = FLAGS.learning_rate
        self.batch_size     = FLAGS.batch_size
        self.t_sne          = False      # selalu False di ablation
        self.ablation_mode  = ablation_mode

        # fp16 (autocast + GradScaler) — match baseline id_sarcasm
        self.use_fp16 = (
            getattr(FLAGS, "use_fp16", False)
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler() if self.use_fp16 else None

        # Lang (untuk ADGCN word-index mapping)
        self.lang = Lang(vocab)

        # IndoBERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "indobenchmark/indobert-base-p1"
        )

        # Model utama (ablation variant)
        self.model = AblationDualModel(FLAGS, len(vocab), embed, ablation_mode)
        self.model.to(self.device)

        # Focal Loss (gamma=2, alpha=0.75 → upweight minority class/sarkasme)
        # Dipakai di SEMUA variant termasuk bert_only (untuk ablation study)
        self.criterion = FocalLoss(gamma=2.0, alpha=0.75, reduction="mean")

        # Optimizer — differential LR + WD scope ala HF Trainer
        # (skip bias + LayerNorm dari weight decay)
        wd = getattr(FLAGS, "weight_decay", 0.01)

        encoder_named = list(self.model.encoder.named_parameters())
        encoder_ids   = set(map(id, [p for _, p in encoder_named]))
        other_named   = [(n, p) for n, p in self.model.named_parameters()
                         if id(p) not in encoder_ids]

        enc_decay, enc_nodecay = _split_decay(encoder_named)
        oth_decay, oth_nodecay = _split_decay(other_named)

        if ablation_mode in NO_BERT_VARIANTS:
            # encoder ada tapi tidak dipakai — exclude dari optimizer
            self.optimizer = optim.AdamW([
                {"params": oth_decay,   "lr": FLAGS.learning_rate, "weight_decay": wd},
                {"params": oth_nodecay, "lr": FLAGS.learning_rate, "weight_decay": 0.0},
            ])
            opt_desc = "no-bert (single LR)"
        elif ablation_mode == "bert_only":
            # Single LR 1e-5 untuk semua param — setara kondisi baseline id_sarcasm
            self.optimizer = optim.AdamW([
                {"params": enc_decay   + oth_decay,   "lr": 1e-5, "weight_decay": wd},
                {"params": enc_nodecay + oth_nodecay, "lr": 1e-5, "weight_decay": 0.0},
            ])
            opt_desc = "single LR 1e-5 (all params)"
        else:
            # full / dep_only / sentic_only: differential LR
            self.optimizer = optim.AdamW([
                {"params": enc_decay,   "lr": 1e-5,                "weight_decay": wd},
                {"params": enc_nodecay, "lr": 1e-5,                "weight_decay": 0.0},
                {"params": oth_decay,   "lr": FLAGS.learning_rate, "weight_decay": wd},
                {"params": oth_nodecay, "lr": FLAGS.learning_rate, "weight_decay": 0.0},
            ])
            opt_desc = "differential-LR (encoder=1e-5, other=1e-3)"

        print(f"[AblationBridgeModel] mode={ablation_mode}  optimizer={opt_desc}  "
              f"WD={wd} (skip bias+LayerNorm)  fp16={self.use_fp16}")


# ─────────────────────────────────────────────────────────────────────────────
# run_ablation_variant
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_variant(
    ablation_mode: str,
    args,
    train_ds,
    val_ds,
    test_ds,
    vocab: list,
    embed: np.ndarray,
    device: torch.device,
) -> dict:
    """Satu training run lengkap untuk satu ablation variant. Returns dict metrics."""
    print(f"\n{'='*60}")
    print(f"  ABLATION VARIANT: {ablation_mode.upper()}")
    print(f"{'='*60}")

    # ── Reset RNG ke state deterministik per-variant ──────────────────────── #
    # Tanpa reset, urutan variant memengaruhi hasil tiap variant karena
    # dataloader shuffle, dropout, dan init layer mengonsumsi RNG state.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"  [Seed] RNG reset to seed={args.seed}")

    # ── DataLoaders ──────────────────────────────────────────────────────── #
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────────────────── #
    flags = build_flags(args, device)
    model = AblationBridgeModel(flags, vocab=vocab, embed=embed,
                                ablation_mode=ablation_mode)

    # ── Checkpoint dir per variant ────────────────────────────────────────── #
    ckpt_dir  = os.path.join(args.checkpoint_dir, ablation_mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    # ── Cosine LR scheduler ───────────────────────────────────────────────── #
    if args.no_scheduler:
        scheduler = None
        print("[Scheduler] Disabled — using constant LR.")
    else:
        total_steps = len(train_loader) * args.epochs
        scheduler   = get_cosine_schedule_with_warmup(
            model.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

    # ── Training loop ─────────────────────────────────────────────────────── #
    best_val_f1      = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        # Gradual unfreeze (skip untuk variant tanpa BERT — encoder tidak di-train)
        # bert_only: skip gradual unfreeze — no ADGCN to warm up while BERT is frozen
        if ablation_mode not in NO_BERT_VARIANTS and ablation_mode != "bert_only":
            freeze = (epoch <= 2)
            for param in model.model.encoder.parameters():
                param.requires_grad = not freeze

        avg_loss    = train_one_epoch(model, train_loader, epoch,
                                      args.drop_rate, scheduler)
        val_metrics = run_evaluation(model, val_loader)
        val_f1      = val_metrics["f1_binary"]

        print(
            f"  [Epoch {epoch:3d}/{args.epochs}]  "
            f"Loss={avg_loss:.4f}  Val-F1-Bin={val_f1:.4f}  "
            f"Patience={patience_counter}/{args.early_stopping_patience}"
        )

        if val_f1 > best_val_f1 + args.early_stopping_threshold:
            best_val_f1      = val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_f1":      best_val_f1,
                    "ablation_mode":    ablation_mode,
                    "args":             vars(args),
                },
                ckpt_path,
            )
            print(f"    → Checkpoint saved (Val-F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"  [Early Stop] best Val-F1={best_val_f1:.4f}")
                break

    # ── Test evaluation (best checkpoint) ─────────────────────────────────── #
    print(f"\n  [Test] Loading best checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_probs = [], []
    for batch in test_loader:
        try:
            _, prob_np = model.stepTrain(batch, inference=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
        y_true.extend(batch["sarcasms"])
        y_probs.extend(prob_np.tolist())

    test_metrics = evaluateClassification(y_true, y_probs)
    y_pred       = np.argmax(y_probs, axis=-1).tolist()

    f1_binary  = sk_f1_score(y_true, y_pred,
                              average="binary", pos_label=1, zero_division=0)
    pre_binary = sk_precision_score(y_true, y_pred,
                                    average="binary", pos_label=1, zero_division=0)
    rec_binary = sk_recall_score(y_true, y_pred,
                                  average="binary", pos_label=1, zero_division=0)

    results = {
        "ablation_variant": ablation_mode,
        "acc":        round(test_metrics["acc"],       4),
        "f1_binary":  round(f1_binary,                 4),
        "pre_binary": round(pre_binary,                4),
        "rec_binary": round(rec_binary,                4),
        "f1_macro":   round(test_metrics["f1_macro"],  4),
        "pre_macro":  round(test_metrics["pre_macro"], 4),
        "rec_macro":  round(test_metrics["rec_macro"], 4),
        "f1_micro":   round(test_metrics["f1_micro"],  4),
        "auc":        round(test_metrics["auc"],        4),
    }

    print(f"\n  {'─'*50}")
    print(f"  [{ablation_mode}] TEST RESULTS")
    print(f"  {'─'*50}")
    print(f"  Accuracy        : {results['acc']:.4f}")
    print(f"  F1-Binary(Sar)  : {results['f1_binary']:.4f}")
    print(f"  Precision-Binary: {results['pre_binary']:.4f}")
    print(f"  Recall-Binary   : {results['rec_binary']:.4f}")
    print(f"  F1-Macro        : {results['f1_macro']:.4f}")
    print(f"  AUC             : {results['auc']:.4f}")

    # Bebaskan GPU memory sebelum variant berikutnya
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def append_results_to_csv(results: dict, csv_path: str) -> None:
    """Append satu baris hasil ke CSV. Buat header jika file belum ada."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"  [CSV] Hasil disimpan ke {csv_path}")


def print_comparison_table(csv_path: str) -> None:
    """Baca CSV dan cetak tabel perbandingan semua variant."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    print("\n" + "=" * 80)
    print("  ABLATION STUDY — COMPARISON TABLE")
    print("=" * 80)
    print(
        f"  {'Variant':<14} {'Acc':>7} {'F1-Bin':>8} "
        f"{'Pre-Bin':>9} {'Rec-Bin':>9} {'F1-Mac':>8} {'AUC':>8}"
    )
    print("  " + "─" * 66)
    for row in rows:
        print(
            f"  {row['ablation_variant']:<14} "
            f"{float(row['acc']):>7.4f} "
            f"{float(row['f1_binary']):>8.4f} "
            f"{float(row['pre_binary']):>9.4f} "
            f"{float(row['rec_binary']):>9.4f} "
            f"{float(row['f1_macro']):>8.4f} "
            f"{float(row['auc']):>8.4f}"
        )
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# parse_args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study untuk multichannel sarcasm detection"
    )

    # Required data paths
    parser.add_argument("--train_data",       required=True)
    parser.add_argument("--test_data",        required=True)
    parser.add_argument("--graph_dir",        required=True)
    parser.add_argument("--train_split_name", required=True)
    parser.add_argument("--test_split_name",  required=True)

    # Ablation control
    parser.add_argument(
        "--ablation",
        default="all",
        choices=ABLATION_MODES + ["all"],
        help="Variant yang dijalankan. 'all' menjalankan semua 5 secara berurutan.",
    )
    parser.add_argument(
        "--results_csv",
        default="./ablation_results.csv",
        help="Path CSV untuk menyimpan hasil (append per variant).",
    )

    # Training hyperparameters (sama dengan train_multichannel.py)
    parser.add_argument("--checkpoint_dir",   default="./checkpoints/ablation")
    parser.add_argument("--epochs",           type=int,   default=100)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--learning_rate",    type=float, default=1e-3)
    parser.add_argument("--val_split",        type=float, default=0.1)
    parser.add_argument("--val_data",         default=None)
    parser.add_argument("--val_split_name",   default=None)
    parser.add_argument("--drop_rate",        type=float, default=0.2)
    parser.add_argument("--early_stopping_patience",  type=int,   default=3,
                        help="Match baseline id_sarcasm = 3")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01)

    # Model/vocab hyperparameters
    parser.add_argument("--voc_size",             type=int,   default=30000)
    parser.add_argument("--n_layers",             type=int,   default=1)
    parser.add_argument("--rnn_type",             default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("--embed_dropout_rate",   type=float, default=0.5)
    parser.add_argument("--cell_dropout_rate",    type=float, default=0.5)
    parser.add_argument("--final_dropout_rate",   type=float, default=0.5)
    parser.add_argument("--lambda1",              type=float, default=0.5)
    parser.add_argument("--weight_decay",         type=float, default=0.03)

    # Embedding/dataset metadata
    parser.add_argument("--dataset_name", default="id_sarcasm")
    parser.add_argument("--data_dir",     default="./")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_scheduler", action="store_true",
                        help="Disable cosine LR scheduler (use constant LR)")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable fp16 mixed precision (default: fp16 ON kalau CUDA)")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
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
        val_ds   = SarcasmDataset(args.val_data, args.graph_dir, args.val_split_name)
        train_ds = full_train_ds
        print(f"[Data] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    else:
        n_val    = max(1, int(len(full_train_ds) * args.val_split))
        n_train  = len(full_train_ds) - n_val
        train_ds, val_ds = random_split(
            full_train_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[Data] Train={n_train}  Val={n_val}  Test={len(test_ds)}")

    # ── Vocab + Embedding ─────────────────────────────────────────────────── #
    print("[Vocab] Building vocab and embedding matrix from train data...")
    vocab, embed, _ = build_vocab_and_embed(full_train_ds, args)
    print(f"[Vocab] Size: {len(vocab)}")

    # ── Tentukan variant list ─────────────────────────────────────────────── #
    variants = ABLATION_MODES if args.ablation == "all" else [args.ablation]
    print(f"\n[Ablation] Variants: {variants}")

    # ── Loop per variant ──────────────────────────────────────────────────── #
    for mode in variants:
        results = run_ablation_variant(
            ablation_mode = mode,
            args          = args,
            train_ds      = train_ds,
            val_ds        = val_ds,
            test_ds       = test_ds,
            vocab         = vocab,
            embed         = embed,
            device        = device,
        )
        append_results_to_csv(results, args.results_csv)

    # ── Tabel perbandingan akhir ──────────────────────────────────────────── #
    print_comparison_table(args.results_csv)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(parse_args())
