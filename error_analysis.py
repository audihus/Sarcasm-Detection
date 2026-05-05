# -*- coding: utf-8 -*-
"""
error_analysis.py
=================
Analisis error (FP dan FN) dari model multichannel sarcasm detection.

Usage:
    python error_analysis.py \
        --test_data       ./data/test.csv          \
        --graph_dir       ./data/                  \
        --checkpoint_dir  ./checkpoints/           \
        --dataset_name    reddit                   \
        --batch_size      32                       \
        --output_dir      ./error_analysis/

Catatan: path train_data untuk rebuild vocab diambil otomatis dari checkpoint.
         Jika file sudah dipindah, berikan --train_data secara eksplisit.
"""

# ── 1. sys.path + mock models.affectivegcn (sama dengan train_multichannel.py) ──
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

# ── 2. Standard imports ──────────────────────────────────────────────────────
import argparse
import csv
import re
import textwrap
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── 3. Import dari train_multichannel (file asli tidak dimodifikasi) ─────────
from train_multichannel import (
    SarcasmDataset,
    collate_fn,
    build_vocab_and_embed,
    build_flags,
    PatchedBridgeModel,
)

# ─────────────────────────────────────────────────────────────────────────────
# Konstanta linguistik
# ─────────────────────────────────────────────────────────────────────────────

# Stopword Bahasa Indonesia (kata fungsi, pronoun, dan kata sangat umum)
# Partikel pragmatis (sih, kok, dong, kan, lah, deh, ya) TIDAK dimasukkan
# agar bisa muncul di daftar top words jika memang sering.
ID_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "juga", "tidak", "adalah", "dalam", "atau", "sudah", "bisa",
    "akan", "tapi", "jadi", "seperti", "lagi", "belum", "karena", "hingga",
    "menjadi", "setelah", "sebuah", "oleh", "bahwa", "ketika", "banyak",
    "hanya", "sangat", "semua", "telah", "bila", "bagi", "namun", "kalau",
    "agar", "saat", "pun", "pernah", "antara", "serta", "jika", "makin",
    "atas", "bawah", "sebelum", "lebih", "kurang", "lain", "sama", "mau",
    "maka", "saya", "aku", "kamu", "kita", "kami", "ia", "dia", "mereka",
    "anda", "gue", "gw", "lo", "lu", "sini", "sana", "situ", "ada", "harus",
    "perlu", "ingin", "si", "se", "nya", "nah", "nih", "tuh",
    # Kemungkinan kata Inggris di dataset campuran
    "the", "a", "an", "is", "it", "i", "and", "or", "not", "be", "to",
    "of", "in", "that", "for", "on", "are", "as", "with", "his", "they",
}

# Partikel pragmatis yang dianalisis terpisah
PRAGMATIC_PARTICLES = ["sih", "kok", "dong", "kan", "lah", "deh", "ya"]

# Regex untuk deteksi emoji (mencakup blok Unicode utama)
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"   # dingbats
    "\U000024C2-\U0001F251"   # enclosed characters
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # extended-A
    "\U00002600-\U000026FF"   # misc symbols
    "]+",
    flags=re.UNICODE,
)

# Regex untuk ellipsis
ELLIPSIS_RE = re.compile(r"\.{3,}|…")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args, device: torch.device) -> tuple:
    """
    Load model dari checkpoint best_model.pt.

    Urutan:
      1. Baca checkpoint untuk mendapat saved_args
      2. Rebuild vocab dari train_data (CLI atau checkpoint)
      3. Build PatchedBridgeModel dan load state_dict

    Returns:
        (model, ckpt_info_dict)
    """
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {ckpt_path}\n"
            f"Pastikan --checkpoint_dir menunjuk ke folder yang berisi best_model.pt"
        )

    print(f"[Checkpoint] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get("args", {})

    print(
        f"[Checkpoint] Epoch={ckpt.get('epoch', '?')}  "
        f"Best-Val-F1={ckpt.get('best_val_f1', 0):.4f}"
    )

    # ── Tentukan path train_data (CLI > checkpoint) ─────────────────────── #
    train_data       = args.train_data       or saved.get("train_data")
    train_split_name = args.train_split_name or saved.get("train_split_name", "train.csv")
    train_graph_dir  = args.train_graph_dir  or saved.get("graph_dir", args.graph_dir)

    if not train_data:
        raise ValueError(
            "Tidak dapat menemukan path train_data untuk rebuild vocab.\n"
            "Berikan --train_data <path/ke/train.csv> sebagai argumen."
        )
    if not os.path.exists(train_data):
        raise FileNotFoundError(
            f"train_data tidak ditemukan: {train_data}\n"
            f"Berikan --train_data dengan path yang benar di environment ini."
        )

    print(f"[Vocab] Rebuilding vocab dari: {train_data}")
    train_ds = SarcasmDataset(train_data, train_graph_dir, train_split_name)

    # Gunakan saved_args untuk build_vocab_and_embed agar vocab identik
    vocab_args = SimpleNamespace(
        voc_size     = saved.get("voc_size",     30000),
        dataset_name = saved.get("dataset_name", args.dataset_name),
        data_dir     = saved.get("data_dir",     "./"),
    )
    vocab, embed, _ = build_vocab_and_embed(train_ds, vocab_args)
    print(f"[Vocab] Size: {len(vocab)}")

    # ── Build model ──────────────────────────────────────────────────────── #
    flags_args = SimpleNamespace(**{
        **saved,
        "batch_size": args.batch_size,
    })
    flags = build_flags(flags_args, device)
    model = PatchedBridgeModel(flags, vocab=vocab, embed=embed)
    model.load_state_dict(ckpt["model_state_dict"])
    print("[Model] State dict loaded.")

    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def collect_predictions(model: PatchedBridgeModel,
                        test_loader: DataLoader) -> list:
    """
    Jalankan inference di seluruh test set.

    Mengikuti pola PERSIS sama dengan train_multichannel.py baris 648-657,
    ditambah pengambilan teks (sentences_raw) dan confidence score.

    Returns:
        list of dict dengan kunci:
            text             : str   — teks asli
            label_asli       : int   — ground-truth (0=non-sarcasm, 1=sarcasm)
            label_prediksi   : int   — prediksi model
            confidence_score : float — P(sarcasm) = softmax(logits)[:, 1]
    """
    samples = []

    for batch in test_loader:
        try:
            _, prob_np = model.stepTrain(batch, inference=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print("[OOM] Skipping eval batch.")
                continue
            raise

        texts       = batch["sentences_raw"]                        # list[str]
        true_labels = batch["sarcasms"]                             # list[int]
        pred_labels = np.argmax(prob_np, axis=-1).tolist()          # list[int]
        confidences = prob_np[:, 1].tolist()                        # P(sarcasm)

        for text, true, pred, conf in zip(texts, true_labels, pred_labels, confidences):
            samples.append({
                "text":             text,
                "label_asli":       int(true),
                "label_prediksi":   int(pred),
                "confidence_score": round(float(conf), 6),
            })

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Kategorisasi
# ─────────────────────────────────────────────────────────────────────────────

def categorize(samples: list) -> dict:
    """
    Pisahkan samples ke TP, TN, FP, FN.

    label 1 = sarcasm (positive class)
    label 0 = non-sarcasm (negative class)

    FP: label_asli=0, label_prediksi=1  (model salah kira non-sarcasm sebagai sarcasm)
    FN: label_asli=1, label_prediksi=0  (model melewatkan sarcasm asli)
    """
    return {
        "tp": [s for s in samples if s["label_asli"] == 1 and s["label_prediksi"] == 1],
        "tn": [s for s in samples if s["label_asli"] == 0 and s["label_prediksi"] == 0],
        "fp": [s for s in samples if s["label_asli"] == 0 and s["label_prediksi"] == 1],
        "fn": [s for s in samples if s["label_asli"] == 1 and s["label_prediksi"] == 0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi analisis statistik
# ─────────────────────────────────────────────────────────────────────────────

def text_length_stats(samples: list) -> dict:
    """Hitung mean/median/max panjang teks dalam jumlah kata."""
    if not samples:
        return {"mean": 0.0, "median": 0.0, "max": 0}
    lengths = [len(s["text"].split()) for s in samples]
    return {
        "mean":   round(float(np.mean(lengths)),   2),
        "median": round(float(np.median(lengths)), 2),
        "max":    int(max(lengths)),
    }


def _clean_word(word: str) -> str:
    """Hapus karakter non-alfanumerik dari satu kata."""
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE).lower()


def top_words(samples: list, n: int = 20) -> list:
    """
    Hitung N kata paling sering di sampel, kecuali stopword, kata tunggal,
    dan angka murni.
    """
    counter = Counter()
    for s in samples:
        for raw in s["text"].lower().split():
            word = _clean_word(raw)
            if word and len(word) > 1 and word not in ID_STOPWORDS and not word.isdigit():
                counter[word] += 1
    return counter.most_common(n)


def feature_presence_pct(samples: list, feature_fn) -> float:
    """Persentase sampel di mana feature_fn(text) == True."""
    if not samples:
        return 0.0
    count = sum(1 for s in samples if feature_fn(s["text"]))
    return round(count / len(samples) * 100, 2)


def confidence_stats(samples: list) -> dict:
    """Statistik confidence_score (P sarcasm)."""
    if not samples:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    scores = [s["confidence_score"] for s in samples]
    return {
        "mean":   round(float(np.mean(scores)),   4),
        "median": round(float(np.median(scores)), 4),
        "min":    round(float(min(scores)),        4),
        "max":    round(float(max(scores)),        4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature detection functions
# ─────────────────────────────────────────────────────────────────────────────

def has_exclamation(text: str) -> bool:
    return "!" in text

def has_question(text: str) -> bool:
    return "?" in text

def has_ellipsis(text: str) -> bool:
    return bool(ELLIPSIS_RE.search(text))

def has_emoji(text: str) -> bool:
    return bool(EMOJI_RE.search(text))

def has_particle(particle: str):
    """Factory: returns a function that checks if `particle` appears as whole word."""
    pattern = re.compile(rf"(?<![a-zA-Z]){re.escape(particle)}(?![a-zA-Z])",
                         re.IGNORECASE)
    def _check(text: str) -> bool:
        return bool(pattern.search(text))
    return _check


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(samples: list, path: str) -> None:
    """Simpan samples ke CSV dengan kolom: text, label_asli, label_prediksi, confidence_score."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["text", "label_asli", "label_prediksi", "confidence_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    print(f"  Saved {len(samples)} rows → {path}")


def _fmt_section(title: str, lines: list) -> str:
    """Format satu seksi teks ringkasan."""
    bar = "─" * 60
    return "\n".join(["", bar, f"  {title}", bar] + [f"  {l}" for l in lines])


def build_summary(dataset_name: str, cats: dict, all_samples: list) -> str:
    """
    Bangun string ringkasan lengkap. Sama dengan yang dicetak ke terminal
    dan disimpan ke .txt.
    """
    fp, fn = cats["fp"], cats["fn"]
    tp, tn = cats["tp"], cats["tn"]
    lines  = []

    # ── Header ────────────────────────────────────────────────────────────── #
    lines.append("=" * 60)
    lines.append(f"  ERROR ANALYSIS: {dataset_name.upper()}")
    lines.append("=" * 60)

    # ── Overview ──────────────────────────────────────────────────────────── #
    lines.append(_fmt_section("OVERVIEW", [
        f"Total test samples : {len(all_samples)}",
        f"True Positive  (TP): {len(tp)}",
        f"True Negative  (TN): {len(tn)}",
        f"False Positive (FP): {len(fp)}  ← non-sarcasm diprediksi sarcasm",
        f"False Negative (FN): {len(fn)}  ← sarcasm asli terlewat",
    ]))

    # ── Panjang teks ──────────────────────────────────────────────────────── #
    st_all = text_length_stats(all_samples)
    st_fp  = text_length_stats(fp)
    st_fn  = text_length_stats(fn)
    lines.append(_fmt_section("DISTRIBUSI PANJANG TEKS (jumlah kata)", [
        f"{'Kategori':<12} {'Mean':>7} {'Median':>8} {'Max':>6}",
        f"{'─'*36}",
        f"{'All':<12} {st_all['mean']:>7.1f} {st_all['median']:>8.1f} {st_all['max']:>6}",
        f"{'FP':<12} {st_fp['mean']:>7.1f} {st_fp['median']:>8.1f} {st_fp['max']:>6}",
        f"{'FN':<12} {st_fn['mean']:>7.1f} {st_fn['median']:>8.1f} {st_fn['max']:>6}",
    ]))

    # ── Top 20 words FP ───────────────────────────────────────────────────── #
    fp_words = top_words(fp, 20)
    fp_word_lines = [f"  {i+1:>2}. {w} ({c})" for i, (w, c) in enumerate(fp_words)] \
                    if fp_words else ["  (tidak ada data)"]
    lines.append(_fmt_section("TOP 20 KATA DI FALSE POSITIVE (bukan stopword)", fp_word_lines))

    # ── Top 20 words FN ───────────────────────────────────────────────────── #
    fn_words = top_words(fn, 20)
    fn_word_lines = [f"  {i+1:>2}. {w} ({c})" for i, (w, c) in enumerate(fn_words)] \
                    if fn_words else ["  (tidak ada data)"]
    lines.append(_fmt_section("TOP 20 KATA DI FALSE NEGATIVE (bukan stopword)", fn_word_lines))

    # ── Fitur linguistik ──────────────────────────────────────────────────── #
    features = [
        ("Tanda seru (!)",   has_exclamation),
        ("Tanda tanya (?)",  has_question),
        ("Ellipsis (...)",   has_ellipsis),
        ("Emoji",            has_emoji),
    ]
    feat_lines = [f"{'Fitur':<22} {'FP %':>7} {'FN %':>7}"]
    feat_lines.append("─" * 38)
    for label, fn_check in features:
        pct_fp = feature_presence_pct(fp, fn_check)
        pct_fn = feature_presence_pct(fn, fn_check)
        feat_lines.append(f"{label:<22} {pct_fp:>6.1f}% {pct_fn:>6.1f}%")
    lines.append(_fmt_section("FITUR LINGUISTIK — PERSENTASE KEHADIRAN", feat_lines))

    # ── Partikel pragmatis ────────────────────────────────────────────────── #
    part_lines = [f"{'Partikel':<12} {'FP %':>7} {'FN %':>7}"]
    part_lines.append("─" * 28)
    for p in PRAGMATIC_PARTICLES:
        check   = has_particle(p)
        pct_fp  = feature_presence_pct(fp, check)
        pct_fn  = feature_presence_pct(fn, check)
        part_lines.append(f"{p:<12} {pct_fp:>6.1f}% {pct_fn:>6.1f}%")
    lines.append(_fmt_section("PARTIKEL PRAGMATIS INDONESIA", part_lines))

    # ── Distribusi confidence score ───────────────────────────────────────── #
    cs_fp = confidence_stats(fp)
    cs_fn = confidence_stats(fn)
    lines.append(_fmt_section("DISTRIBUSI CONFIDENCE SCORE  [P(sarcasm)]", [
        "(Tinggi = model yakin prediksi sarcasm, Rendah = yakin non-sarcasm)",
        "",
        f"{'Kategori':<8} {'Mean':>7} {'Median':>8} {'Min':>7} {'Max':>7}",
        "─" * 42,
        f"{'FP':<8} {cs_fp['mean']:>7.4f} {cs_fp['median']:>8.4f} "
        f"{cs_fp['min']:>7.4f} {cs_fp['max']:>7.4f}",
        f"{'FN':<8} {cs_fn['mean']:>7.4f} {cs_fn['median']:>8.4f} "
        f"{cs_fn['min']:>7.4f} {cs_fn['max']:>7.4f}",
        "",
        "FP confidence tinggi = model sangat yakin (tapi salah arah sarcasm)",
        "FN confidence rendah = borderline errors; tinggi = hard negatives",
    ]))

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Load model ────────────────────────────────────────────────────────── #
    model, _ = load_model(args, device)

    # ── Load test dataset ─────────────────────────────────────────────────── #
    print(f"[Data] Loading test data: {args.test_data}")
    test_ds = SarcasmDataset(args.test_data, args.graph_dir, args.test_split_name)
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )
    print(f"[Data] Test samples: {len(test_ds)}")

    # ── Inference ─────────────────────────────────────────────────────────── #
    print("[Inference] Running on test set...")
    all_samples = collect_predictions(model, test_loader)
    print(f"[Inference] Collected {len(all_samples)} predictions.")

    # ── Kategorisasi ──────────────────────────────────────────────────────── #
    cats = categorize(all_samples)
    fp, fn = cats["fp"], cats["fn"]

    # ── Simpan FP dan FN ke CSV ───────────────────────────────────────────── #
    os.makedirs(args.output_dir, exist_ok=True)
    fp_path = os.path.join(args.output_dir, f"{args.dataset_name}_false_positive.csv")
    fn_path = os.path.join(args.output_dir, f"{args.dataset_name}_false_negative.csv")

    print("\n[Output] Saving CSV files...")
    save_csv(fp, fp_path)
    save_csv(fn, fn_path)

    # ── Bangun dan cetak ringkasan ─────────────────────────────────────────── #
    summary = build_summary(args.dataset_name, cats, all_samples)
    print(summary)

    # ── Simpan ringkasan ke txt ────────────────────────────────────────────── #
    txt_path = os.path.join(args.output_dir, f"{args.dataset_name}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n[Output] Summary saved → {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# parse_args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Error analysis (FP/FN) untuk multichannel sarcasm detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data test
    parser.add_argument("--test_data",       required=True,
                        help="Path ke test CSV (kolom: content, label)")
    parser.add_argument("--test_split_name", required=True,
                        help="Nama split test untuk .graph.new/.sentic, misal 'test.csv'")
    parser.add_argument("--graph_dir",       required=True,
                        help="Folder berisi .graph.new dan .sentic untuk test split")

    # Checkpoint
    parser.add_argument("--checkpoint_dir",  required=True,
                        help="Folder berisi best_model.pt")

    # Metadata
    parser.add_argument("--dataset_name",    required=True,
                        help="Nama dataset untuk penamaan output, misal 'reddit' atau 'twitter'")

    # Training data untuk rebuild vocab (opsional — diambil dari checkpoint jika tersedia)
    parser.add_argument("--train_data",       default=None,
                        help="[Opsional] Path ke train CSV untuk rebuild vocab. "
                             "Jika tidak diberikan, diambil dari args yang disimpan di checkpoint.")
    parser.add_argument("--train_split_name", default=None,
                        help="[Opsional] Nama split train, misal 'train.csv'. "
                             "Default diambil dari checkpoint.")
    parser.add_argument("--train_graph_dir",  default=None,
                        help="[Opsional] Graph dir untuk train split. "
                             "Default: diambil dari checkpoint, fallback ke --graph_dir.")

    # Inference
    parser.add_argument("--batch_size",  type=int, default=32)

    # Output
    parser.add_argument("--output_dir",  default="./error_analysis/",
                        help="Folder output untuk CSV dan summary txt")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_analysis(parse_args())
