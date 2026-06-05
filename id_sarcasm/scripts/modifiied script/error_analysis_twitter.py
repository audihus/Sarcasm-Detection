#!/usr/bin/env python3
"""
Error analysis untuk Twitter fusion model.
Memerlukan val_predictions.json yang dihasilkan run_classification_fusion.py.

Cara pakai:
    python scripts/error_analysis_twitter.py \
        --val_pred path/ke/val_predictions.json \
        --inset_pos real_data/twitter/positive.tsv \
        --inset_neg real_data/twitter/negative.tsv \
        --output twitter_error_analysis.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.augment_pipeline import detect_polarity_clash, load_inset_lexicon

# ---------------------------------------------------------------------------
# Surface signal helpers
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027FF"
    "\U00002702-\U000027B0"
    "]+",
    flags=re.UNICODE,
)

def _punct_count(text: str) -> int:
    return text.count("!") + text.count("?") + text.count("...")

def _has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))

def _has_elongation(text: str) -> bool:
    return bool(re.search(r"([a-zA-Z])\1{2,}", text))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis Twitter fusion model")
    parser.add_argument("--val_pred",  required=True, help="Path ke val_predictions.json")
    parser.add_argument("--inset_pos", required=True, help="Path ke positive.tsv InSet")
    parser.add_argument("--inset_neg", required=True, help="Path ke negative.tsv InSet")
    parser.add_argument("--output",    default="twitter_error_analysis.csv")
    args = parser.parse_args()

    # --- Load predictions ---
    with open(args.val_pred, encoding="utf-8") as f:
        data = json.load(f)

    texts  = data["texts"]
    preds  = data["preds"]
    labels = data["labels"]
    probs  = data["probs"]

    # --- Load InSet lexicon (pakai ulang fungsi yang sama dengan training) ---
    print("Loading InSet lexicon...")
    inset_pos, inset_neg = load_inset_lexicon(args.inset_pos, args.inset_neg)

    # --- Hitung sinyal per item (hanya yang salah) ---
    rows = []
    for i, (text, gold, pred, prob) in enumerate(zip(texts, labels, preds, probs)):
        if gold == pred:
            continue  # hanya error

        clash, _ = detect_polarity_clash(text, inset_pos, inset_neg)
        rows.append({
            "id":             i,
            "text":           text,
            "gold":           gold,
            "pred":           pred,
            "prob":           round(prob, 4),
            "has_clash":      1 if clash else 0,
            "punct_count":    _punct_count(text),
            "has_emoji":      1 if _has_emoji(text) else 0,
            "elongation":     1 if _has_elongation(text) else 0,
            "kategori_manual": "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Error CSV saved: {args.output}  ({len(df)} item salah dari {len(texts)} val items)")

    fn = df[df["gold"] == 1]
    fp = df[df["gold"] == 0]

    # --- Ringkasan ---
    print("\n" + "=" * 57)
    print("RINGKASAN ERROR ANALYSIS — TWITTER FUSION")
    print("=" * 57)
    print(f"\nTotal val items      : {len(texts)}")
    print(f"False Negatives (FN) : {len(fn)}  (gold sarkas, pred tidak)")
    print(f"False Positives (FP) : {len(fp)}  (gold tidak, pred sarkas)")

    if len(fn) > 0:
        n = len(fn)
        clash_fn      = int(fn["has_clash"].sum())
        surface_fn    = int(((fn["has_clash"] == 0) &
                             ((fn["punct_count"] > 0) |
                              (fn["has_emoji"] == 1) |
                              (fn["elongation"] == 1))).sum())
        no_signal_fn  = int(((fn["has_clash"] == 0) &
                              (fn["punct_count"] == 0) &
                              (fn["has_emoji"] == 0) &
                              (fn["elongation"] == 0)).sum())

        print(f"\n--- False Negative breakdown (n={n}) ---")
        print(f"\n  [1] has_clash=True (sinyal ada di leksikon, model melewatkan):")
        print(f"      {clash_fn/n:.1%}  ({clash_fn}/{n})")
        print(f"\n  [2] Penanda permukaan (!/emoji/elongation) tanpa clash:")
        print(f"      {surface_fn/n:.1%}  ({surface_fn}/{n})")
        print(f"\n  [3] Tidak ada sinyal apapun (butuh pengetahuan dunia):")
        print(f"      {no_signal_fn/n:.1%}  ({no_signal_fn}/{n})")
        print(f"\n  Cek: {clash_fn}+{surface_fn}+{no_signal_fn} = {clash_fn+surface_fn+no_signal_fn}/{n}")

    if len(fp) > 0:
        n_fp = len(fp)
        clash_fp    = int(fp["has_clash"].sum())
        no_clash_fp = n_fp - clash_fp
        print(f"\n--- False Positive breakdown (n={n_fp}) ---")
        print(f"\n  [1] has_clash=True (clash terdeteksi tapi gold tidak sarkas):")
        print(f"      {clash_fp/n_fp:.1%}  ({clash_fp}/{n_fp})")
        print(f"\n  [2] has_clash=False:")
        print(f"      {no_clash_fp/n_fp:.1%}  ({no_clash_fp}/{n_fp})")

    print("\n" + "=" * 57)
    print("\nIsi kolom 'kategori_manual' di CSV:")
    print("  clash              → ada kata positif DAN negatif di teks")
    print("  penanda_permukaan  → ada !/emoji/elongation tapi tidak clash")
    print("  pengetahuan_dunia  → tidak ada sinyal, butuh world knowledge")
    print("  lainnya            → kasus lain")


if __name__ == "__main__":
    main()
