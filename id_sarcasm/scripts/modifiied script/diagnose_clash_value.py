#!/usr/bin/env python3
"""
Diagnosa nilai fitur clash terhadap plain IndoBERT baseline.

Pertanyaan inti:
  A. Dari item test yang clash=True DAN gold=sarcasm, berapa persen sudah ditangkap
     benar oleh plain baseline (tanpa fitur clash)?
  B. Dari item test yang clash=True DAN gold=non-sarcasm, berapa persen plain salah
     tebak jadi sarcasm (false positive)?

Kalau A tinggi → clash redundan untuk deteksi sarkas (BERT sudah tangkap sendiri).
Kalau B tinggi → clash korelasi palsu yang bikin plain banyak FP.
Kalau A rendah → clash masih punya ruang memberi sinyal kalau dideteksi benar.

Usage (Kaggle):
    python scripts/diagnose_clash_value.py \
        --output_dir /kaggle/working/outputs/twitter-plain-baseline-multiseed \
        --inset_pos_path /kaggle/working/sarcasm/real_data/twitter/positive.tsv \
        --inset_neg_path /kaggle/working/sarcasm/real_data/twitter/negative.tsv \
        --seeds "42,1,2,3,4"

Usage (lokal, seed 42 saja):
    python scripts/diagnose_clash_value.py \
        --output_dir outputs/twitter-plain-baseline-multiseed \
        --inset_pos_path real_data/twitter/positive.tsv \
        --inset_neg_path real_data/twitter/negative.tsv \
        --seeds "42"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.augment_pipeline import detect_polarity_clash, load_inset_lexicon


# ---------------------------------------------------------------------------
# Per-seed analysis
# ---------------------------------------------------------------------------

def analyse_seed(
    texts: List[str],
    preds: List[int],
    labels: List[int],
    inset_pos: frozenset,
    inset_neg: frozenset,
) -> Dict:
    """Hitung clash label tiap item, lalu pisahkan ke grup clash/non-clash."""
    clash_flags = [
        detect_polarity_clash(t, inset_pos, inset_neg)[0] for t in texts
    ]

    groups: Dict[str, Dict] = {
        "clash_true":  {"sar_correct": 0, "sar_miss": 0, "non_correct": 0, "non_fp": 0},
        "clash_false": {"sar_correct": 0, "sar_miss": 0, "non_correct": 0, "non_fp": 0},
    }

    for clash, pred, label in zip(clash_flags, preds, labels):
        grp = groups["clash_true"] if clash else groups["clash_false"]
        if label == 1:                 # gold = sarcasm
            if pred == 1:
                grp["sar_correct"] += 1
            else:
                grp["sar_miss"] += 1
        else:                          # gold = non-sarcasm
            if pred == 0:
                grp["non_correct"] += 1
            else:
                grp["non_fp"] += 1

    result = {"n_total": len(texts), "clash_flags": clash_flags}
    for key, g in groups.items():
        n_sar  = g["sar_correct"] + g["sar_miss"]
        n_non  = g["non_correct"] + g["non_fp"]
        n_grp  = n_sar + n_non
        result[key] = {
            "n":             n_grp,
            "pct_of_total":  n_grp / len(texts) * 100 if texts else 0,
            "n_sar":         n_sar,
            "n_non":         n_non,
            # A: recall-for-sarcasm dalam grup clash
            "sar_recall":    g["sar_correct"] / n_sar * 100 if n_sar else float("nan"),
            # B: false-positive-rate-for-non-sarcasm dalam grup clash
            "non_fpr":       g["non_fp"]      / n_non * 100 if n_non else float("nan"),
            "sar_correct":   g["sar_correct"],
            "sar_miss":      g["sar_miss"],
            "non_correct":   g["non_correct"],
            "non_fp":        g["non_fp"],
        }
    return result


def print_seed_report(seed: int, r: Dict) -> None:
    ct = r["clash_true"]
    cf = r["clash_false"]
    sep = "-" * 60
    print(f"\n{'='*60}")
    print(f"  Seed {seed}  |  Test set: {r['n_total']} item")
    print(f"{'='*60}")
    print(f"\n[CLASH = TRUE]  {ct['n']} item  ({ct['pct_of_total']:.1f}% dari test set)")
    print(sep)
    print(f"  Gold=sarcasm  ({ct['n_sar']} item):")
    print(f"    Benar (pred=sar)  : {ct['sar_correct']:4d}  → {ct['sar_recall']:5.1f}%  ← [A]")
    print(f"    Salah (pred=non)  : {ct['sar_miss']:4d}")
    print(f"  Gold=non-sarcasm  ({ct['n_non']} item):")
    print(f"    Benar (pred=non)  : {ct['non_correct']:4d}")
    print(f"    False-pos (pred=sar): {ct['non_fp']:4d}  → {ct['non_fpr']:5.1f}%  ← [B]")

    print(f"\n[CLASH = FALSE]  {cf['n']} item  ({cf['pct_of_total']:.1f}% dari test set)")
    print(sep)
    print(f"  Gold=sarcasm  ({cf['n_sar']} item):")
    print(f"    Benar (pred=sar)  : {cf['sar_correct']:4d}  → {cf['sar_recall']:5.1f}%")
    print(f"    Salah (pred=non)  : {cf['sar_miss']:4d}")
    print(f"  Gold=non-sarcasm  ({cf['n_non']} item):")
    print(f"    Benar (pred=non)  : {cf['non_correct']:4d}")
    print(f"    False-pos (pred=sar): {cf['non_fp']:4d}  → {cf['non_fpr']:5.1f}%")

    print(f"\n  Interpretasi [A]: plain BERT benar di {ct['sar_recall']:.1f}% kasus clash+sarkas")
    print(f"  Interpretasi [B]: plain BERT salah tebak sarkas di {ct['non_fpr']:.1f}% kasus clash+non-sarkas")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnosa nilai fitur clash vs plain baseline")
    parser.add_argument("--output_dir", required=True,
                        help="Root output dir multi-seed plain baseline run")
    parser.add_argument("--inset_pos_path", required=True,
                        help="Path ke positive.tsv InSet")
    parser.add_argument("--inset_neg_path", required=True,
                        help="Path ke negative.tsv InSet")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds (default: '42')")
    args = parser.parse_args()

    base = Path(args.output_dir)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    multi_seed = len(seeds) > 1

    print("Loading InSet lexicon...")
    inset_pos, inset_neg = load_inset_lexicon(args.inset_pos_path, args.inset_neg_path)

    all_results = []
    for seed in seeds:
        seed_dir = base / f"seed_{seed}" if multi_seed else base
        test_path = seed_dir / "test_predictions.json"
        if not test_path.exists():
            print(f"\n[seed={seed}] SKIP — test_predictions.json tidak ada: {test_path}")
            continue

        data = json.load(open(test_path, encoding="utf-8"))
        texts  = data["texts"]
        preds  = data["preds"]
        labels = data["labels"]

        print(f"\nSeed {seed}: computing clash labels untuk {len(texts)} item test...")
        r = analyse_seed(texts, preds, labels, inset_pos, inset_neg)
        r["seed"] = seed
        all_results.append(r)
        print_seed_report(seed, r)

    if not all_results:
        print("\nTidak ada seed yang berhasil dibaca.")
        return

    # ---- Ringkasan agregat (kalau >1 seed) ----
    if len(all_results) > 1:
        sar_recalls = [r["clash_true"]["sar_recall"] for r in all_results
                       if not np.isnan(r["clash_true"]["sar_recall"])]
        non_fprs    = [r["clash_true"]["non_fpr"]    for r in all_results
                       if not np.isnan(r["clash_true"]["non_fpr"])]

        print(f"\n{'='*60}")
        print(f"  AGREGAT {len(all_results)} seed")
        print(f"{'='*60}")
        print(f"  [A] Recall sarkas di clash=True  : {np.mean(sar_recalls):.1f}% ± {np.std(sar_recalls):.1f}%")
        print(f"  [B] FPR non-sarkas di clash=True : {np.mean(non_fprs):.1f}% ± {np.std(non_fprs):.1f}%")

    # ---- Simpan JSON ----
    out_path = base / "clash_diagnosis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        # Buang clash_flags (besar) sebelum disimpan
        save_results = []
        for r in all_results:
            r_copy = {k: v for k, v in r.items() if k != "clash_flags"}
            save_results.append(r_copy)
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nHasil disimpan → {out_path}")

    # ---- Jawaban eksplisit ----
    print(f"\n{'='*60}")
    print("  JAWABAN EKSPLISIT")
    print(f"{'='*60}")
    if sar_recalls if len(all_results) > 1 else True:
        seed_r = all_results[0] if len(all_results) == 1 else None
        a_val = np.mean(sar_recalls) if len(all_results) > 1 else all_results[0]["clash_true"]["sar_recall"]
        b_val = np.mean(non_fprs)    if len(all_results) > 1 else all_results[0]["clash_true"]["non_fpr"]
        print(f"  [A] Plain BERT tangkap {a_val:.1f}% item clash+sarkas → ", end="")
        if a_val >= 70:
            print("clash REDUNDAN untuk sarkas (BERT sudah paham tanpa fitur)")
        elif a_val >= 50:
            print("clash MASIH ADA RUANG (BERT setengah-setengah, fitur bisa bantu)")
        else:
            print("clash PUNYA NILAI (BERT sering gagal di sini)")
        print(f"  [B] Plain BERT salah tebak {b_val:.1f}% item clash+non-sarkas → ", end="")
        if b_val >= 30:
            print("clash memicu FP (fitur clash harusnya menurunkan kepercayaan, bukan menaikkan)")
        else:
            print("FP akibat clash relatif kecil")


if __name__ == "__main__":
    main()
