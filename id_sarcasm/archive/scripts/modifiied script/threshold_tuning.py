#!/usr/bin/env python3
"""
Threshold tuning analisis untuk multi-seed run.

Untuk tiap seed:
  1. Sapu threshold 0.10 – 0.90 (step 0.01) di VALIDATION probs.
  2. Pilih threshold yang memaksimalkan F1-binary di val.
  3. Terapkan threshold itu ke TEST probs seed yang sama.
  4. Bandingkan F1 test @0.5 default vs @tuned threshold.

Tidak ada training ulang — hanya membaca val_predictions.json dan
test_predictions.json yang sudah disimpan oleh run_classification_fusion.py.

Usage:
    python scripts/threshold_tuning.py \
        --output_dir /kaggle/working/outputs/twitter-plain-baseline-multiseed \
        --seeds "42,1,2,3,4"
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import f1_score


def sweep_threshold(
    probs: List[float],
    labels: List[int],
    lo: float = 0.10,
    hi: float = 0.90,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Kembalikan (best_threshold, best_f1) yang memaksimalkan F1-binary."""
    best_thr, best_f1 = 0.5, 0.0
    thr = lo
    while thr <= hi + 1e-9:
        preds = [1 if p >= thr else 0 for p in probs]
        f1 = f1_score(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
        thr = round(thr + step, 10)
    return best_thr, best_f1


def f1_at_threshold(probs: List[float], labels: List[int], thr: float) -> float:
    preds = [1 if p >= thr else 0 for p in probs]
    return f1_score(labels, preds, average="binary", zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold tuning analisis")
    parser.add_argument("--output_dir", required=True,
                        help="Root output dir multi-seed run")
    parser.add_argument("--seeds", default="42,1,2,3,4",
                        help="Comma-separated seeds (default: '42,1,2,3,4')")
    args = parser.parse_args()

    base = Path(args.output_dir)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    multi_seed = len(seeds) > 1

    rows = []
    for seed in seeds:
        seed_dir = base / f"seed_{seed}" if multi_seed else base

        val_path  = seed_dir / "val_predictions.json"
        test_path = seed_dir / "test_predictions.json"

        if not val_path.exists():
            print(f"  [seed={seed}] SKIP — val_predictions.json tidak ditemukan: {val_path}")
            continue
        if not test_path.exists():
            print(f"  [seed={seed}] SKIP — test_predictions.json tidak ditemukan: {test_path}")
            continue

        val_data  = json.load(open(val_path,  encoding="utf-8"))
        test_data = json.load(open(test_path, encoding="utf-8"))

        val_probs   = val_data["probs"]
        val_labels  = val_data["labels"]
        test_probs  = test_data["probs"]
        test_labels = test_data["labels"]

        # Pilih threshold dari val
        best_thr, val_f1_tuned = sweep_threshold(val_probs, val_labels)
        val_f1_default = f1_at_threshold(val_probs, val_labels, 0.5)

        # Terapkan ke test
        test_f1_default = f1_at_threshold(test_probs, test_labels, 0.5)
        test_f1_tuned   = f1_at_threshold(test_probs, test_labels, best_thr)

        rows.append({
            "seed":             seed,
            "val_thr":          round(best_thr, 2),
            "val_f1_default":   round(val_f1_default,   4),
            "val_f1_tuned":     round(val_f1_tuned,     4),
            "test_f1_default":  round(test_f1_default,  4),
            "test_f1_tuned":    round(test_f1_tuned,    4),
            "test_delta":       round(test_f1_tuned - test_f1_default, 4),
        })

    if not rows:
        print("Tidak ada seed yang berhasil dibaca.")
        return

    # ---- Cetak tabel ----
    hdr = (f"{'Seed':>6}  {'Val thr':>8}  "
           f"{'Val F1@0.5':>10}  {'Val F1@thr':>10}  "
           f"{'Test F1@0.5':>11}  {'Test F1@thr':>11}  {'Δ test':>8}")
    sep = "-" * len(hdr)
    print(f"\n{hdr}")
    print(sep)
    for r in rows:
        print(
            f"{r['seed']:>6}  {r['val_thr']:>8.2f}  "
            f"{r['val_f1_default']:>10.4f}  {r['val_f1_tuned']:>10.4f}  "
            f"{r['test_f1_default']:>11.4f}  {r['test_f1_tuned']:>11.4f}  "
            f"{r['test_delta']:>+8.4f}"
        )
    print(sep)

    td = [r["test_delta"]       for r in rows]
    tf = [r["test_f1_default"]  for r in rows]
    tt = [r["test_f1_tuned"]    for r in rows]
    print(
        f"{'mean':>6}  {'':>8}  "
        f"{'':>10}  {'':>10}  "
        f"{np.mean(tf):>11.4f}  {np.mean(tt):>11.4f}  "
        f"{np.mean(td):>+8.4f}"
    )
    print(
        f"{'std':>6}  {'':>8}  "
        f"{'':>10}  {'':>10}  "
        f"{np.std(tf):>11.4f}  {np.std(tt):>11.4f}  "
        f"{np.std(td):>8.4f}"
    )

    all_positive = all(d > 0 for d in td)
    print(f"\nΔ test positif semua seed : {all_positive}")
    print(f"Δ test mean ± std         : {np.mean(td):+.4f} ± {np.std(td):.4f}")
    print(f"Test F1@0.5   mean ± std  : {np.mean(tf):.4f} ± {np.std(tf):.4f}")
    print(f"Test F1@tuned mean ± std  : {np.mean(tt):.4f} ± {np.std(tt):.4f}")

    # ---- Simpan hasil ----
    out = {
        "output_dir": str(base),
        "seeds": seeds,
        "per_seed": rows,
        "summary": {
            "test_f1_default_mean": float(np.mean(tf)),
            "test_f1_default_std":  float(np.std(tf)),
            "test_f1_tuned_mean":   float(np.mean(tt)),
            "test_f1_tuned_std":    float(np.std(tt)),
            "delta_mean":           float(np.mean(td)),
            "delta_std":            float(np.std(td)),
            "all_deltas_positive":  all_positive,
        },
    }
    out_path = base / "threshold_tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nHasil disimpan → {out_path}")


if __name__ == "__main__":
    main()
