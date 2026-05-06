# -*- coding: utf-8 -*-
"""
debug_sentic.py
Analisis statistik .sentic dan .graph.new tanpa training.
Output: stdout + sentic_debug_report.md

Jalankan dari direktori multi_channel_method/:
    python debug_sentic.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "reddit": {
        "csv":    os.path.join(BASE, "real_data", "reddit", "train.csv"),
        "sentic": os.path.join(BASE, "real_data", "reddit", "train.csv.sentic"),
        "dep":    os.path.join(BASE, "real_data", "reddit", "train.csv.graph.new"),
    },
    "twitter": {
        "csv":    os.path.join(BASE, "real_data", "twitter", "train.csv"),
        "sentic": os.path.join(BASE, "real_data", "twitter", "train.csv.sentic"),
        "dep":    os.path.join(BASE, "real_data", "twitter", "train.csv.graph.new"),
    },
}

LEXICON_POS = os.path.join(BASE, "data", "lexicon", "positive.tsv")
LEXICON_NEG = os.path.join(BASE, "data", "lexicon", "negative.tsv")


def load_inset_words():
    """Load InSet lexicon words (lowercase) from positive.tsv and negative.tsv."""
    words = set()
    for path in [LEXICON_POS, LEXICON_NEG]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                word, weight = parts
                try:
                    float(weight)
                    words.add(word.lower())
                except ValueError:
                    continue
    return words


def analyze_dataset(name, paths, lexicon):
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"\n{'='*64}")
    log(f"  DATASET: {name.upper()}")
    log(f"{'='*64}")

    df = pd.read_csv(paths["csv"])
    texts  = df["content"].astype(str).tolist()
    labels = df["label"].tolist()
    n_sar  = sum(labels)
    log(f"  Samples     : {len(texts):,}")
    log(f"  Sarcasm     : {n_sar:,}  ({n_sar/len(labels)*100:.1f}%)")
    log(f"  Non-Sarcasm : {len(labels)-n_sar:,}  ({(len(labels)-n_sar)/len(labels)*100:.1f}%)")

    total   = 0
    matched = 0
    for text in texts:
        toks    = text.lower().split()
        total   += len(toks)
        matched += sum(1 for t in toks if t in lexicon)
    coverage = matched / max(total, 1)

    log(f"\n[a] InSet Coverage (raw-split tokenization):")
    log(f"    Total tokens : {total:,}")
    log(f"    Matched      : {matched:,}")
    log(f"    Coverage     : {coverage*100:.2f}%")
    log(f"    Note: actual coverage lower — sentic_graph.py uses Stanza lemmas, not raw split")

    with open(paths["sentic"], "rb") as f:
        idx2sentic = pickle.load(f)
    with open(paths["dep"], "rb") as f:
        idx2dep = pickle.load(f)

    all_vals = np.concatenate([m.flatten() for m in idx2sentic.values()])
    nonzero  = all_vals[all_vals != 0]

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")]
    hist, _ = (np.histogram(nonzero, bins=bins)
               if len(nonzero) > 0
               else (np.zeros(len(bins)-1, int), None))

    log(f"\n[b] Sentic Matrix Stats (all entries):")
    log(f"    Total entries  : {len(all_vals):,}")
    log(f"    Mean           : {np.mean(all_vals):.6f}")
    log(f"    Std            : {np.std(all_vals):.6f}")
    log(f"    Pct zero       : {np.mean(all_vals==0)*100:.2f}%")
    log(f"    Nonzero count  : {len(nonzero):,}")
    if len(nonzero) > 0:
        log(f"    Nonzero mean   : {np.mean(nonzero):.6f}")
        log(f"    Nonzero std    : {np.std(nonzero):.6f}")
        log(f"    Nonzero min    : {np.min(nonzero):.6f}")
        log(f"    Nonzero max    : {np.max(nonzero):.6f}")
        scale = max(int(max(hist)) // 40, 1)
        log(f"    Histogram nonzero (each # = ~{scale} entries):")
        for i, count in enumerate(hist):
            lo  = bins[i]
            hi  = bins[i + 1]
            bar = "#" * (count // scale)
            lbl = f"  >1.0 " if hi == float("inf") else f"  ({lo:.1f},{hi:.1f})"
            log(f"    {lbl}:  {count:7,}  {bar}")

    dep_vals        = np.concatenate([m.flatten() for m in idx2dep.values()])
    dep_sparsity    = float(np.mean(dep_vals == 0))
    sentic_sparsity = float(np.mean(all_vals == 0))

    log(f"\n[c] Sparsity Comparison:")
    log(f"    Sentic % zero : {sentic_sparsity*100:.2f}%")
    log(f"    Dep    % zero : {dep_sparsity*100:.2f}%")
    log(f"    Sentic is {(sentic_sparsity - dep_sparsity)*100:.2f}pp sparser than dep")

    cls_mean = {0: [], 1: []}
    cls_max  = {0: [], 1: []}

    for i, label in enumerate(labels):
        if i not in idx2sentic:
            continue
        vals = np.abs(idx2sentic[i].flatten())
        cls_mean[label].append(float(np.mean(vals)))
        cls_max[label].append(float(np.max(vals)) if len(vals) > 0 else 0.0)

    log(f"\n[d] Per-Class Sentic Stats (mean absolute magnitude per sample):")
    for cls, cls_name in [(0, "Non-Sarcasm"), (1, "Sarcasm   ")]:
        if cls_mean[cls]:
            log(f"    Class {cls} ({cls_name}):")
            log(f"      Avg mean_mag : {np.mean(cls_mean[cls]):.6f}")
            log(f"      Std mean_mag : {np.std(cls_mean[cls]):.6f}")
            log(f"      Avg max_mag  : {np.mean(cls_max[cls]):.6f}")
        else:
            log(f"    Class {cls}: no samples")

    if cls_mean[0] and cls_mean[1]:
        diff = np.mean(cls_mean[1]) - np.mean(cls_mean[0])
        log(f"\n    Mean-mag diff (sarcasm - non-sarcasm): {diff:+.6f}")
        if abs(diff) < 0.005:
            log(f"    => Very small: sentic signal is CLASS-AGNOSTIC (poor discriminator)")
        else:
            log(f"    => Detectable difference: sentic has some class-separating signal")

    return "\n".join(lines)


def main():
    print("[Load] Reading InSet lexicon...")
    lexicon = load_inset_words()
    print(f"[Load] InSet words: {len(lexicon):,} unique (lowercase)")

    report_lines = [
        "# Sentic Debug Report",
        "",
        f"InSet lexicon: {len(lexicon):,} unique lowercased words",
        "",
    ]

    for name, paths in DATASETS.items():
        section = analyze_dataset(name, paths, lexicon)
        report_lines.append(section)
        report_lines.append("")

    report_path = os.path.join(BASE, "sentic_debug_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[Done] Full report saved: {report_path}")


if __name__ == "__main__":
    main()
