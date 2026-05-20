#!/usr/bin/env python
# coding=utf-8
"""
Analyze cosine similarity distribution between parent_text and comment text
in the reddit_with_context dataset.

Usage:
    python scripts/analyze_context_similarity.py

Dependencies (install if missing):
    pip install sentence-transformers matplotlib numpy

Output:
    - Printed statistics to stdout
    - cosine_sim_distribution.png in the current working directory
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config — adjust paths here if needed
# ---------------------------------------------------------------------------
DATASET_PATH = "reddit_with_context"  # relative to working directory (id_sarcasm/)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 64
OUTPUT_PLOT = "cosine_sim_distribution.png"
TRUNC = 200  # character truncation for sample display
# ---------------------------------------------------------------------------


def truncate(text: str, n: int = TRUNC) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "..."


def print_section(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def print_sample(idx: int, sim: float, label: int, parent: str, comment: str) -> None:
    label_str = "sarcasm" if label == 1 else "non-sarcasm"
    print(f"  [{idx}] sim={sim:.4f}  label={label_str}")
    print(f"       parent : {truncate(parent)}")
    print(f"       comment: {truncate(comment)}")


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print_section("Loading Dataset")
    ds = load_from_disk(DATASET_PATH)
    print(f"  Splits: {list(ds.keys())}")
    train = ds["train"]
    print(f"  Train size (before filter): {len(train)}")

    # ------------------------------------------------------------------
    # 2. Filter samples with non-empty parent_text
    # ------------------------------------------------------------------
    def has_parent(example):
        pt = example["parent_text"]
        return pt is not None and str(pt).strip() != ""

    train_filtered = train.filter(has_parent)
    print(f"  Train size (after filter, has parent_text): {len(train_filtered)}")

    parents = [str(x) for x in train_filtered["parent_text"]]
    comments = [str(x) for x in train_filtered["text"]]
    labels = list(train_filtered["label"])

    # ------------------------------------------------------------------
    # 3. Encode with sentence-transformers
    # ------------------------------------------------------------------
    print_section("Encoding Texts")
    print(f"  Model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("  Encoding parent_text...")
    parent_embs = model.encode(
        parents,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("  Encoding comment text...")
    comment_embs = model.encode(
        comments,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # 4. Cosine similarity (dot product on L2-normalised embeddings)
    # ------------------------------------------------------------------
    sims = np.sum(parent_embs * comment_embs, axis=1)  # shape: (N,)
    labels_arr = np.array(labels)

    # ------------------------------------------------------------------
    # 5. Overall distribution statistics
    # ------------------------------------------------------------------
    print_section("Overall Distribution Statistics")
    print(f"  N samples : {len(sims)}")
    print(f"  Min       : {sims.min():.4f}")
    print(f"  Max       : {sims.max():.4f}")
    print(f"  Mean      : {sims.mean():.4f}")
    print(f"  Median    : {np.median(sims):.4f}")
    print(f"  Std       : {sims.std():.4f}")

    # ------------------------------------------------------------------
    # 6. Percentiles
    # ------------------------------------------------------------------
    print_section("Percentile Distribution")
    percentiles = [5, 10, 20, 25, 30, 40, 50, 60, 75, 90]
    for p in percentiles:
        print(f"  P{p:<3d} : {np.percentile(sims, p):.4f}")

    # ------------------------------------------------------------------
    # 7. By label
    # ------------------------------------------------------------------
    print_section("Statistics by Label")
    for lbl, name in [(0, "non-sarcasm"), (1, "sarcasm")]:
        mask = labels_arr == lbl
        subset = sims[mask]
        print_subsection(f"{name} (label={lbl}, n={mask.sum()})")
        print(f"  Mean   : {subset.mean():.4f}")
        print(f"  Median : {np.median(subset):.4f}")
        print(f"  Std    : {subset.std():.4f}")

    # ------------------------------------------------------------------
    # 8. Plot histogram overlay
    # ------------------------------------------------------------------
    print_section("Plotting Histogram")
    fig, ax = plt.subplots(figsize=(10, 5))

    sims_sarcasm = sims[labels_arr == 1]
    sims_nonsarc = sims[labels_arr == 0]

    ax.hist(sims_nonsarc, bins=50, alpha=0.5, label="non-sarcasm (0)", color="steelblue")
    ax.hist(sims_sarcasm, bins=50, alpha=0.5, label="sarcasm (1)", color="tomato")

    ax.set_xlabel("Cosine Similarity (parent vs comment)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Parent–Comment Cosine Similarity\n(reddit_with_context, train split)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"  Saved to: {os.path.abspath(OUTPUT_PLOT)}")

    # ------------------------------------------------------------------
    # 9. Sample inspection
    # ------------------------------------------------------------------
    print_section("Sample Inspection")
    sorted_idx = np.argsort(sims)  # ascending

    n = len(sims)
    mid_start = n // 2 - 1

    groups = [
        ("Bottom 3 (lowest cosine similarity)", sorted_idx[:3]),
        ("Median 3 (around median)", sorted_idx[mid_start : mid_start + 3]),
        ("Top 3 (highest cosine similarity)", sorted_idx[-3:][::-1]),
    ]

    for group_name, indices in groups:
        print_subsection(group_name)
        for rank, i in enumerate(indices, 1):
            print_sample(rank, sims[i], labels[i], parents[i], comments[i])

    print()
    print("Done.")


if __name__ == "__main__":
    main()
