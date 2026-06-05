#!/usr/bin/env python
# coding=utf-8
"""
Analyze cosine similarity distribution between parent_text and body (pure comment)
in the reddit_with_context dataset.

Fixes over v1:
  - Encode `body` (pure comment) instead of `text` (which already bakes [CTX]/[TGT])
  - Clean parent_text: strip [deleted]/[removed] markers before encoding

Usage:
    cd id_sarcasm/
    python scripts/analyze_context_similarity.py

Dependencies (install if missing):
    pip install sentence-transformers matplotlib numpy

Output:
    - Printed statistics to stdout (all splits)
    - cosine_sim_distribution.png  (train split histogram, saved to CWD)
"""

import re
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
DATASET_PATH = "reddit_with_context"   # relative to CWD (id_sarcasm/)
MODEL_NAME   = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE   = 64
OUTPUT_PLOT  = "cosine_sim_distribution.png"
TRUNC        = 200   # character truncation for sample display
# ---------------------------------------------------------------------------

_DELETED_RE = re.compile(r"\[(deleted|removed)[^\]]*\]", re.IGNORECASE)


def clean_parent(text) -> str:
    """Strip [deleted]/[removed] markers; return empty string if nothing left."""
    if text is None:
        return ""
    s = str(text).strip()
    # Treat literal string "None" as no-parent (artefact of dataset serialisation)
    if s.lower() == "none":
        return ""
    s = _DELETED_RE.sub("", s).strip()
    return s


def truncate(text: str, n: int = TRUNC) -> str:
    return text[:n] + "..." if len(text) > n else text


def print_section(title: str) -> None:
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def print_subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def print_sample(rank: int, sim: float, label: int, parent: str, body: str) -> None:
    label_str = "sarcasm" if label == 1 else "non-sarcasm"
    print(f"  [{rank}] sim={sim:.4f}  label={label_str}")
    print(f"       parent: {truncate(parent)}")
    print(f"       body  : {truncate(body)}")


def analyze_split(split_name: str, split_ds, model: SentenceTransformer,
                  do_plot: bool = False, do_samples: bool = False):
    """Run full analysis for one dataset split. Returns sims, labels arrays."""

    print_section(f"{split_name.upper()} SPLIT  (raw size: {len(split_ds)})")

    # ------------------------------------------------------------------
    # 1. Clean parent_text and filter
    # ------------------------------------------------------------------
    raw_parents = [clean_parent(r["parent_text"]) for r in split_ds]
    bodies      = [str(r["body"])                 for r in split_ds]
    labels_all  = [int(r["label"])                for r in split_ds]

    # Count how many became empty after cleaning
    originally_nonempty = sum(
        1 for r in split_ds
        if r["parent_text"] is not None
        and str(r["parent_text"]).strip() != ""
        and str(r["parent_text"]).strip().lower() != "none"
    )
    became_empty = sum(
        1 for raw, cleaned in zip(split_ds["parent_text"], raw_parents)
        if (raw is not None
            and str(raw).strip() != ""
            and str(raw).strip().lower() != "none"
            and cleaned == "")
    )

    mask_has_parent = [p != "" for p in raw_parents]
    parents  = [p for p, m in zip(raw_parents, mask_has_parent) if m]
    comments = [c for c, m in zip(bodies,      mask_has_parent) if m]
    labels   = [l for l, m in zip(labels_all,  mask_has_parent) if m]

    total     = len(split_ds)
    no_parent = total - len(parents)

    print(f"  Total rows            : {total}")
    print(f"  Rows with parent_text : {len(parents)}")
    print(f"  Rows without parent   : {no_parent}")
    print(f"  Became empty (cleaned): {became_empty}  "
          f"(originally non-empty → empty after [deleted]/[removed] strip)")

    if len(parents) == 0:
        print("  [!] No samples with parent_text — skipping this split.")
        return None, None

    # ------------------------------------------------------------------
    # 2. Encode
    # ------------------------------------------------------------------
    print(f"\n  Encoding {len(parents)} parent texts...")
    parent_embs = model.encode(
        parents, batch_size=BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True,
    )
    print(f"  Encoding {len(comments)} body texts...")
    comment_embs = model.encode(
        comments, batch_size=BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # 3. Cosine similarity
    # ------------------------------------------------------------------
    sims       = np.sum(parent_embs * comment_embs, axis=1)
    labels_arr = np.array(labels)

    # ------------------------------------------------------------------
    # 4. Overall statistics
    # ------------------------------------------------------------------
    print_subsection("Overall Statistics")
    print(f"  N       : {len(sims)}")
    print(f"  Min     : {sims.min():.4f}")
    print(f"  Max     : {sims.max():.4f}")
    print(f"  Mean    : {sims.mean():.4f}")
    print(f"  Median  : {np.median(sims):.4f}")
    print(f"  Std     : {sims.std():.4f}")

    # ------------------------------------------------------------------
    # 5. Percentiles
    # ------------------------------------------------------------------
    print_subsection("Percentile Distribution")
    for p in [5, 10, 20, 25, 30, 40, 50, 60, 75, 90]:
        print(f"  P{p:<3d} : {np.percentile(sims, p):.4f}")

    # ------------------------------------------------------------------
    # 6. By label
    # ------------------------------------------------------------------
    print_subsection("Statistics by Label")
    for lbl, name in [(0, "non-sarcasm"), (1, "sarcasm")]:
        mask   = labels_arr == lbl
        subset = sims[mask]
        if mask.sum() == 0:
            print(f"  {name} (label={lbl}): no samples")
            continue
        print(f"  {name} (label={lbl}, n={mask.sum()})")
        print(f"    Mean  : {subset.mean():.4f}")
        print(f"    Median: {np.median(subset):.4f}")
        print(f"    Std   : {subset.std():.4f}")

    # ------------------------------------------------------------------
    # 7. Histogram (train only)
    # ------------------------------------------------------------------
    if do_plot:
        print_subsection("Plotting Histogram")
        fig, ax = plt.subplots(figsize=(10, 5))
        sims_0 = sims[labels_arr == 0]
        sims_1 = sims[labels_arr == 1]
        ax.hist(sims_0, bins=50, alpha=0.5, label="non-sarcasm (0)", color="steelblue")
        ax.hist(sims_1, bins=50, alpha=0.5, label="sarcasm (1)",     color="tomato")
        ax.set_xlabel("Cosine Similarity (parent_text vs body)")
        ax.set_ylabel("Count")
        ax.set_title(
            "Distribution of Parent–Comment Cosine Similarity\n"
            f"({os.path.basename(DATASET_PATH)}, {split_name} split)"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150)
        plt.close()
        print(f"  Saved to: {os.path.abspath(OUTPUT_PLOT)}")

    # ------------------------------------------------------------------
    # 8. Sample inspection (train only)
    # ------------------------------------------------------------------
    if do_samples:
        print_subsection("Sample Inspection")
        sorted_idx = np.argsort(sims)
        n = len(sims)
        mid_start = n // 2 - 1

        groups = [
            ("Bottom 3 (lowest cosine similarity)",  sorted_idx[:3]),
            ("Median 3 (around median)",              sorted_idx[mid_start: mid_start + 3]),
            ("Top 3 (highest cosine similarity)",     sorted_idx[-3:][::-1]),
        ]
        for group_name, indices in groups:
            print(f"\n  >> {group_name}")
            for rank, i in enumerate(indices, 1):
                print_sample(rank, sims[i], labels[i], parents[i], comments[i])

    return sims, labels_arr


def main():
    print_section("Loading Dataset")
    ds = load_from_disk(DATASET_PATH)
    print(f"  Path  : {os.path.abspath(DATASET_PATH)}")
    print(f"  Splits: {list(ds.keys())}")

    print_section("Loading Encoder")
    print(f"  Model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Run analysis for each split; plot + sample inspection only for train
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            print(f"\n[!] Split '{split_name}' not found — skipping.")
            continue
        analyze_split(
            split_name=split_name,
            split_ds=ds[split_name],
            model=model,
            do_plot=(split_name == "train"),
            do_samples=(split_name == "train"),
        )

    print()
    print("=" * 62)
    print("  Done.")
    print("=" * 62)


if __name__ == "__main__":
    main()
