"""
preprocess_datasets.py
======================
Apply text augmentation variants to sarcasm datasets and save as DatasetDict.

Usage:
    python scripts/preprocess_datasets.py \
        --dataset reddit \
        --variant structural_reddit \
        --project_root .

Output: preprocessed_data/{dataset}/{variant}/ (DatasetDict, load_from_disk-compatible)
"""

import argparse
import sys
from pathlib import Path

# Project root = parent of the scripts/ directory, regardless of cwd.
# This makes the script location-independent on Kaggle, Colab, and local.
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_DEFAULT_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "reddit": {
        "hub_name": "w11wo/reddit_indonesia_sarcastic",
        "hub_text_col": "text",
        "local_csv_dir": "real_data/reddit",
        "local_text_col": "content",
    },
    "twitter": {
        "hub_name": "w11wo/twitter_indonesia_sarcastic",
        "hub_text_col": "tweet",
        "local_csv_dir": "real_data/twitter",
        "local_text_col": "content",
    },
}

# VARIANT_CONFIG is built after importing add_structural_markers.
# Each entry: { "applies_to": dataset_name, "fn": text -> text }
# All non-structural pipeline components (clash, emo_conflict, emoji_expand) are OFF.
def _build_variant_config(add_structural_markers_fn):
    return {
        "structural_reddit": {
            "applies_to": "reddit",
            "fn": lambda text: add_structural_markers_fn(text, "reddit"),
        },
        "structural_twitter": {
            "applies_to": "twitter",
            "fn": lambda text: add_structural_markers_fn(text, "twitter"),
        },
        "structural_reddit_twitter_style": {
            "applies_to": "reddit",
            "fn": lambda text: add_structural_markers_fn(text, "twitter"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess sarcasm datasets with augmentation variants.")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()),
                        help="Dataset to preprocess: reddit or twitter")
    parser.add_argument("--variant", required=True,
                        help="Variant name (e.g. structural_reddit, structural_twitter)")
    parser.add_argument("--project_root", default=".",
                        help="Root directory of the id_sarcasm project")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    from datasets import load_dataset
    from preprocessing.augment_pipeline import add_structural_markers

    VARIANT_CONFIG = _build_variant_config(add_structural_markers)

    if args.variant not in VARIANT_CONFIG:
        raise ValueError(
            f"Unknown variant: '{args.variant}'. "
            f"Available: {list(VARIANT_CONFIG.keys())}"
        )

    variant_cfg = VARIANT_CONFIG[args.variant]
    dataset_cfg = DATASET_CONFIG[args.dataset]

    if variant_cfg["applies_to"] != args.dataset:
        raise ValueError(
            f"Variant '{args.variant}' is for dataset '{variant_cfg['applies_to']}', "
            f"but --dataset={args.dataset} was specified. "
            f"Use --variant structural_{args.dataset} for this dataset."
        )

    # --- Load data ---
    local_csv_dir = project_root / dataset_cfg["local_csv_dir"]
    train_csv = local_csv_dir / "train.csv"

    if train_csv.exists():
        print(f"[preprocess] Loading from local CSV: {local_csv_dir}")
        raw = load_dataset("csv", data_files={
            "train": str(local_csv_dir / "train.csv"),
            "validation": str(local_csv_dir / "validation.csv"),
            "test": str(local_csv_dir / "test.csv"),
        })
        text_col = dataset_cfg["local_text_col"]
    else:
        hub_name = dataset_cfg["hub_name"]
        print(f"[preprocess] Local CSV not found; loading from Hub: {hub_name}")
        raw = load_dataset(hub_name)
        text_col = dataset_cfg["hub_text_col"]

    # Normalize column name to "text" so downstream command is always --text_column_names text
    if text_col != "text":
        print(f"[preprocess] Renaming column '{text_col}' -> 'text'")
        raw = raw.rename_column(text_col, "text")

    # --- Apply variant transform ---
    transform_fn = variant_cfg["fn"]
    print(f"[preprocess] Applying variant '{args.variant}' (dataset={args.dataset})...")

    def _apply(batch):
        batch["text"] = [transform_fn(t) if isinstance(t, str) else t for t in batch["text"]]
        return batch

    processed = raw.map(_apply, batched=True, desc=f"Applying {args.variant}")

    # --- Save ---
    out_dir = project_root / "preprocessed_data" / args.dataset / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[preprocess] Saving DatasetDict to {out_dir}")
    processed.save_to_disk(str(out_dir))

    # --- Coverage summary ---
    print("\n[preprocess] Coverage summary:")
    for split in processed.keys():
        n = len(processed[split])
        orig_texts = raw[split]["text"]
        proc_texts = processed[split]["text"]
        n_marked = sum(1 for o, p in zip(orig_texts, proc_texts) if o != p)
        print(f"  {split:12s}: {n:5d} rows | {n_marked:4d} marked ({100*n_marked/n:.1f}%)")

    print("\n[preprocess] Done.")


if __name__ == "__main__":
    main()
