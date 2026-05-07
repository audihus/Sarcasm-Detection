"""
preprocess_datasets.py
======================
Apply augmentation pipeline ke full dataset, simpan dalam format
yang kompatibel dengan run_classification.py existing.

Output: HuggingFace DatasetDict yang disimpan ke disk.
Struktur output sama dengan input (kolom text + label dipertahankan),
hanya kolom 'text' yang dimodifikasi.

Usage:
    # Preprocess full hybrid (clash + emo_conflict + emoji_expand)
    python preprocess_datasets.py --dataset reddit --variant full_hybrid

    # Ablation: hanya emoji expand (tanpa marker)
    python preprocess_datasets.py --dataset reddit --variant emoji_only

    # Ablation: hanya clash marker
    python preprocess_datasets.py --dataset reddit --variant clash_only

    # Ablation: hanya emo_conflict marker
    python preprocess_datasets.py --dataset reddit --variant emo_conflict_only

    # Preprocess semua dataset dan semua variant sekaligus
    python preprocess_datasets.py --all

Variants yang tersedia:
    full_hybrid     : [CLASH] + [EMO_CONFLICT] + emoji expand (eksperimen utama)
    clash_only      : hanya [CLASH] + emoji expand
    emo_only        : hanya [EMO_CONFLICT] + emoji expand
    emoji_only      : hanya emoji expand, tanpa marker
    clash_no_emoji  : hanya [CLASH], tanpa emoji expand (ablation murni)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.augment_pipeline import AugmentPipeline

try:
    from datasets import load_dataset, DatasetDict, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[ERROR] 'datasets' library tidak tersedia. Install: pip install datasets")


# ===========================================================================
# VARIANT CONFIGURATIONS
# Setiap variant mengaktifkan/menonaktifkan komponen pipeline
# Ini yang akan dipakai untuk ablation study (Tahap E)
# ===========================================================================

VARIANT_CONFIG = {
    "full_hybrid": {
        "use_clash": True,
        "use_emo_conflict": True,
        "use_emoji_expand": True,
        "description": "Full hybrid: [CLASH] + [EMO_CONFLICT] + emoji expansion"
    },
    "clash_only": {
        "use_clash": True,
        "use_emo_conflict": False,
        "use_emoji_expand": True,
        "description": "Clash marker + emoji expansion"
    },
    "emo_only": {
        "use_clash": False,
        "use_emo_conflict": True,
        "use_emoji_expand": True,
        "description": "EMO_CONFLICT marker + emoji expansion"
    },
    "emoji_only": {
        "use_clash": False,
        "use_emo_conflict": False,
        "use_emoji_expand": True,
        "description": "Emoji expansion only, no markers"
    },
    "clash_no_emoji": {
        "use_clash": True,
        "use_emo_conflict": False,
        "use_emoji_expand": False,
        "description": "Clash marker only, no emoji expansion"
    },
}

DATASET_CONFIG = {
    "reddit": {
        "hf_name": "w11wo/reddit_indonesia_sarcastic",
        "text_col": "text",
        "label_col": "label",
        "inset_base": "real_data/reddit",
    },
    "twitter": {
        "hf_name": "w11wo/twitter_indonesia_sarcastic",
        "text_col": "tweet",
        "label_col": "label",
        "inset_base": "real_data/twitter",
    }
}


# ===========================================================================
# CORE PREPROCESSING FUNCTION
# ===========================================================================

def preprocess_dataset(
    dataset_name: str,
    variant: str,
    project_root: str = ".",
    output_base_dir: str = "preprocessed_data",
) -> str:
    """
    Load dataset dari HuggingFace, apply augmentation, simpan ke disk.

    Args:
        dataset_name: 'reddit' atau 'twitter'
        variant: salah satu dari VARIANT_CONFIG
        project_root: root directory project
        output_base_dir: base directory untuk output

    Returns:
        path ke dataset yang sudah disimpan
    """
    if not DATASETS_AVAILABLE:
        sys.exit(1)

    assert dataset_name in DATASET_CONFIG, f"Unknown dataset: {dataset_name}"
    assert variant in VARIANT_CONFIG, f"Unknown variant: {variant}. Pilihan: {list(VARIANT_CONFIG.keys())}"

    project_root = Path(project_root)
    cfg = DATASET_CONFIG[dataset_name]
    var_cfg = VARIANT_CONFIG[variant]

    print(f"\n{'='*60}")
    print(f"Preprocessing: {dataset_name.upper()} | Variant: {variant}")
    print(f"Config: {var_cfg['description']}")
    print(f"{'='*60}")

    # Resolve InSet path
    pos_path = project_root / cfg["inset_base"] / "positive.tsv"
    neg_path = project_root / cfg["inset_base"] / "negative.tsv"

    if not pos_path.exists():
        # Fallback ke dataset lain
        for other in ["reddit", "twitter"]:
            fp = project_root / f"real_data/{other}/positive.tsv"
            fn = project_root / f"real_data/{other}/negative.tsv"
            if fp.exists() and fn.exists():
                print(f"[WARN] InSet fallback: {fp}")
                pos_path, neg_path = fp, fn
                break

    if not pos_path.exists():
        print(f"[ERROR] InSet lexicon tidak ditemukan: {pos_path}")
        sys.exit(1)

    # Inisialisasi pipeline
    pipeline = AugmentPipeline(
        positive_path=str(pos_path),
        negative_path=str(neg_path),
        use_clash=var_cfg["use_clash"],
        use_emo_conflict=var_cfg["use_emo_conflict"],
        use_emoji_expand=var_cfg["use_emoji_expand"],
    )

    # Load dataset
    print(f"\n[INFO] Loading '{cfg['hf_name']}'...")
    ds = load_dataset(cfg["hf_name"])

    # Apply preprocessing ke setiap split
    def augment_split(examples):
        """Batch map function untuk HuggingFace datasets."""
        texts = examples[cfg["text_col"]]
        augmented_texts = []

        for text in texts:
            if text and isinstance(text, str) and text.strip():
                aug, _ = pipeline.augment(text)
                augmented_texts.append(aug)
            else:
                augmented_texts.append(text or "")

        return {cfg["text_col"]: augmented_texts}

    processed_splits = {}
    for split_name, split_data in ds.items():
        print(f"[INFO] Processing split '{split_name}' ({len(split_data)} samples)...")
        processed = split_data.map(
            augment_split,
            batched=True,
            batch_size=512,
            desc=f"Augmenting {split_name}",
        )
        processed_splits[split_name] = processed
        print(f"[INFO] Split '{split_name}' done.")

        # Quick sanity check: tampilkan 3 contoh
        print("\n  Sample check (3 contoh):")
        for i in range(min(3, len(processed))):
            orig_text = split_data[cfg["text_col"]][i]
            aug_text = processed[cfg["text_col"]][i]
            label = processed[cfg["label_col"]][i]
            if orig_text != aug_text:  # hanya tampilkan yang berubah
                print(f"  [{i}] ORIG: {orig_text[:80]}")
                print(f"  [{i}] AUG : {aug_text[:80]}")
                print(f"  [{i}] LABEL: {label}")
                print()

    processed_ds = DatasetDict(processed_splits)

    # Simpan ke disk
    output_path = Path(output_base_dir) / dataset_name / variant
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ds.save_to_disk(str(output_path))

    print(f"\n[INFO] Dataset disimpan ke: {output_path}")
    print(f"[INFO] Gunakan di run_classification.py dengan:")
    print(f"       --dataset_name {output_path} --dataset_config None")
    print(f"       (atau sesuai implementasi load_dataset di script kamu)")

    return str(output_path)


def verify_saved_dataset(saved_path: str, dataset_name: str):
    """
    Verifikasi dataset yang sudah disimpan bisa di-load dan formatnya benar.
    """
    from datasets import load_from_disk

    print(f"\n[INFO] Verifying saved dataset: {saved_path}")
    ds = load_from_disk(saved_path)

    cfg = DATASET_CONFIG[dataset_name]
    for split_name, split_data in ds.items():
        n = len(split_data)
        cols = split_data.column_names
        print(f"  Split '{split_name}': {n} samples, columns: {cols}")

        # Cek kolom yang dibutuhkan ada
        assert cfg["text_col"] in cols, f"Missing column: {cfg['text_col']}"
        assert cfg["label_col"] in cols, f"Missing column: {cfg['label_col']}"

        # Cek beberapa augmented teks mengandung marker atau expanded emoji
        sample_texts = split_data[cfg["text_col"]][:20]
        n_clash = sum(1 for t in sample_texts if "[CLASH]" in str(t))
        n_emo = sum(1 for t in sample_texts if "[EMO_CONFLICT]" in str(t))
        n_emoji_expanded = sum(1 for t in sample_texts if "emoji_" in str(t))
        print(f"    (dari 20 sampel pertama: {n_clash} [CLASH], {n_emo} [EMO_CONFLICT], "
              f"{n_emoji_expanded} emoji_*)")

    print("[INFO] Verification OK.\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess IdSarcasm datasets dengan text augmentation pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=["reddit", "twitter"],
        help="Dataset yang ingin dipreprocess"
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_CONFIG.keys()),
        default="full_hybrid",
        help=f"Variant preprocessing (default: full_hybrid). Pilihan: {list(VARIANT_CONFIG.keys())}"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Preprocess semua dataset dan semua variant"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Root directory project (default: current dir)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="preprocessed_data",
        help="Output directory untuk dataset yang sudah dipreprocess"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verifikasi dataset setelah disimpan (default: True)"
    )

    args = parser.parse_args()

    if args.all:
        # Preprocess semua kombinasi
        print("[INFO] Mode --all: preprocess semua dataset x semua variant")
        jobs = [
            (ds, var)
            for ds in ["reddit", "twitter"]
            for var in VARIANT_CONFIG.keys()
        ]
    else:
        if not args.dataset:
            parser.error("--dataset harus diisi jika tidak pakai --all")
        jobs = [(args.dataset, args.variant)]

    saved_paths = []
    for dataset_name, variant in jobs:
        saved_path = preprocess_dataset(
            dataset_name=dataset_name,
            variant=variant,
            project_root=args.project_root,
            output_base_dir=args.output_dir,
        )
        saved_paths.append((dataset_name, variant, saved_path))

        if args.verify:
            verify_saved_dataset(saved_path, dataset_name)

    # Summary semua yang sudah diproses
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    for ds, var, path in saved_paths:
        print(f"  {ds:10s} | {var:20s} | {path}")
    print()
    print("Langkah berikutnya:")
    print("  1. Review sanity_check_report_{dataset}.md")
    print("  2. Jika GO, jalankan run_classification.py dengan dataset yang sudah dipreprocess")
    print("  3. Untuk ablation study, jalankan training untuk setiap variant")


if __name__ == "__main__":
    main()