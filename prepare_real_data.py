# -*- coding: utf-8 -*-
"""
prepare_real_data.py
====================
Konversi dataset Arrow (HuggingFace) ke CSV + generate dependency/sentic graphs.

Jalankan sekali per dataset sebelum training:
    python prepare_real_data.py --dataset reddit  --output_dir ./real_data/reddit
    python prepare_real_data.py --dataset twitter --output_dir ./real_data/twitter

Output per split (train / validation / test):
    {output_dir}/{split}.csv            <- teks + label
    {output_dir}/{split}.csv.graph.new  <- dependency adjacency matrix
    {output_dir}/{split}.csv.sentic     <- sentiment adjacency matrix

CATATAN: Saat pertama kali dijalankan, Stanza akan mengunduh model Indonesian
(~500MB) ke ~/stanza_resources/. Pastikan koneksi internet aktif.
Graph files menggunakan format pickle biner yang sudah dipakai oleh
dependency_graph.py dan sentic_graph.py (format internal proyek ini).
"""

import os
import sys
import shutil
import argparse

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR    = os.path.join(SCRIPT_DIR, "multichannel-sarcasm-detection")
LEXICON_DIR = os.path.join(SCRIPT_DIR, "data", "lexicon")

DATASET_PATHS = {
    "reddit":  os.path.join(SCRIPT_DIR, "data", "reddit_indonesia_sarcastic"),
    "twitter": os.path.join(SCRIPT_DIR, "data", "twitter_indonesia_sarcastic"),
}
TEXT_COLUMNS = {
    "reddit":  "text",
    "twitter": "tweet",
}


def export_csvs(dataset_name: str, output_dir: str, splits: list) -> dict:
    """
    Load Arrow dataset dan export satu CSV per split.
    Kolom teks di-rename ke 'content' agar konsisten dengan SarcasmDataset.
    Returns dict {split_name: csv_path}.
    """
    from datasets import load_from_disk

    arrow_path = DATASET_PATHS[dataset_name]
    text_col   = TEXT_COLUMNS[dataset_name]

    print(f"\n[Load] Membaca Arrow dataset dari: {arrow_path}")
    ds = load_from_disk(arrow_path)

    csv_paths = {}
    for split in splits:
        if split not in ds:
            print(f"  [SKIP] Split '{split}' tidak ada di dataset.")
            continue

        df = ds[split].to_pandas()
        df = df[[text_col, "label"]].rename(columns={text_col: "content"})
        df = df.reset_index(drop=True)

        csv_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        csv_paths[split] = csv_path

        dist = df["label"].value_counts().to_dict()
        print(f"  [{split}] {len(df)} rows -> {csv_path}  (label dist: {dist})")

    return csv_paths


def copy_lexicon(output_dir: str):
    """
    Salin positive.tsv dan negative.tsv ke output_dir.
    sentic_graph.py mencari file ini di CWD (kita os.chdir ke output_dir sebelum proses).
    """
    for fname in ["positive.tsv", "negative.tsv"]:
        src = os.path.join(LEXICON_DIR, fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.exists(dst):
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  [Lexikon] Disalin: {fname}")
            else:
                print(f"  [Lexikon] PERINGATAN: {src} tidak ditemukan, "
                      "sentic_graph.py akan mengunduh dari GitHub.")


def generate_graphs(csv_paths: dict, output_dir: str, skip_existing: bool):
    """
    Generate .graph.new (dependency) dan .sentic (sentiment) untuk setiap CSV.
    Menggunakan Stanza Indonesian pipeline.
    Format output mengikuti konvensi dependency_graph.py dan sentic_graph.py.
    """
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    import dependency_graph as dep_mod
    import sentic_graph     as sen_mod

    abs_output_dir = os.path.abspath(output_dir)
    abs_csv_paths  = {split: os.path.abspath(p) for split, p in csv_paths.items()}
    orig_cwd = os.getcwd()
    os.chdir(abs_output_dir)   # sentic_graph mencari lexicon di CWD

    try:
        for split, csv_path in abs_csv_paths.items():
            graph_path  = os.path.join(abs_output_dir, f"{split}.csv.graph.new")
            sentic_path = os.path.join(abs_output_dir, f"{split}.csv.sentic")

            print(f"\n[Graph] Memproses split '{split}' ...")

            if skip_existing and os.path.exists(graph_path):
                print(f"  [SKIP] Dependency graph sudah ada: {graph_path}")
            else:
                print(f"  -> Membuat dependency graph ...")
                dep_mod.process(csv_path, "content", graph_path)
                print(f"  OK Tersimpan: {graph_path}")

            if skip_existing and os.path.exists(sentic_path):
                print(f"  [SKIP] Sentic graph sudah ada: {sentic_path}")
            else:
                print(f"  -> Membuat sentic graph ...")
                sen_mod.process(csv_path, "content", sentic_path)
                print(f"  OK Tersimpan: {sentic_path}")
    finally:
        os.chdir(orig_cwd)


def print_training_command(dataset_name: str, output_dir: str, splits: list):
    has_val = "validation" in splits
    val_lines = (
        f"  --val_data        \"{os.path.join(output_dir, 'validation.csv')}\" \\\n"
        f"  --val_split_name   validation.csv \\\n"
    ) if has_val else ""

    print(f"""
{'=' * 65}
  Data siap! Jalankan training dengan perintah:
{'=' * 65}
python train_multichannel.py \\
  --train_data       "{os.path.join(output_dir, 'train.csv')}" \\
{val_lines}  --test_data        "{os.path.join(output_dir, 'test.csv')}" \\
  --graph_dir        "{output_dir}" \\
  --train_split_name   train.csv \\
  --test_split_name    test.csv \\
  --epochs 100 --batch_size 32 \\
  --checkpoint_dir   ./ckpt_{dataset_name}/
{'=' * 65}
""")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Persiapan data asli: Arrow -> CSV + graph files"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["reddit", "twitter"],
        help="Dataset yang diproses"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Folder output (default: ./real_data/{dataset})"
    )
    parser.add_argument(
        "--splits", default="train,validation,test",
        help="Split yang diproses, pisahkan koma"
    )
    parser.add_argument(
        "--skip_graphs", action="store_true",
        help="Skip generate graph jika file sudah ada"
    )
    parser.add_argument(
        "--csv_only", action="store_true",
        help="Hanya export CSV, skip generate graph"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "real_data", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("=" * 65)
    print(f"  prepare_real_data.py  --  dataset: {args.dataset}")
    print("=" * 65)
    print(f"  Output dir : {output_dir}")
    print(f"  Splits     : {splits}")

    csv_paths = export_csvs(args.dataset, output_dir, splits)

    if args.csv_only:
        print("\n[--csv_only] Selesai. Graph tidak di-generate.")
        print_training_command(args.dataset, output_dir, splits)
        return

    print("\n[Lexikon] Menyalin file lexikon InSet ...")
    copy_lexicon(output_dir)

    generate_graphs(csv_paths, output_dir, skip_existing=args.skip_graphs)

    print_training_command(args.dataset, output_dir, splits)


if __name__ == "__main__":
    main()
