#!/usr/bin/env python3
"""
Siapkan SmSA (IndoNLU) sebagai tugas-antara.

Kenapa: tahap-antara butuh data sentimen yang kaya. Teksnya kita lewatkan
to_encoder_text yang SAMA dengan tahap sarkasme, supaya representasi input
konsisten antar-tahap (encoder tidak kaget berganti format di tahap 2).

Output: smsa_train.csv, smsa_validation.csv, smsa_test.csv (kolom: content, label).

Jalankan:
  python prepare_smsa.py --out_dir preprocessed_data/smsa_ready
"""
import argparse
import os

import pandas as pd

from sarcasm_preprocess import to_encoder_text

TEXT_CANDIDATES = ["text", "sentence", "tweet", "content", "kalimat"]


def pick_text_col(cols, override):
    if override:
        return override
    for c in TEXT_CANDIDATES:
        if c in cols:
            return c
    raise SystemExit(f"Kolom teks tidak terdeteksi dari {cols}. Pakai --text_field.")


def main():
    from datasets import load_dataset  # lazy: hanya perlu saat unduh

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", default="indonlp/indonlu")
    ap.add_argument("--hf_config", default="smsa")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--text_field", default=None)
    ap.add_argument("--label_field", default="label")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.hf_dataset, args.hf_config, trust_remote_code=True)

    first = list(ds.keys())[0]
    print("split:", list(ds.keys()))
    print("kolom:", ds[first].column_names)
    print("contoh:", ds[first][0])

    text_col = pick_text_col(ds[first].column_names, args.text_field)
    print(f"kolom teks: {text_col} | kolom label: {args.label_field}")

    for split in ds.keys():
        df = ds[split].to_pandas()
        out = pd.DataFrame({
            "content": df[text_col].astype(str).map(to_encoder_text),
            "label": df[args.label_field],
        })
        path = os.path.join(args.out_dir, f"smsa_{split}.csv")
        out.to_csv(path, index=False)
        print(f"{split}: {len(out)} baris -> {path} | label: {sorted(out['label'].unique())}")


if __name__ == "__main__":
    main()