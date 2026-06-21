#!/usr/bin/env python3
"""
Augmentasi kelas sarkas (minoritas) dengan sampel sarkastik Reddit.

Berbeda dari augment_markers.py (yang memvariasikan marker dari teks yang sama),
script ini menambah TEKS SARKASTIK ASLI baru dari dataset Reddit Indonesia
(real_data/reddit/) — info leksikal benar-benar baru. Tiap teks Reddit
dilewatkan to_encoder_text supaya konsisten dengan pipeline v1.

Hanya teks sarkas (label==1) yang ditambahkan ke TRAIN Twitter. Validation &
test tidak disentuh. Default menambah cukup sampel untuk menyeimbangkan 1:1.

Catatan jujur: Reddit beda domain (median ~9 kata vs Twitter ~16, gaya berbeda,
nyaris tanpa placeholder <username>/<link>). Ini menambah keragaman leksikal tapi
membawa risiko domain shift. Pantau apakah recall naik tanpa precision jatuh.

Jalankan:
  python do_augment_reddit.py \
    --reddit_train ../real_data/reddit/train.csv \
    --base_train ../preprocessed_data/twitter_ready_v1/train.csv \
    --out ../preprocessed_data/twitter_ready_v1/train_redditaug.csv \
    --n_add -1 --seed 42
"""
import argparse
import os
import random
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework"))
from sarcasm_preprocess import to_encoder_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reddit_train", required=True, help="CSV Reddit MENTAH (real_data/reddit/train.csv)")
    ap.add_argument("--base_train", required=True, help="CSV train v1 yang sudah ber-marker")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text_column", default="content")
    ap.add_argument("--label_column", default="label")
    ap.add_argument("--n_add", type=int, default=-1, help="-1 = seimbangkan 1:1 (n_neg - n_pos)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    reddit = pd.read_csv(args.reddit_train)
    base = pd.read_csv(args.base_train)[[args.text_column, args.label_column]].copy()

    sarc = reddit[reddit[args.label_column] == 1][args.text_column].astype(str).tolist()
    print(f"reddit sarkastik tersedia: {len(sarc)}")

    n_pos = int((base[args.label_column] == 1).sum())
    n_neg = int((base[args.label_column] == 0).sum())
    n_add = (n_neg - n_pos) if args.n_add < 0 else args.n_add
    n_add = min(n_add, len(sarc))

    picked = rng.sample(sarc, n_add)
    aug = pd.DataFrame({
        args.text_column: [to_encoder_text(t) for t in picked],
        args.label_column: 1,
    })

    out_df = pd.concat([base, aug], ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # acak
    out_df.to_csv(args.out, index=False)

    print(f"sebelum: pos={n_pos} neg={n_neg}")
    print(f"sesudah: pos={int((out_df[args.label_column]==1).sum())} "
          f"neg={int((out_df[args.label_column]==0).sum())} (+{n_add} reddit)")
    print(f"disimpan -> {args.out}")


if __name__ == "__main__":
    main()
