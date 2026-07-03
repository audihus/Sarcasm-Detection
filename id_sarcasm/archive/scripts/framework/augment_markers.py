#!/usr/bin/env python3
"""
Augmentasi marker in-domain untuk kelas sarkas (minoritas).

Ide (lahir dari temuan kita): sinyal sarkasme di data ini ada pada MARKER
ortografis (elongasi, kapital, tanda baca berulang), bukan pada pilihan kata.
Jadi kita perbanyak contoh sarkas dengan MEMVARIASIKAN marker-nya, kata tetap.
Tiap varian lalu dilewatkan to_encoder_text supaya konsisten dengan pipeline.

Hanya kelas label==1 yang diaugmentasi. Validation & test tidak disentuh.

Catatan jujur: ini menambah keragaman EMPHASIS, bukan keragaman kata. Itu
disengaja karena sinyalnya di emphasis. Risiko yang harus dipantau: kalau positif
dibanjiri contoh ber-marker, model bisa terlalu mengaitkan marker dengan sarkas
dan precision turun (19-26% tweet non-sarkas juga ber-marker). Pantau P/R sesudahnya.

Jalankan:
  python augment_markers.py \
    --raw_train real_data/twitter/train.csv \
    --preprocessed_train preprocessed_data/twitter_ready/train.csv \
    --out preprocessed_data/twitter_ready/train_markeraug.csv \
    --n_variants 2 --seed 42
"""
import argparse
import random
import re

import pandas as pd
from sarcasm_preprocess import to_encoder_text

VOWELS = "aeiouAEIOU"
PUNCT_RUNS = ["!!!", "!!!!", "!!", "??", "???", "?!?!", "...", "?!"]
PLACEHOLDER = re.compile(r"^<[a-z]+>$")


def _eligible(tok):
    # kata alfabet >=2 huruf, bukan placeholder/marker
    return tok.isalpha() and len(tok) > 1 and not PLACEHOLDER.match(tok)


def _vary_caps(text, rng):
    toks = text.split()
    cand = [i for i, t in enumerate(toks) if _eligible(t) and not t.isupper()]
    if not cand:
        return text
    for i in rng.sample(cand, k=min(len(cand), rng.randint(1, 2))):
        toks[i] = toks[i].upper()          # -> memicu [CAPS]
    return " ".join(toks)


def _vary_elongation(text, rng):
    toks = text.split()
    cand = [i for i, t in enumerate(toks) if _eligible(t) and len(t) >= 3]
    if not cand:
        return text
    for i in rng.sample(cand, k=min(len(cand), rng.randint(1, 2))):
        t = toks[i]
        pos = [j for j, c in enumerate(t) if c in VOWELS]
        j = rng.choice(pos) if pos else len(t) - 1
        rep = t[j] * rng.randint(3, 6)     # ulang >=3 -> memicu [ELONG]
        toks[i] = t[:j] + rep + t[j + 1:]
    return " ".join(toks)


def _vary_reppunct(text, rng, force=False):
    # tambah/ganti run tanda baca emfatik -> memicu [REPPUNC]
    run = rng.choice(PUNCT_RUNS)
    text = re.sub(r"[\s]*[!?.]+\s*$", "", text)   # rapikan tanda baca akhir lama
    return f"{text} {run}"


def perturb(text, rng):
    ops = ["caps", "elong", "punct"]
    rng.shuffle(ops)
    chosen = ops[: rng.randint(1, 3)]
    out, changed = text, False
    if "caps" in chosen:
        new = _vary_caps(out, rng); changed |= new != out; out = new
    if "elong" in chosen:
        new = _vary_elongation(out, rng); changed |= new != out; out = new
    if "punct" in chosen:
        new = _vary_reppunct(out, rng); changed |= new != out; out = new
    if not changed:                       # jamin varian selalu beda
        out = _vary_reppunct(out, rng, force=True)
    return out


def augment_sarcastic(raw_texts, n_variants, rng):
    """Kembalikan list teks ber-marker (sudah to_encoder_text) hasil augmentasi."""
    aug = []
    for raw in raw_texts:
        seen = set()
        tries = 0
        made = 0
        while made < n_variants and tries < n_variants * 6:
            tries += 1
            v = perturb(str(raw), rng)
            if v in seen or v == str(raw):
                continue
            seen.add(v)
            aug.append(to_encoder_text(v))
            made += 1
    return aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_train", required=True, help="CSV teks MENTAH (real_data/.../train.csv)")
    ap.add_argument("--preprocessed_train", required=True, help="CSV train yang sudah ber-marker (twitter_ready/train.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text_column", default="content")
    ap.add_argument("--label_column", default="label")
    ap.add_argument("--n_variants", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    raw = pd.read_csv(args.raw_train)
    pre = pd.read_csv(args.preprocessed_train)

    sarc_raw = raw[raw[args.label_column] == 1][args.text_column].astype(str).tolist()
    print(f"sarkas mentah: {len(sarc_raw)} | n_variants={args.n_variants}")

    aug_texts = augment_sarcastic(sarc_raw, args.n_variants, rng)
    aug_df = pd.DataFrame({args.text_column: aug_texts, args.label_column: 1})

    base = pre[[args.text_column, args.label_column]].copy()
    out_df = pd.concat([base, aug_df], ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # acak
    out_df.to_csv(args.out, index=False)

    n_pos = int((out_df[args.label_column] == 1).sum())
    n_neg = int((out_df[args.label_column] == 0).sum())
    print(f"sebelum: pos={int((base[args.label_column]==1).sum())} neg={int((base[args.label_column]==0).sum())}")
    print(f"sesudah: pos={n_pos} neg={n_neg} (+{len(aug_texts)} augmentasi)")
    print(f"disimpan -> {args.out}")


if __name__ == "__main__":
    main()