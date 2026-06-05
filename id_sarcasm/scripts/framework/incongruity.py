"""
Penurunan label incongruity (auxiliary) dari InSet berbobot.

Definisi clash (sudah dikunci):
  clash = 1  jika dalam satu tweet ADA kata/frasa positif kuat DAN negatif kuat
             (|skor| >= STRONG), setelah negasi diperhitungkan.
  clash = 0  selain itu.

Kenapa bukan rata-rata sentimen: "bagus" (+4) dan "rusak" (-4) kalau dirata-rata
jadi 0, padahal justru ko-eksistensinya itu sinyal clash-nya. Maka bukti positif
dan negatif dipisah, clash terjadi kalau dua-duanya hadir.

Input token diharapkan dari sarcasm_preprocess.to_lexicon_tokens (sudah lowercase,
slang/negasi dibakukan, elongasi diciutkan), supaya cocok dengan entri InSet.
"""
import csv

# Negator dalam bentuk baku (preprocessing sudah memetakan gk/ga/... -> tidak).
NEG_SET = {"tidak", "bukan", "belum", "jangan", "tak"}
# Konjungsi kontras menutup jangkauan negasi.
CONTRAST_SET = {"tapi", "tetapi", "namun", "padahal", "walau", "walaupun",
                "meski", "meskipun"}

# --- knob yang bisa di-tuning (sweep sambil lihat clash rate + phi) --------
STRONG = 2.5          # |skor| minimum agar dihitung "kuat"; coba 2 / 3 / 4
WINDOW = 5          # jarak maks antara unit positif & negatif agar dihitung clash;
                    # None = nonaktif (positif & negatif di mana pun); coba 3 / 5 / None
NEG_WINDOW = 3      # negator menjangkau berapa token ke depan


def _read_inset_file(path, inset, sign):
    """Baca satu file 'word<TAB>weight'. sign=+1 untuk file positif, -1 untuk negatif.
    Tanda diambil dari max(abs) lalu dipasang sesuai sign, jadi aman baik file negatif
    menyimpan '-2' maupun '2'. Frasa multi-kata didukung."""
    max_len = 1
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            word = row[0].strip().lower()
            try:
                weight = sign * abs(float(row[1]))   # baris header 'weight' gagal -> otomatis dilewati
            except ValueError:
                continue
            if word:
                inset[word] = weight
                max_len = max(max_len, len(word.split()))
    return max_len


def load_inset(pos_path, neg_path):
    """Gabung file positif dan negatif InSet jadi satu dict word -> skor bertanda."""
    inset = {}
    m1 = _read_inset_file(pos_path, inset, +1)
    m2 = _read_inset_file(neg_path, inset, -1)
    return inset, max(m1, m2)


def _score_units(tokens, inset, max_len):
    """Telusuri token kiri-ke-kanan, ambil (posisi, skor) tiap unit sentimen
    (longest-match frasa lebih dulu), balik polaritas yang kena jangkauan negasi."""
    units, i, neg, n = [], 0, 0, len(tokens)
    while i < n:
        w = tokens[i]
        if w in CONTRAST_SET:                   # kontras -> tutup scope negasi
            neg, i = 0, i + 1
            continue
        if w in NEG_SET:                        # buka scope negasi
            neg, i = NEG_WINDOW, i + 1
            continue
        matched = False
        for L in range(min(max_len, n - i), 0, -1):   # coba frasa terpanjang dulu
            phrase = " ".join(tokens[i:i + L])
            if phrase in inset:
                s = inset[phrase]
                if neg > 0 and s != 0:
                    s = -s                      # negasi membalik polaritas
                units.append((i, s))
                neg = max(0, neg - L)
                i += L
                matched = True
                break
        if not matched:
            neg = max(0, neg - 1)               # token netral mempersempit scope
            i += 1
    return units


def incongruity_label(tokens, inset, max_len, strong=STRONG, window=WINDOW):
    """Kembalikan (clash 0/1, strength). clash=1 kalau ada unit positif kuat DAN
    negatif kuat (|skor|>=strong); jika window di-set, keduanya harus berjarak
    <= window token (incongruity itu lokal). strength = sisi yang lebih lemah,
    0 kalau tidak clash. Target utama biner (InSet berisik, biner lebih kokoh)."""
    units = _score_units(tokens, inset, max_len)
    pos = [(i, s) for i, s in units if s >= strong]
    neg = [(i, s) for i, s in units if s <= -strong]
    if not (pos and neg):
        return 0, 0
    if window is None:                          # mode global: cukup dua-duanya hadir
        clash = 1
    else:                                       # mode lokal: harus ada pasangan berdekatan
        clash = 1 if any(abs(ip - jn) <= window for ip, _ in pos for jn, _ in neg) else 0
    strength = min(max(s for _, s in pos), -min(s for _, s in neg)) if clash else 0
    return clash, strength


if __name__ == "__main__":
    # Jalankan lokal:  python incongruity.py positive.tsv negative.tsv train.csv
    import sys
    import pandas as pd
    from sarcasm_preprocess import to_lexicon_tokens

    pos_path = sys.argv[1] if len(sys.argv) > 1 else "positive.tsv"
    neg_path = sys.argv[2] if len(sys.argv) > 2 else "negative.tsv"
    data_path = sys.argv[3] if len(sys.argv) > 3 else "train.csv"

    inset, max_len = load_inset(pos_path, neg_path)
    print(f"InSet: {len(inset)} entri, frasa terpanjang {max_len} kata")
    print(f"config: STRONG={STRONG}, WINDOW={WINDOW}, NEG_WINDOW={NEG_WINDOW}")

    df = pd.read_csv(data_path)
    toks = df["content"].astype(str).map(to_lexicon_tokens)
    out = toks.map(lambda t: incongruity_label(t, inset, max_len))
    df["incongruity"] = [c for c, _ in out]
    df["clash_strength"] = [s for _, s in out]

    # === DIAGNOSTIK (inti nomor 1) ===
    print(f"\nclash rate keseluruhan : {df['incongruity'].mean():.1%}")
    print("clash rate per label   :")
    print(df.groupby("label")["incongruity"].mean().to_string())

    a, b = df["incongruity"].astype(float), df["label"].astype(float)
    phi = ((a * b).mean() - a.mean() * b.mean()) / ((a.var() * b.var()) ** 0.5 + 1e-12)
    print(f"\nkorelasi phi (clash vs sarkas): {phi:.3f}")
    print("tabel silang label x clash:")
    print(pd.crosstab(df["label"], df["incongruity"]).to_string())

    df.to_csv("train_with_incongruity.csv", index=False)
    print("\ndisimpan: train_with_incongruity.csv")