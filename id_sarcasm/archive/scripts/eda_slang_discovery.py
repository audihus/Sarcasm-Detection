"""Bottom-up slang discovery dari training data Twitter.

Tidak butuh KBBI eksternal — pakai heuristik:
  - Kata baku Indonesia punya pola vokal teratur (minimal 1 vokal per 3 huruf)
  - Singkatan/slang cenderung consonant-heavy atau pendek tanpa pola silabik
  - Slang informal sering punya pola fonetis tidak baku (gitu, gimana, ntar, dll)
"""
import csv
import io
import os
import re
import sys
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "real_data", "twitter")

# Load train only
train_rows = []
with open(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        train_rows.append(row)

train_texts  = [r["content"] for r in train_rows]
train_labels = [int(r["label"]) for r in train_rows]
N = len(train_texts)

# --- Sudah ditangani (jangan masukkan ke kandidat baru) -----------------------
KNOWN_NEGATORS = {"gak","ga","gk","kaga","nggak","ngga","enggak","engga",
                  "tdk","tak","gada","bkn","blm","jgn"}
KNOWN_SLANG    = {"yg","dgn","dg","utk","tp","tpi","krn","karna","kalo","klo",
                  "aja","udah","udh","dr","jd","jg","sm","org","dpt","bgt",
                  "stlh","lg","dl","sprt","hrs","gmn","bgmn"}
ALREADY_KNOWN  = KNOWN_NEGATORS | KNOWN_SLANG

# Stopwords Indonesia yang sangat umum (kata baku) — jangan flag sebagai slang
STOPWORDS_ID = {
    "yang","dan","di","ke","dari","ini","itu","dengan","untuk","tidak","ada",
    "saya","aku","kamu","dia","mereka","kita","kami","akan","sudah","juga",
    "atau","tapi","tetapi","jika","kalau","karena","maka","setelah","sebelum",
    "pada","dalam","atas","bawah","oleh","antara","bisa","harus","baru","lagi",
    "saja","sama","lebih","sangat","sudah","belum","jangan","serta","bahwa",
    "namun","saat","ketika","bagi","tentang","seperti","semua","para","pak",
    "ibu","ya","oh","ah","iya","tidak","adalah","dapat","punya","bilang","kata",
    "orang","sini","sana","mana","bagaimana","siapa","kapan","berapa","mengapa",
    "waktu","hari","tahun","bulan","lalu","kemarin","besok","sekarang","masih",
    "sudah","telah","ingin","mau","perlu","agar","supaya","hingga","sampai",
    "sangat","sekali","banget","paling","sudah","jadi","terus","lalu","kemudian",
    "pun","pula","selain","juga","memang","emang","malah","justru","padahal",
    "walaupun","meskipun","biarpun","walau","meski","soal","hal","cara","tempat",
    "diri","kali","kini","baik","besar","kecil","banyak","sedikit","boleh",
    "sebenarnya","seharusnya","setiap","semua","selalu","pernah","biasa","hanya",
    "cuma","saja","malah","bahkan","apalagi","ternyata","memang","emang",
}

# --- Ekstrak semua token dari training data -----------------------------------
token_counter = Counter()
token_per_label = {0: Counter(), 1: Counter()}

for text, lbl in zip(train_texts, train_labels):
    # Buang placeholder, angka, tanda baca — sisakan huruf saja
    clean = re.sub(r"<[a-zA-Z_]+>", " ", text)          # buang placeholder
    clean = re.sub(r"\\u[0-9a-fA-F]{4}", " ", clean)    # buang unicode escape
    clean = re.sub(r"[^a-zA-Z\s]", " ", clean)          # sisakan huruf
    tokens = clean.lower().split()
    tokens = [t for t in tokens if len(t) >= 2]
    token_counter.update(tokens)
    token_per_label[lbl].update(tokens)


# --- Heuristik: apakah token ini kemungkinan slang/singkatan? ----------------
VOWELS = set("aeiou")

def vowel_ratio(word):
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if c in VOWELS) / len(letters)

def is_likely_abbreviation(word):
    """Singkatan cenderung: pendek (<=4) dan consonant-heavy (vokal < 30%)."""
    return len(word) <= 4 and vowel_ratio(word) < 0.30

def is_likely_informal(word):
    """Slang informal: vokal ganda (ee, aa, oo), akhiran tidak baku, fonetis."""
    patterns = [
        r"(.)\1{1,}",           # huruf berulang (wkwk, haha, xixi)
        r"^(nge|ke|ge|nge)",    # prefiks informal
        r"(nya|in|an|kan)$",    # sufiks informal (beda dari baku -nya hanya kalau pendek)
        r"^(w|gw|gue|lo|lu)$",  # kata ganti informal
    ]
    return any(re.search(p, word) for p in patterns) and len(word) <= 6

def classify(word):
    if word in ALREADY_KNOWN:
        return "SUDAH_DIKETAHUI"
    if word in STOPWORDS_ID:
        return "STOPWORD_BAKU"
    if is_likely_abbreviation(word):
        return "KANDIDAT_SINGKATAN"
    if is_likely_informal(word):
        return "KANDIDAT_SLANG"
    if len(word) <= 5 and vowel_ratio(word) < 0.40:
        return "KANDIDAT_SINGKATAN"
    return "LAINNYA"


# --- Filter dan tampilkan ------------------------------------------------------
MIN_FREQ = 3  # abaikan token yang muncul < 3x

candidates = {
    word: cnt
    for word, cnt in token_counter.items()
    if cnt >= MIN_FREQ
    and word not in STOPWORDS_ID
    and word not in ALREADY_KNOWN
    and len(word) >= 2
}

# Hitung distribusi per label untuk setiap kandidat
def label_pct(word):
    s = token_per_label[1][word]
    n = token_per_label[0][word]
    total = s + n
    if total == 0:
        return 0, 0
    return s/total*100, n/total*100

# --- Output -------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Token discovery dari {N} training samples (min freq={MIN_FREQ})")
print(f"Total unique token (setelah filter): {len(candidates)}")
print(f"{'='*70}\n")

# BAGIAN A: Kandidat singkatan (consonant-heavy, pendek)
print("-- A. KANDIDAT SINGKATAN (pendek + consonant-heavy, belum di SLANG list) --")
print(f"{'Token':<15} {'Freq':>6}  {'%Sarc':>7}  {'%Non':>7}  {'Klasifikasi'}")
print("-" * 60)
singkatan = {w: c for w, c in candidates.items() if classify(w) in ("KANDIDAT_SINGKATAN",)}
for word, cnt in sorted(singkatan.items(), key=lambda x: -x[1])[:80]:
    ps, pn = label_pct(word)
    print(f"  {word:<13} {cnt:>6}  {ps:>6.1f}%  {pn:>6.1f}%")

# BAGIAN B: Kandidat slang informal
print("\n-- B. KANDIDAT SLANG INFORMAL (fonetis/prefiks informal) --")
print(f"{'Token':<15} {'Freq':>6}  {'%Sarc':>7}  {'%Non':>7}")
print("-" * 50)
slang_inf = {w: c for w, c in candidates.items() if classify(w) == "KANDIDAT_SLANG"}
for word, cnt in sorted(slang_inf.items(), key=lambda x: -x[1])[:60]:
    ps, pn = label_pct(word)
    print(f"  {word:<13} {cnt:>6}  {ps:>6.1f}%  {pn:>6.1f}%")

# BAGIAN C: Token frekuensi tinggi yang belum dikategorikan (mungkin ada slang)
print("\n-- C. TOP 100 TOKEN FREKUENSI TINGGI (semua kategori, belum di list) --")
print(f"{'Token':<15} {'Freq':>6}  {'%Sarc':>7}  {'%Non':>7}  {'Klasifikasi'}")
print("-" * 65)
for word, cnt in sorted(candidates.items(), key=lambda x: -x[1])[:100]:
    ps, pn = label_pct(word)
    cat = classify(word)
    print(f"  {word:<13} {cnt:>6}  {ps:>6.1f}%  {pn:>6.1f}%  {cat}")

# BAGIAN D: Ringkasan per kategori
print("\n-- D. RINGKASAN KATEGORI --")
from collections import defaultdict
by_cat = defaultdict(list)
for word, cnt in candidates.items():
    by_cat[classify(word)].append((word, cnt))
for cat, items in sorted(by_cat.items()):
    top5 = ", ".join(w for w, _ in sorted(items, key=lambda x: -x[1])[:5])
    print(f"  {cat:<25} {len(items):>4} token  |  top: {top5}")

print(f"\n{'='*70}\nDone.\n")
