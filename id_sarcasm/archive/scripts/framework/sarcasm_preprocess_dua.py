"""
Two-path preprocessing untuk deteksi sarkasme Indonesia (IdSarcasm Twitter).

Satu teks mentah -> dua keluaran dengan perlakuan yang SENGAJA berlawanan:

  to_encoder_text(raw)    -> str  untuk IndoBERT.
      Noise DIPERTAHANKAN tapi di-surface jadi token eksplisit
      ([CAPS], [ELONG], [REPPUNC], [EMOTICON]); placeholder & emoji dijaga.
      Alasannya: tokenizer IndoBERT melumat sinyal permukaan (uncased,
      subword), jadi penanda harus ditandai dulu sebelum masuk.

  to_lexicon_tokens(raw)  -> list[str]  untuk lookup InSet (label incongruity).
      Noise DINORMALKAN: slang/singkatan dibakukan, negasi dikanonkan,
      elongasi diciutkan, placeholder/emoji dibuang, supaya kata cocok
      dengan entri kamus InSet.

Urutan langkah (1..6 yang sudah kita sepakati):
  1 repair_encoding   paling awal; semua langkah lain jalan di atas teks ini
  2 placeholder       <username>/<link>/<hashtag> jadi special token; runtun di-collapse
  3 emoji & emoticon  setelah repair
  4 surface_markers   jalur encoder
  5 normalisasi        jalur leksikon
  6 tidy_whitespace   kosmetik
"""
import re
import ftfy
import emoji

# Token yang HARUS didaftarkan ke tokenizer via add_special_tokens, lalu
# model.resize_token_embeddings(len(tokenizer)).
SPECIAL_TOKENS = [
    "<username>", "<link>", "<hashtag>",
    "[ELONG]", "[REPPUNC]", "[EMOTICON]",
]
PLACEHOLDERS = {"<username>", "<link>", "<hashtag>"}

# Emoticon teks. Ditangani SEBELUM demojize biar tidak bentrok dengan alias emoji.
EMOTICON_RE = re.compile(r"(?:<3|[:;=x]['`\-\^]?[)(\]\[dpov3/\\|]+)", re.IGNORECASE)

# Negator informal -> bentuk baku. Ini KRUSIAL untuk incongruity: negasi adalah
# pembalik polaritas, jadi harus dikenali sebelum skor InSet dihitung.
NEGATORS = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "kaga": "tidak",
    "nggak": "tidak", "ngga": "tidak", "enggak": "tidak", "engga": "tidak",
    "tdk": "tidak", "tak": "tidak", "gada": "tidak ada", "bkn": "bukan",
    "blm": "belum", "jgn": "jangan",
}

# Slang/singkatan -> baku (starter; perluas dengan Colloquial Indonesian Lexicon).
SLANG = {
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "tp": "tapi",
    "tpi": "tapi", "krn": "karena", "karna": "karena", "kalo": "kalau",
    "klo": "kalau", "aja": "saja", "udah": "sudah", "udh": "sudah", "dr": "dari",
    "jd": "jadi", "jg": "juga", "sm": "sama", "org": "orang", "dpt": "dapat",
    "bgt": "banget", "stlh": "setelah", "lg": "lagi", "dl": "dulu",
    "sprt": "seperti", "hrs": "harus", "gmn": "bagaimana", "bgmn": "bagaimana",
}


# --- 1. perbaikan encoding -------------------------------------------------
def _decode_escapes(s):
    """Pulihkan \\uXXXX literal (termasuk surrogate pair emoji) jadi karakter asli."""
    def repl(m):
        return (m.group(0).encode("latin1", "backslashreplace")
                .decode("unicode_escape").encode("utf-16", "surrogatepass")
                .decode("utf-16"))
    return re.sub(r"(\\u[0-9a-fA-F]{4})+", repl, s)

def repair_encoding(text):
    # ftfy menangani mojibake (ðŸ’ª) + unescape HTML entity (&amp;); _decode_escapes
    # menangani \uXXXX literal yang ftfy tidak sentuh.
    return ftfy.fix_text(_decode_escapes(str(text)))


# --- 6. kosmetik (dipakai di akhir jalur encoder) --------------------------
def tidy_whitespace(text):
    text = re.sub(r"\s+([,.])", r"\1", text)   # buang spasi sebelum koma/titik
    return re.sub(r"\s+", " ", text).strip()


# --- 2. placeholder --------------------------------------------------------
def collapse_placeholders(text):
    # Banyak tweet diawali 4-5 <username> beruntun yang cuma menggelembungkan
    # panjang tanpa sinyal -> ciutkan jadi satu. Sama untuk <hashtag>.
    text = re.sub(r"(<username>)(\s+<username>)+", r"\1", text)
    text = re.sub(r"(<hashtag>)(\s+<hashtag>)+", r"\1", text)
    return text


# --- 3 + 4. jalur ENCODER --------------------------------------------------
def _join_repeated_punct(text):
    # "?? ?" -> "???", ".. ?" -> "..?"  supaya run tanda baca terbaca utuh.
    return re.sub(r"(?<=[!?.])\s+(?=[!?.])", "", text)

def surface_markers(text):
    text = _join_repeated_punct(text)
    text = re.sub(r"[!?]{2,}", " ! [REPPUNC] ", text)   # !!! ??? ?!
    text = re.sub(r"\.{3,}", " . [REPPUNC] ", text)      # ....
    out = []
    for tok in text.split():
        if tok in SPECIAL_TOKENS or tok in PLACEHOLDERS or tok.startswith(":"):
            out.append(tok)                              # jangan ganggu marker/placeholder/alias emoji
            continue
        flags = []
        if re.search(r"(.)\1{2,}", tok):                 # elongasi: huruf sama >=3x
            tok = re.sub(r"(.)\1{2,}", r"\1", tok)
            flags.append("[ELONG]")
        # [CAPS] dihapus: presence ALL CAPS non-diskriminatif (lift 1.03 di train),
        # mayoritas jatuh di entitas/acronym (DKI, DPR, MK), bukan kata bermuatan.
        # IndoBERT uncased -> casing toh hilang di tokenizer, jadi tidak ada info yang ikut hilang.
        out.append(tok)
        out.extend(flags)                                # flag muncul tepat setelah kata yang ditandai
    return " ".join(out)

def to_encoder_text(raw):
    t = repair_encoding(raw)
    t = EMOTICON_RE.sub(" [EMOTICON] ", t)               # emoticon -> marker
    t = emoji.demojize(t, delimiters=(" :", ": "))       # emoji -> alias :rolling_eyes:
    t = surface_markers(t)
    t = collapse_placeholders(t)
    return tidy_whitespace(t)


# --- 5. jalur LEKSIKON -----------------------------------------------------
def to_lexicon_tokens(raw):
    t = repair_encoding(raw).lower()
    t = re.sub(r"<[a-z]+>", " ", t)                      # buang placeholder
    t = emoji.replace_emoji(t, " ")                      # buang emoji
    t = EMOTICON_RE.sub(" ", t)                          # buang emoticon
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)               # ciutkan elongasi -> cocok InSet
    t = re.sub(r"([a-z]+)2\b", r"\1", t)                 # reduplikasi "kata2" -> "kata"
    t = re.sub(r"[^a-z\s]", " ", t)                      # sisakan huruf saja
    toks = []
    for w in t.split():
        w = NEGATORS.get(w, SLANG.get(w, w))             # kanonkan negasi & slang
        toks.extend(w.split())                           # flatten map multi-kata (gada -> tidak ada)
    return toks


def preprocess(raw):
    """Kembalikan dua jalur sekaligus."""
    return {"encoder_text": to_encoder_text(raw),
            "lexicon_tokens": to_lexicon_tokens(raw)}