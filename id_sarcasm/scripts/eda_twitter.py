"""EDA Twitter dataset — pattern frequency (train only) + tokenizer fragmentation (all splits)."""
import csv
import io
import os
import re
import sys
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "real_data", "twitter")


def load_split(split):
    path = os.path.join(DATA_DIR, f"{split}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


train_rows = load_split("train")
all_rows   = train_rows + load_split("validation") + load_split("test")

train_texts  = [r["content"] for r in train_rows]
train_labels = [int(r["label"]) for r in train_rows]
all_texts    = [r["content"] for r in all_rows]

N_train = len(train_texts)
N_all   = len(all_texts)
N_sarc  = sum(train_labels)
N_non   = N_train - N_sarc

print(f"\n{'='*65}")
print(f"Twitter dataset  |  train={N_train}  val={len(load_split('validation'))}  test={len(load_split('test'))}  total={N_all}")
print(f"Train label dist |  sarcastic={N_sarc}  non-sarcastic={N_non}")
print(f"{'='*65}\n")


# --- Pattern detectors -------------------------------------------------------

EMOTICON_RE = re.compile(r"(?:<3|[:;=x]['\`\-\^]?[)(\]\[dpov3/\\|]+)", re.IGNORECASE)
EMOJI_RE    = re.compile(
    "[\U00010000-\U0010ffff\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF\U0001F680-\U0001F9FF☀-➿]+",
    flags=re.UNICODE,
)
UNICODE_ESC_RE = re.compile(r"\\u[0-9a-fA-F]{4}")
CAPS_RE        = re.compile(r"\b[A-Z]{2,}\b")
ELONG_RE       = re.compile(r"([a-zA-Z])\1{2,}")
REPPUNC_RE     = re.compile(r"[!?]{2,}|\.{3,}")
REDUP_RE       = re.compile(r"\b([a-zA-Z]{2,})2\b")
URL_RE         = re.compile(r"https?://\S+|www\.\S+")

NEGATORS = {"gak","ga","gk","kaga","nggak","ngga","enggak","engga","tdk","tak","gada","bkn","blm","jgn"}
SLANG    = {"yg","dgn","dg","utk","tp","tpi","krn","karna","kalo","klo","aja","udah","udh","dr",
            "jd","jg","sm","org","dpt","bgt","stlh","lg","dl","sprt","hrs","gmn","bgmn"}

def _words(t):
    return set(re.sub(r"[^a-zA-Z\s]", " ", t).lower().split())

patterns = {
    "ALL_CAPS word"                    : lambda t: bool(CAPS_RE.search(re.sub(r"<[a-zA-Z_]+>", "", t))),
    "Elongasi (huruf berulang >=3x)"   : lambda t: bool(ELONG_RE.search(t)),
    "Tanda baca berulang (!!!  ???)"   : lambda t: bool(REPPUNC_RE.search(t)),
    "Emoticon teks (:) :D)"            : lambda t: bool(EMOTICON_RE.search(t)),
    "Emoji unicode"                    : lambda t: bool(EMOJI_RE.search(t)),
    "Unicode escape (\\uXXXX)"         : lambda t: bool(UNICODE_ESC_RE.search(t)),
    "Reduplikasi (kata2)"              : lambda t: bool(REDUP_RE.search(t)),
    "URL mentah (https://)"            : lambda t: bool(URL_RE.search(t)),
    "Negasi informal (gak/ga/nggak)"   : lambda t: bool(_words(t) & NEGATORS),
    "Slang/singkatan (yg/bgt/dll)"     : lambda t: bool(_words(t) & SLANG),
}


# --- SEKSI 1: Pattern Frequency (TRAIN ONLY) ---------------------------------
print(f"[TRAIN ONLY - {N_train} sampel]")
print("-- 1. PATTERN FREQUENCY --")
print(f"{'Pattern':<42} {'Count':>6}  {'%Total':>7}  {'%Sarc':>7}  {'%Non':>7}")
print("-" * 72)
for name, fn in patterns.items():
    matched  = [i for i, t in enumerate(train_texts) if fn(t)]
    cnt      = len(matched)
    pct      = cnt / N_train * 100
    s_cnt    = sum(1 for i in matched if train_labels[i] == 1)
    n_cnt    = cnt - s_cnt
    pct_s    = s_cnt / N_sarc * 100 if N_sarc else 0
    pct_n    = n_cnt / N_non  * 100 if N_non  else 0
    print(f"{name:<42} {cnt:>6}  {pct:>6.1f}%  {pct_s:>6.1f}%  {pct_n:>6.1f}%")


# --- SEKSI 2: Top Negator Tokens (TRAIN ONLY) --------------------------------
print("\n-- 2. TOP NEGATOR TOKENS (train) --")
neg_counter = Counter()
for t in train_texts:
    neg_counter.update(w for w in _words(t) if w in NEGATORS)
for tok, cnt in neg_counter.most_common(15):
    print(f"  {tok:<15}  {cnt:>5}")


# --- SEKSI 3: Top Slang Tokens (TRAIN ONLY) ----------------------------------
print("\n-- 3. TOP SLANG TOKENS (train) --")
slang_counter = Counter()
for t in train_texts:
    slang_counter.update(w for w in _words(t) if w in SLANG)
for tok, cnt in slang_counter.most_common(20):
    print(f"  {tok:<15}  {cnt:>5}")


# --- SEKSI 4: Sample Tweets per Pattern (TRAIN ONLY) ------------------------
print("\n-- 4. SAMPLE TWEETS PER PATTERN (5 per noise, train only) --")
SHOW = 5
for name, fn in patterns.items():
    examples = [(train_texts[i], train_labels[i]) for i in range(N_train) if fn(train_texts[i])][:SHOW]
    if not examples:
        print(f"\n  [{name}]  -- tidak ditemukan")
        continue
    print(f"\n  [{name}]")
    for text, lbl in examples:
        tag = "SARC" if lbl == 1 else "non "
        print(f"    [{tag}] {text[:110]}")


# --- SEKSI 5: Noise Co-occurrence vs Label (TRAIN ONLY) ----------------------
print("\n-- 5. NOISE CO-OCCURRENCE vs LABEL (train) --")
noise_fns = list(patterns.values())
for k in [1, 2, 3]:
    s_cnt = sum(1 for i, t in enumerate(train_texts) if train_labels[i] == 1 and sum(fn(t) for fn in noise_fns) >= k)
    n_cnt = sum(1 for i, t in enumerate(train_texts) if train_labels[i] == 0 and sum(fn(t) for fn in noise_fns) >= k)
    print(f"  >= {k} pattern noise  |  sarcastic: {s_cnt}/{N_sarc} = {s_cnt/N_sarc*100:.1f}%  "
          f"|  non-sarc: {n_cnt}/{N_non} = {n_cnt/N_non*100:.1f}%")


# --- SEKSI 6: Tokenizer Fragmentation (ALL SPLITS) ---------------------------
print(f"\n{'='*65}")
print(f"[ALL SPLITS - {N_all} sampel]")
print("-- 6. TOKENIZER FRAGMENTATION --")

try:
    from transformers import AutoTokenizer

    def analyze_tokenizer(model_id, label, texts):
        print(f"\n  [{label}]")
        tok = AutoTokenizer.from_pretrained(model_id)
        unk_id     = tok.unk_token_id
        total_ids  = 0
        total_unk  = 0
        over128    = 0
        word_frags = []

        for t in texts:
            ids = tok(t, truncation=False)["input_ids"]
            total_ids += len(ids)
            if unk_id is not None:
                total_unk += ids.count(unk_id)
            if len(ids) > 128:
                over128 += 1
            # fragmentasi per kata: tokenisasi tiap kata secara individual
            for word in t.split():
                n_sub = len(tok.tokenize(word))
                if n_sub > 0:
                    word_frags.append(n_sub)

        n = len(texts)
        avg_tok  = total_ids / n
        avg_frag = sum(word_frags) / len(word_frags) if word_frags else 0
        unk_rate = total_unk / total_ids * 100 if total_ids else 0
        over_pct = over128 / n * 100

        print(f"    avg token/tweet      : {avg_tok:.1f}")
        print(f"    avg subword/word     : {avg_frag:.2f}  (1.0 = ideal, >2.0 = banyak fragmentasi)")
        print(f"    UNK rate             : {unk_rate:.3f}%")
        print(f"    tweets > 128 tokens  : {over128} / {n}  ({over_pct:.1f}%)")

    analyze_tokenizer("indobenchmark/indobert-base-p1", "IndoBERT-base (uncased)", all_texts)
    analyze_tokenizer("FacebookAI/xlm-roberta-base",   "XLM-R-base (cased)",     all_texts)

except ImportError:
    print("  transformers tidak tersedia -- skip tokenizer analysis")
except Exception as e:
    print(f"  Error: {e}")

print(f"\n{'='*65}\nDone.\n")
