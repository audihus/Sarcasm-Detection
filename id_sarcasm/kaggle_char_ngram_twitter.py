# ============================================================================
# Character N-gram TF-IDF — Classical + Hybrid (Twitter Indonesia Sarcastic)
#
# Gap dari paper IdSarcasm: hanya pakai word unigram TF-IDF, tidak ada
# character-level features. Script ini menguji apakah char n-gram menangkap
# pola yang tidak bisa ditangkap word n-gram:
#   - Elongasi informal: "bangetttt", "gilsss"
#   - Afiks bahasa Indonesia: "me-", "-kan", "-lah", "ter-"
#   - Variasi ejaan: "gabisa" vs "ga bisa"
#
# Ablasi:
#   B0: LR word(1,2) alone          = referensi MVSC 0.7509
#   B1: LR char(best_range) alone   = ?
#   B2: LR word+char combined       = ? (main classical)
#   B3: LR word + tiap transformer  = referensi 0.7900 (xlmr_base)
#   B4: LR char + tiap transformer  = ?
#   B5: LR word+char + transformer  = ? (main hybrid)
#
# Char range tuning: (2,4), (3,5), (3,6) — dipilih dari val F1, bukan test.
#
# KAGGLE: Settings -> GPU T4 x1, Internet = ON.
#   !pip install -q transformers==4.46.3 datasets==3.1.0 evaluate==0.4.3 \
#                   accelerate==1.1.1 scikit-learn nltk sentencepiece scipy
# ============================================================================
import warnings, json
import numpy as np
warnings.filterwarnings("ignore")

DATASET  = "w11wo/twitter_indonesia_sarcastic"
TEXT_COL = "tweet"
SEED, MAX_LEN, BATCH = 42, 128, 32
SOTA   = 0.7692
REF_B0 = 0.7509   # LR word(1,2) alone — MVSC
REF_B3 = 0.7900   # LR word + xlmr_base prob-avg — existing best hybrid
np.random.seed(SEED)

MODELS = {
    "indobert_base":  "w11wo/indobert-base-p1-twitter-indonesia-sarcastic",
    "indobert_large": "w11wo/indobert-large-p1-twitter-indonesia-sarcastic",
    "indobert_lem":   "w11wo/indobert-base-uncased-twitter-indonesia-sarcastic",
    "mbert":          "w11wo/bert-base-multilingual-cased-twitter-indonesia-sarcastic",
    "xlmr_base":      "w11wo/xlm-roberta-base-twitter-indonesia-sarcastic",
    "xlmr_large":     "w11wo/xlm-roberta-large-twitter-indonesia-sarcastic",
}
PAPER_F1 = {
    "indobert_base": 0.7273, "indobert_large": 0.7160, "indobert_lem": 0.6462,
    "mbert": 0.6467, "xlmr_base": 0.7386, "xlmr_large": 0.7692,
}

# ---------------- data ----------------
from datasets import load_dataset
ds = load_dataset(DATASET)
tr_t = list(ds["train"][TEXT_COL])
va_t = list(ds["validation"][TEXT_COL])
te_t = list(ds["test"][TEXT_COL])
ytr  = np.array(ds["train"]["label"])
yva  = np.array(ds["validation"]["label"])
yte  = np.array(ds["test"]["label"])
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} (pos {yte.sum()})")

# ---------------- metrik ----------------
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

def best_threshold(y, s):
    bt, bf = 0.5, -1.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)

def report(y, s, thr):
    p = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"f1": round(f1,4), "prec": round(pr,4), "rec": round(rc,4),
            "acc": round(accuracy_score(y,p),4)}

def bootstrap(y, pred, n=5000):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(f1_score(y[b], pred[b], zero_division=0))
    out = np.array(out)
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5)), float((out>SOTA).mean())

def sweep_wavg(pv_a, pv_b, pt_a, pt_b):
    """Cari bobot w terbaik di val (prob-avg), evaluasi test."""
    best = (-1.0, 0.5, 0.5)
    for w in np.linspace(0, 1, 41):
        fv = w * pv_a + (1-w) * pv_b
        thr, f1 = best_threshold(yva, fv)
        if f1 > best[0]: best = (f1, w, thr)
    val_f1, w, thr = best
    ft = w * pt_a + (1-w) * pt_b
    return ft, thr, val_f1, round(w, 3)

# ---------------- classical ----------------
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except: pass
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import clone
from scipy.sparse import hstack, vstack

C_GRID      = [0.05, 0.1, 0.3, 1, 3, 10, 30]
CHAR_RANGES = [(2,4), (3,5), (3,6)]

def fit_lr(X_tr, X_va, X_te, name):
    base = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
    ps   = PredefinedSplit([-1]*len(ytr) + [0]*len(yva))
    gs   = GridSearchCV(base, {"C": C_GRID}, scoring="f1", cv=ps, n_jobs=-1, refit=True)
    gs.fit(vstack([X_tr, X_va]), np.concatenate([ytr, yva]))
    valm = clone(base).set_params(C=gs.best_params_["C"]).fit(X_tr, ytr)
    print(f"  [{name}] best C={gs.best_params_['C']}")
    return valm.predict_proba(X_va)[:,1], gs.best_estimator_.predict_proba(X_te)[:,1]

# --- B0: LR word(1,2) ---
print("\n[classical] Fitting word n-gram LR (B0) ...")
word_vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None,
                           ngram_range=(1,2), min_df=2, sublinear_tf=True)
Xtr_w = word_vec.fit_transform(tr_t)
Xva_w = word_vec.transform(va_t)
Xte_w = word_vec.transform(te_t)
pv_word, pt_word = fit_lr(Xtr_w, Xva_w, Xte_w, "LR word(1,2)")

# sanity check B0
b0_thr, b0_valf1 = best_threshold(yva, pv_word)
b0_m = report(yte, pt_word, b0_thr)
print(f"  B0 sanity: val={b0_valf1:.4f}  test={b0_m['f1']:.4f}  "
      f"(ref={REF_B0})  {'OK' if abs(b0_m['f1']-REF_B0)<0.01 else 'CEK!'}")

# --- Char range tuning: pilih best dari val F1 standalone ---
print("\n[classical] Char n-gram range tuning (val F1, bukan test) ...")
best_char_range  = None
best_char_valf1  = -1.0
best_pv_char     = None
best_pt_char     = None
best_Xtr_c = best_Xva_c = best_Xte_c = None
char_tune_results = []

for ngram in CHAR_RANGES:
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram,
                               min_df=2, sublinear_tf=True)
    Xtr_c = char_vec.fit_transform(tr_t)
    Xva_c = char_vec.transform(va_t)
    Xte_c = char_vec.transform(te_t)
    pv_c, pt_c = fit_lr(Xtr_c, Xva_c, Xte_c, f"char{ngram}")
    thr_c, vf1_c = best_threshold(yva, pv_c)
    m_c = report(yte, pt_c, thr_c)
    marker = " ← best" if vf1_c > best_char_valf1 else ""
    print(f"  char{ngram}  val={vf1_c:.4f}  test={m_c['f1']:.4f}{marker}")
    char_tune_results.append({"range": list(ngram), "val_f1": round(vf1_c,4),
                               "test_f1": m_c["f1"]})
    if vf1_c > best_char_valf1:
        best_char_range  = ngram
        best_char_valf1  = vf1_c
        best_pv_char     = pv_c
        best_pt_char     = pt_c
        best_Xtr_c, best_Xva_c, best_Xte_c = Xtr_c, Xva_c, Xte_c

print(f"\n  >> Best char range: {best_char_range}  (val={best_char_valf1:.4f})")

# --- B1: LR char(best_range) alone ---
b1_thr, _ = best_threshold(yva, best_pv_char)
b1_m = report(yte, best_pt_char, b1_thr)
print(f"  B1 char{best_char_range} alone: test={b1_m['f1']:.4f}")

# --- B2: LR word+char combined ---
print(f"\n[classical] Fitting word+char combined LR (B2) ...")
Xtr_wc = hstack([Xtr_w, best_Xtr_c])
Xva_wc = hstack([Xva_w, best_Xva_c])
Xte_wc = hstack([Xte_w, best_Xte_c])
pv_wc, pt_wc = fit_lr(Xtr_wc, Xva_wc, Xte_wc, f"LR word+char{best_char_range}")

b2_thr, b2_valf1 = best_threshold(yva, pv_wc)
b2_m = report(yte, pt_wc, b2_thr)
beat_b0 = " ← BEAT B0" if b2_m["f1"] > REF_B0 else ""
print(f"  B2 word+char: val={b2_valf1:.4f}  test={b2_m['f1']:.4f}{beat_b0}")

# ---------------- transformer ----------------
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[transformer] device={DEVICE}")

def _pos(cfg):
    l2i = getattr(cfg, "label2id", None) or {}
    for k in ("1", "LABEL_1"):
        if k in l2i: return int(l2i[k])
    return 1

@torch.no_grad()
def transformer_probs(model_name, texts):
    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "model_type","") in ("xlm-roberta","roberta"):
        cfg._attn_implementation = "eager"
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=cfg).to(DEVICE).eval()
    pos, probs = _pos(model.config), []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i+BATCH], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        probs.append(torch.softmax(model(**enc).logits, dim=-1)[:,pos].cpu().numpy())
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return np.concatenate(probs)

P_val  = {}
P_test = {}
for key, name in MODELS.items():
    print(f"  [infer] {key} ...")
    P_val[key]  = transformer_probs(name, va_t)
    P_test[key] = transformer_probs(name, te_t)

print("\n=== SANITY (transformer @0.5 vs paper) ===")
for k in MODELS:
    f = report(yte, P_test[k], 0.5)["f1"]
    print(f"  {k:18s}  F1@0.5={f:.4f}  paper={PAPER_F1[k]:.4f}  "
          f"{'OK' if abs(f-PAPER_F1[k])<0.01 else 'CEK!'}")

# ============================================================================
# HYBRID: 6 transformer × 3 LR variant
# ============================================================================
LR_VARIANTS = [
    ("word",      pv_word, pt_word),
    (f"char{best_char_range}", best_pv_char, best_pt_char),
    ("word+char", pv_wc,   pt_wc),
]

print("\n=== HYBRID: 6 TRANSFORMER × 3 LR VARIANT (prob-avg) ===")
print(f"  {'transformer':18s}  {'lr_var':12s}  {'val':6}  {'test':6}  "
      f"{'CI':15}  {'P(>SOTA)':9}  {'gap':7}  {'w':5}")
print("  " + "-"*95)

hybrid_results = []
order = sorted(MODELS, key=lambda k: PAPER_F1[k], reverse=True)

for tf_key in order:
    pv_tf = P_val[tf_key]
    pt_tf = P_test[tf_key]
    for lr_label, pv_lr, pt_lr in LR_VARIANTS:
        ft, thr, val_f1, w = sweep_wavg(pv_lr, pv_tf, pt_lr, pt_tf)
        m  = report(yte, ft, thr)
        lo, hi, pgt = bootstrap(yte, (ft >= thr).astype(int))
        gap  = round(val_f1 - m["f1"], 4)
        beat = ""
        if m["f1"] > REF_B3: beat = " ← BEAT B3"
        elif m["f1"] > SOTA:  beat = " ← BEAT SOTA"
        print(f"  {tf_key:18s}  {lr_label:12s}  {val_f1:.4f}  {m['f1']:.4f}  "
              f"[{lo:.3f},{hi:.3f}]  {pgt:7.1%}  {gap:+.4f}  {w:.3f}{beat}")
        hybrid_results.append({
            "transformer": tf_key, "lr_variant": lr_label,
            "val_f1": round(val_f1,4), "test_f1": m["f1"],
            "prec": m["prec"], "rec": m["rec"], "acc": m["acc"],
            "ci": [round(lo,4), round(hi,4)], "p_beat_sota": round(pgt,3),
            "val_test_gap": gap, "w": w, "thr": round(thr,4),
        })
    print()

# ============================================================================
# RINGKASAN
# ============================================================================
print("\n=== RINGKASAN CLASSICAL ===")
print(f"  B0 LR word(1,2)           test={b0_m['f1']:.4f}  (ref MVSC)")
print(f"  B1 LR char{best_char_range}       test={b1_m['f1']:.4f}")
beat_b2 = " ← BEAT B0" if b2_m["f1"] > REF_B0 else ""
print(f"  B2 LR word+char           test={b2_m['f1']:.4f}{beat_b2}")

print(f"\n=== RINGKASAN HYBRID (best per lr_variant, by val_F1) ===")
print(f"  {'lr_variant':12s}  {'best_transformer':18s}  {'val':6}  {'test':6}  {'P(>SOTA)':9}  {'gap':7}")
print("  " + "-"*75)
for lr_label, _, _ in LR_VARIANTS:
    rows = [r for r in hybrid_results if r["lr_variant"] == lr_label]
    best = max(rows, key=lambda r: r["val_f1"])
    beat = ""
    if best["test_f1"] > REF_B3: beat = " ← BEAT B3"
    elif best["test_f1"] > SOTA:  beat = " ← BEAT SOTA"
    print(f"  {lr_label:12s}  {best['transformer']:18s}  {best['val_f1']:.4f}  "
          f"{best['test_f1']:.4f}  {best['p_beat_sota']:7.1%}  {best['val_test_gap']:+.4f}{beat}")

print(f"\n  Ref B3 (word+xlmr_base prob-avg) = {REF_B3}  |  SOTA = {SOTA}")

# ============================================================================
# SIMPAN
# ============================================================================
output = {
    "char_range_tuning": char_tune_results,
    "best_char_range": list(best_char_range),
    "classical": {
        "B0_word": {**b0_m, "val_f1": round(b0_valf1,4), "thr": round(b0_thr,4)},
        "B1_char": {**b1_m, "val_f1": round(best_char_valf1,4), "thr": round(b1_thr,4)},
        "B2_wordchar": {**b2_m, "val_f1": round(b2_valf1,4), "thr": round(b2_thr,4)},
    },
    "hybrid": hybrid_results,
    "references": {"MVSC_word_alone": REF_B0, "best_hybrid_word": REF_B3, "SOTA": SOTA},
}
with open("char_ngram_results_twitter.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nsaved -> char_ngram_results_twitter.json")
