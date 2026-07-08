# ============================================================================
# CONTROL EXPERIMENT — Tuning-Asymmetry Check (Twitter)
#
# Pertanyaan: apakah bobot fusion (w*) tetap condong ke LR kalau LR-nya SAMA
# SEKALI TIDAK dituning (no C search, no threshold tuning, TF-IDF default)?
#
# Motivasi: review Devil's Advocate menemukan bahwa klaim paper "LR = jangkar
# struktural yang dominan" (w* >= 0.625 di keenam pairing transformer,
# Section IV-C) bisa jadi cuma efek dari LR-nya dituning ekstensif secara
# spesifik untuk korpus ini (grid search C, TF-IDF bigram+sublinear_tf+
# min_df, threshold tuning di val) sedangkan keenam transformer-nya adalah
# checkpoint beku dari paper lain, TANPA tuning apapun di data ini. Kalau
# betul cuma soal tuning-asymmetry, maka LR versi "polos" (default sklearn,
# tanpa tuning apapun) seharusnya TIDAK lagi mendominasi bobot fusion.
#
# Skrip ini BERDIRI SENDIRI — tidak mengedit kaggle_stacking_full_twitter.py.
# Fusion-weight search (wavg) dan helper lain sengaja disalin persis dari
# skrip itu supaya kedua eksperimen apples-to-apples.
#
# KAGGLE: Settings -> GPU T4 x1, Internet = ON. Pin versi spt baseline:
#   !pip install -q transformers==4.46.3 datasets==3.1.0 evaluate==0.4.3 \
#                   accelerate==1.1.1 scikit-learn nltk sentencepiece
# ============================================================================
import warnings, json
import numpy as np
warnings.filterwarnings("ignore")

DATASET  = "w11wo/twitter_indonesia_sarcastic"
TEXT_COL = "tweet"
SEED, MAX_LEN, BATCH = 41, 128, 32
SOTA = 0.7692
np.random.seed(SEED)

# 6 model fine-tuned penulis paper acuan (idsarcasm) + F1 paper (sanity check).
MODELS = {
    "indobert_base":  "w11wo/indobert-base-p1-twitter-indonesia-sarcastic",
    "indobert_large": "w11wo/indobert-large-p1-twitter-indonesia-sarcastic",
    "indobert_lem":   "w11wo/indobert-base-uncased-twitter-indonesia-sarcastic",
    "mbert":          "w11wo/bert-base-multilingual-cased-twitter-indonesia-sarcastic",
    "xlmr_base":      "w11wo/xlm-roberta-base-twitter-indonesia-sarcastic",
    "xlmr_large":     "w11wo/xlm-roberta-large-twitter-indonesia-sarcastic",
}
PAPER_F1 = {"indobert_base": 0.7273, "indobert_large": 0.7160, "indobert_lem": 0.6462,
            "mbert": 0.6467, "xlmr_base": 0.7386, "xlmr_large": 0.7692}

# Referensi dari paper (Section IV-C / Fig. 3): w* Optimized LR per transformer.
# Cuma dua titik yang disebutkan eksplisit di teks paper; sisanya cuma
# diketahui memenuhi w* >= 0.625 (par 223). Diisi None supaya TIDAK mengarang
# angka yang tidak tersedia secara lokal (nilai lengkapnya ada di
# weight_landscape_twitter.json hasil run Kaggle sebelumnya, tidak ter-sync
# ke repo ini) -- isi manual dari Fig. 3 kalau mau tabel referensi lengkap.
OPTIMIZED_LR_WSTAR = {
    "indobert_base":  None,
    "indobert_large": None,
    "indobert_lem":   0.975,   # weakest standalone model -> near-total reliance on LR
    "mbert":          None,
    "xlmr_base":      None,
    "xlmr_large":     0.650,   # strongest standalone model -> LR masih dapat mayoritas bobot
}
OPTIMIZED_LR_FUSED_F1 = {   # dari Table II paper (Late Fusion Performance Across Transformer Families)
    "mbert": 0.7543, "indobert_lem": 0.7455, "indobert_base": 0.7518,
    "indobert_large": 0.7639, "xlmr_large": 0.7751, "xlmr_base": 0.7900,
}

# ---------------- data ----------------
from datasets import load_dataset
ds = load_dataset(DATASET)
tr_t, va_t, te_t = list(ds["train"][TEXT_COL]), list(ds["validation"][TEXT_COL]), list(ds["test"][TEXT_COL])
ytr = np.array(ds["train"]["label"]); yva = np.array(ds["validation"]["label"]); yte = np.array(ds["test"]["label"])
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} (pos {yte.sum()})")

# ---------------- helpers (identik dengan kaggle_stacking_full_twitter.py) ----------------
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression

def best_threshold(y, s):
    bt, bf = 0.5, -1.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return float(bt), float(bf)

def report(y, s, thr):
    p = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"f1": round(f1, 4), "prec": round(pr, 4), "rec": round(rc, 4), "acc": round(accuracy_score(y, p), 4)}

# ---------------- 1) CONTROL LR — TANPA TUNING SAMA SEKALI ----------------
# Beda dari get_lr_probs() di kaggle_stacking_full_twitter.py:
#   - TF-IDF ngram_range=(1,1)   [bukan (1,2)]
#   - sublinear_tf=False         [default, bukan True]
#   - min_df=1                   [default, bukan 2 -- tidak ada pruning]
#   - LogisticRegression() polos [tidak ada class_weight='balanced']
#   - C = 1.0 default            [tidak ada GridSearchCV]
#   - threshold = 0.5 tetap      [tidak ada threshold tuning di val]
#   - fit HANYA di train         [val tidak disentuh sama sekali untuk LR ini]
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception: pass
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def get_control_lr_probs():
    vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)   # semua default sklearn
    Xtr, Xva, Xte = vec.fit_transform(tr_t), vec.transform(va_t), vec.transform(te_t)
    clf = LogisticRegression(max_iter=3000, random_state=SEED)          # C=1.0 default, tanpa class_weight
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]

P_val, P_test = {}, {}
P_val["lr_control"], P_test["lr_control"] = get_control_lr_probs()

LR_CONTROL_THR = 0.5   # FIXED -- tidak dituning
m_lr_control = report(yte, P_test["lr_control"], LR_CONTROL_THR)
print("\n=== CONTROL LR (untuned) — standalone ===")
print(f"  F1={m_lr_control['f1']:.4f}  P={m_lr_control['prec']:.4f}  "
      f"R={m_lr_control['rec']:.4f}  Acc={m_lr_control['acc']:.4f}  (threshold fixed @0.5)")
print("  Sanity check: F1 ini SEHARUSNYA jelas di bawah Optimized LR (0.7509) dan")
print("  bahkan di bawah LR reproduksi paper asli (0.7171) -- kalau tidak, LR ini")
print("  belum benar-benar 'tanpa tuning' dan eksperimen kontrolnya tidak valid.")

# ---------------- 2) transformers (inference only, sama seperti stacking_full) ----------------
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("\ndevice:", DEVICE)

def _pos(cfg):
    # identik dgn kaggle_stacking_full_twitter.py: index kelas sarkas dari
    # label2id model, bukan menebak.
    l2i = getattr(cfg, "label2id", None) or {}
    for k in ("1", "LABEL_1"):
        if k in l2i:
            return int(l2i[k])
    for kk, v in l2i.items():
        if "sarc" in str(kk).lower():
            return int(v)
    return 1

@torch.no_grad()
def transformer_probs(model_name, texts):
    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "model_type", "") in ("xlm-roberta", "roberta"):
        cfg._attn_implementation = "eager"          # konsisten dgn skrip lain (XLM-R sensitif thd SDPA)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(DEVICE).eval()
    pos, probs = _pos(model.config), []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        probs.append(torch.softmax(model(**enc).logits, dim=-1)[:, pos].cpu().numpy())
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return np.concatenate(probs)

order = sorted(MODELS, key=lambda k: PAPER_F1[k], reverse=True)   # kuat -> lemah
for k in order:
    print(f"[infer] {k} ...")
    name = MODELS[k]
    P_val[k], P_test[k] = transformer_probs(name, va_t), transformer_probs(name, te_t)

print("\n  [Transformer — sanity check vs PAPER_F1 @0.5]")
print(f"  {'model':15s}  {'F1':>7}  {'F1_paper':>8}  {'check':>5}")
for k in order:
    m = report(yte, P_test[k], 0.5)
    chk = 'OK' if abs(m['f1'] - PAPER_F1[k]) < 0.01 else 'CEK!'
    print(f"  {k:15s}  {m['f1']:7.4f}  {PAPER_F1[k]:8.4f}  {chk}")

# ---------------- 3) fusion-weight search: CONTROL LR + tiap transformer ----------------
def wavg(a_key, b_key):
    """Identik dgn wavg() di kaggle_stacking_full_twitter.py: sweep w di val,
    pilih threshold F1-optimal di val, evaluasi di test."""
    best = (-1.0, 0.5, 0.5)
    for w in np.linspace(0, 1, 41):
        thr, f1 = best_threshold(yva, w * P_val[a_key] + (1 - w) * P_val[b_key])
        if f1 > best[0]: best = (f1, w, thr)
    val_f1, w, thr = best
    fused_test = w * P_test[a_key] + (1 - w) * P_test[b_key]
    return fused_test, thr, val_f1, w

print("\n=== FUSION WEIGHT SEARCH: control (untuned) LR + tiap transformer ===")
print(f"{'transformer':15s} {'w*_control':>11s} {'w*_optimized':>13s} {'fused_F1_ctrl':>14s} {'fused_F1_opt':>13s}")
print("  " + "-"*80)
results = {}
for k in order:
    ft, thr, val_f1, w_star = wavg("lr_control", k)
    m = report(yte, ft, thr)
    ref_w = OPTIMIZED_LR_WSTAR.get(k)
    ref_w_str = f"{ref_w:.3f}" if ref_w is not None else "n/a (see Fig.3)"
    ref_f1 = OPTIMIZED_LR_FUSED_F1.get(k)
    ref_f1_str = f"{ref_f1:.4f}" if ref_f1 is not None else "n/a"
    print(f"{k:15s} {w_star:11.3f} {ref_w_str:>13s} {m['f1']:14.4f} {ref_f1_str:>13s}")
    results[k] = {
        "w_star_control": round(w_star, 3),
        "w_star_optimized_lr_reference": ref_w,
        "threshold": round(thr, 4),
        "val_f1_at_wstar": round(val_f1, 4),
        "test_f1_control_fusion": m["f1"],
        "test_precision_control_fusion": m["prec"],
        "test_recall_control_fusion": m["rec"],
        "test_acc_control_fusion": m["acc"],
        "test_f1_optimized_lr_fusion_reference": ref_f1,
    }

print("\nInterpretasi:")
print("  - Kalau w*_control TETAP tinggi (mendekati/mirip w*_optimized) -> temuan paper")
print("    'LR = jangkar dominan' BERTAHAN meski LR tidak dituning -> confound tuning-")
print("    asymmetry TIDAK terbukti jadi penjelasan utama, klaim arsitektural bisa")
print("    dipertahankan (mungkin perlu dilunakkan sedikit, tapi intinya selamat).")
print("  - Kalau w*_control jauh LEBIH RENDAH dari w*_optimized (bergeser ke arah")
print("    transformer) -> confound tuning-asymmetry TERBUKTI: dominasi bobot LR di")
print("    paper sebagian besar adalah efek LR-nya dituning, bukan superioritas")
print("    arsitektural classical vs attention -> klaim Section IV-C & Conclusion")
print("    perlu direscope.")

out = {
    "control_lr_standalone": m_lr_control,
    "control_lr_threshold_fixed": LR_CONTROL_THR,
    "fusion_results": results,
}
with open("tuning_asymmetry_control_results_twitter.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved -> tuning_asymmetry_control_results_twitter.json")
