# ============================================================================
# MVSC Bootstrap CI & P(beats SOTA) — val-tuned threshold (F1=0.7509)
#
# Konteks: ditemukan bug di run_mvsc_classification.py (headline = max(f1_default,
# f1_tuned) — test-set selection). Angka yang benar adalah 0.7509 dari val-tuned
# threshold. Script ini menghitung ulang 95% CI dan P(MVSC > SOTA) secara bersih.
#
# Perbandingan:
#   MVSC : LR TF-IDF(1,2), val-tuned threshold  → F1 = 0.7509
#   SOTA : XLM-R large, threshold=0.5 (paper)   → F1 = 0.7692
#
# KAGGLE: Settings -> GPU T4 x1, Internet = ON.
#   !pip install -q transformers==4.46.3 datasets==3.1.0 scikit-learn nltk sentencepiece
# ============================================================================
import warnings
import numpy as np
warnings.filterwarnings("ignore")

DATASET  = "w11wo/twitter_indonesia_sarcastic"
TEXT_COL = "tweet"
SEED     = 42
N_BOOT   = 10000
SOTA_F1  = 0.7692
np.random.seed(SEED)

# ---------------- data ----------------
from datasets import load_dataset
ds  = load_dataset(DATASET)
tr_t = list(ds["train"][TEXT_COL])
va_t = list(ds["validation"][TEXT_COL])
te_t = list(ds["test"][TEXT_COL])
ytr  = np.array(ds["train"]["label"])
yva  = np.array(ds["validation"]["label"])
yte  = np.array(ds["test"]["label"])
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} (pos_test={yte.sum()})")

# ---------------- helpers ----------------
from sklearn.metrics import f1_score, precision_recall_fscore_support

def best_threshold(y, s):
    bt, bf = 0.5, -1.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)

def paired_bootstrap(y, pred_a, pred_b, n=N_BOOT, seed=SEED):
    """Paired bootstrap: P(F1_a > F1_b) dan 95% CI untuk F1_a."""
    rng  = np.random.default_rng(seed)
    idx  = np.arange(len(y))
    f1_a, f1_b = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        f1_a.append(f1_score(y[b], pred_a[b], zero_division=0))
        f1_b.append(f1_score(y[b], pred_b[b], zero_division=0))
    f1_a = np.array(f1_a)
    f1_b = np.array(f1_b)
    ci_lo  = float(np.percentile(f1_a, 2.5))
    ci_hi  = float(np.percentile(f1_a, 97.5))
    p_beat = float((f1_a > f1_b).mean())
    return ci_lo, ci_hi, p_beat

# ============================================================================
# 1) MVSC: LR TF-IDF(1,2), val-tuned threshold
# ============================================================================
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
from scipy.sparse import vstack

print("\n[MVSC] Fitting LR TF-IDF(1,2) ...")
vec  = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None,
                       ngram_range=(1,2), min_df=2, sublinear_tf=True)
Xtr  = vec.fit_transform(tr_t)
Xva  = vec.transform(va_t)
Xte  = vec.transform(te_t)

base = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
ps   = PredefinedSplit([-1]*len(ytr) + [0]*len(yva))
gs   = GridSearchCV(base, {"C": [0.05, 0.1, 0.3, 1, 3, 10, 30]},
                    scoring="f1", cv=ps, n_jobs=-1, refit=True)
gs.fit(vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
best_C = gs.best_params_["C"]
print(f"  best C = {best_C}")

# val-tuned threshold: cari dari train-only model (sama persis dengan kaggle scripts)
valm    = clone(base).set_params(C=best_C).fit(Xtr, ytr)
pv_lr   = valm.predict_proba(Xva)[:, 1]
thr, _  = best_threshold(yva, pv_lr)
print(f"  val-tuned threshold = {thr:.4f}")

# test predictions: gunakan full model (train+val) dengan threshold dari val
fullm   = gs.best_estimator_
pt_lr   = fullm.predict_proba(Xte)[:, 1]
pred_mvsc = (pt_lr >= thr).astype(int)

# sanity: harus 0.7509
f1_mvsc_check = f1_score(yte, pred_mvsc, zero_division=0)
print(f"  MVSC F1 @val-tuned thr={thr:.4f} = {f1_mvsc_check:.4f}  "
      f"({'OK' if abs(f1_mvsc_check-0.7509)<0.002 else 'CEK! expected ~0.7509'})")

# juga hitung @0.5 untuk referensi (angka lama)
pred_mvsc_05 = (pt_lr >= 0.5).astype(int)
f1_mvsc_05   = f1_score(yte, pred_mvsc_05, zero_division=0)
print(f"  MVSC F1 @0.5 (ref/lama) = {f1_mvsc_05:.4f}")

# ============================================================================
# 2) SOTA: XLM-R large @0.5 (paper protocol)
# ============================================================================
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[SOTA] XLM-R large (device={DEVICE}) ...")

XLMR_LARGE = "w11wo/xlm-roberta-large-twitter-indonesia-sarcastic"
cfg = AutoConfig.from_pretrained(XLMR_LARGE)
cfg._attn_implementation = "eager"   # reproduksi paper persis
tok   = AutoTokenizer.from_pretrained(XLMR_LARGE)
model = AutoModelForSequenceClassification.from_pretrained(
            XLMR_LARGE, config=cfg).to(DEVICE).eval()

# cari index kelas sarkasme
l2i = getattr(model.config, "label2id", {}) or {}
pos = next((int(v) for k, v in l2i.items() if str(k) in ("1","LABEL_1")), 1)

BATCH = 32
probs_sota = []
with torch.no_grad():
    for i in range(0, len(te_t), BATCH):
        enc = tok(te_t[i:i+BATCH], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to(DEVICE)
        probs_sota.append(torch.softmax(model(**enc).logits, dim=-1)[:,pos].cpu().numpy())
del model
if DEVICE == "cuda": torch.cuda.empty_cache()
pt_sota = np.concatenate(probs_sota)

pred_sota = (pt_sota >= 0.5).astype(int)
f1_sota   = f1_score(yte, pred_sota, zero_division=0)
print(f"  XLM-R large F1 @0.5 = {f1_sota:.4f}  "
      f"({'OK' if abs(f1_sota-SOTA_F1)<0.001 else 'CEK! expected 0.7692'})")

# ============================================================================
# 3) Paired bootstrap
# ============================================================================
print(f"\n[Bootstrap] n={N_BOOT}, seed={SEED} ...")
lo, hi, p_beat = paired_bootstrap(yte, pred_mvsc, pred_sota)

# juga hitung untuk angka lama (F1=0.7536 @0.5) sebagai perbandingan
lo_old, hi_old, p_beat_old = paired_bootstrap(yte, pred_mvsc_05, pred_sota)

# ============================================================================
# 4) Laporan
# ============================================================================
pr, rc, f1, _ = precision_recall_fscore_support(yte, pred_mvsc, average="binary", zero_division=0)

print(f"""
============================================================
HASIL RECOMPUTE BOOTSTRAP — MVSC vs SOTA (Twitter)
============================================================

MVSC (val-tuned threshold = {thr:.4f}):
  F1     = {f1_mvsc_check:.4f}
  Prec   = {pr:.4f}  |  Recall = {rc:.4f}
  95% CI = [{lo:.4f}, {hi:.4f}]
  P(F1_MVSC > F1_SOTA) = {p_beat:.1%}

MVSC lama (@0.5, ada bug test-set selection):
  F1     = {f1_mvsc_05:.4f}
  95% CI = [{lo_old:.4f}, {hi_old:.4f}]
  P(F1_MVSC > F1_SOTA) = {p_beat_old:.1%}

SOTA — XLM-R large @0.5:
  F1     = {f1_sota:.4f}

PERBANDINGAN P(beat SOTA):
  Lama (F1=0.7536, @0.5)      : {p_beat_old:.1%}
  Baru (F1=0.7509, val-tuned) : {p_beat:.1%}
  Delta                       : {p_beat - p_beat_old:+.1%}
============================================================
""")

import json
with open("mvsc_bootstrap_results_twitter.json", "w") as f:
    json.dump({
        "mvsc_val_tuned": {
            "threshold": round(thr, 4), "f1": round(f1_mvsc_check, 4),
            "ci_95": [round(lo, 4), round(hi, 4)],
            "p_beat_sota": round(p_beat, 4),
        },
        "mvsc_default_05": {
            "threshold": 0.5, "f1": round(f1_mvsc_05, 4),
            "ci_95": [round(lo_old, 4), round(hi_old, 4)],
            "p_beat_sota": round(p_beat_old, 4),
        },
        "sota_xlmr_large": {"f1": round(f1_sota, 4), "threshold": 0.5},
        "n_bootstrap": N_BOOT, "seed": SEED,
    }, f, indent=2)
print("saved -> mvsc_bootstrap_results_twitter.json")
