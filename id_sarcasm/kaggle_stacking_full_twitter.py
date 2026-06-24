# ============================================================================
# FULL Stacking Ablation — Classical (LR) x ALL paper transformers (Twitter)
# Untuk laporan komprehensif: semua 6 transformer paper + LR, di-stack dengan
# berbagai konfigurasi + double-stacking 2-level. Semua OOF + bootstrap CI.
#
# KAGGLE: Settings -> GPU T4 x1, Internet = ON. Pin versi spt baseline:
#   !pip install -q transformers==4.46.3 datasets==3.1.0 evaluate==0.4.3 \
#                   accelerate==1.1.1 scikit-learn nltk sentencepiece
#
# CATATAN KEJUJURAN: makin banyak model di-stack pada val kecil (268) makin
# rawan OVERFIT. Karena itu kita laporkan VAL_F1 (OOF) di samping TEST_F1 —
# jarak val-test yang lebar = sinyal overfit. Bandingkan, jangan asal pilih
# yang TEST-nya tertinggi.
# ============================================================================
import warnings, json
import numpy as np
warnings.filterwarnings("ignore")

DATASET  = "w11wo/twitter_indonesia_sarcastic"
TEXT_COL = "tweet"
SEED, MAX_LEN, BATCH = 42, 128, 32
SOTA = 0.7692
np.random.seed(SEED)

# 6 model fine-tuned penulis paper + F1 paper (untuk sanity).
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

# ---------------- data ----------------
from datasets import load_dataset
ds = load_dataset(DATASET)
tr_t, va_t, te_t = list(ds["train"][TEXT_COL]), list(ds["validation"][TEXT_COL]), list(ds["test"][TEXT_COL])
ytr = np.array(ds["train"]["label"]); yva = np.array(ds["validation"]["label"]); yte = np.array(ds["test"]["label"])
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} (pos {yte.sum()})")

# ---------------- helpers ----------------
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict, StratifiedKFold
from sklearn.base import clone
SKF = StratifiedKFold(5, shuffle=True, random_state=SEED)

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

def bootstrap(y, pred, n=5000):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(f1_score(y[b], pred[b], zero_division=0))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out > SOTA).mean())

# ---------------- 1) classical LR (MVSC) ----------------
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception: pass
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack

def get_lr_probs():
    vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr, Xva, Xte = vec.fit_transform(tr_t), vec.transform(va_t), vec.transform(te_t)
    base = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
    ps = PredefinedSplit([-1] * len(ytr) + [0] * len(yva))
    gs = GridSearchCV(base, {"C": [0.05, 0.1, 0.3, 1, 3, 10, 30]}, scoring="f1", cv=ps, n_jobs=-1, refit=True)
    gs.fit(vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    valm = clone(base).set_params(C=gs.best_params_["C"]).fit(Xtr, ytr)
    return valm.predict_proba(Xva)[:, 1], gs.best_estimator_.predict_proba(Xte)[:, 1]

# ---------------- 2) transformers (inference only) ----------------
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

def _pos(cfg):
    l2i = getattr(cfg, "label2id", None) or {}
    for k in ("1", "LABEL_1"):
        if k in l2i: return int(l2i[k])
    return 1

@torch.no_grad()
def transformer_probs(model_name, texts):
    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "model_type", "") in ("xlm-roberta", "roberta"):
        cfg._attn_implementation = "eager"          # konsisten run_classification.py (XLM-R sensitif)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(DEVICE).eval()
    pos, probs = _pos(model.config), []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        probs.append(torch.softmax(model(**enc).logits, dim=-1)[:, pos].cpu().numpy())
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return np.concatenate(probs)

P_val, P_test = {}, {}
P_val["lr"], P_test["lr"] = get_lr_probs()
for key, name in MODELS.items():
    print(f"[infer] {key} ...")
    P_val[key], P_test[key] = transformer_probs(name, va_t), transformer_probs(name, te_t)

# ---- sanity: tiap model @0.5 vs paper ----
print("\n=== MODEL ALONE (transformer @0.5 = paper protocol) ===")
for k in MODELS:
    f = report(yte, P_test[k], 0.5)["f1"]
    print(f"  {k:15s} F1@0.5={f:.4f}  paper={PAPER_F1[k]:.4f}  {'OK' if abs(f-PAPER_F1[k])<0.01 else 'CEK!'}")
lr_thr, _ = best_threshold(yva, P_val["lr"])
print(f"  {'lr (MVSC)':15s} F1={report(yte, P_test['lr'], lr_thr)['f1']:.4f} @tuned")

# ---------------- 3) fusion engines ----------------
def wavg(keys):
    a, b = keys; best = (-1.0, 0.5, 0.5)
    for w in np.linspace(0, 1, 41):
        thr, f1 = best_threshold(yva, w * P_val[a] + (1 - w) * P_val[b])
        if f1 > best[0]: best = (f1, w, thr)
    val_f1, w, thr = best
    return w * P_test[a] + (1 - w) * P_test[b], thr, val_f1

def _meta_oof(keys):
    """meta-LR over keys: kembalikan (test_probs, oof_val_probs)."""
    Mv = np.column_stack([P_val[k] for k in keys]); Mt = np.column_stack([P_test[k] for k in keys])
    meta = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    oof = cross_val_predict(meta, Mv, yva, cv=SKF, method="predict_proba")[:, 1]
    meta.fit(Mv, yva)
    return meta.predict_proba(Mt)[:, 1], oof

def stack(keys):
    ft, oof = _meta_oof(keys)
    thr, val_f1 = best_threshold(yva, oof)
    return ft, thr, val_f1

def double_stack():
    """2-level: stack IndoBERT-family & XLM-R-family dulu (level-1, OOF), lalu
    stack [LR + indo_stack + xlmr_stack] (level-2). Rawan overfit -> sbg ablation."""
    # level-1 produce OOF val preds + test preds for each family
    iv = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
                           np.column_stack([P_val[k] for k in ["indobert_base", "indobert_large", "indobert_lem"]]),
                           yva, cv=SKF, method="predict_proba")[:, 1]
    xv = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
                           np.column_stack([P_val[k] for k in ["xlmr_base", "xlmr_large"]]),
                           yva, cv=SKF, method="predict_proba")[:, 1]
    it, _ = _meta_oof(["indobert_base", "indobert_large", "indobert_lem"])
    xt, _ = _meta_oof(["xlmr_base", "xlmr_large"])
    Mv = np.column_stack([P_val["lr"], iv, xv]); Mt = np.column_stack([P_test["lr"], it, xt])
    meta2 = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    oof2 = cross_val_predict(meta2, Mv, yva, cv=SKF, method="predict_proba")[:, 1]
    thr, val_f1 = best_threshold(yva, oof2)
    meta2.fit(Mv, yva)
    return meta2.predict_proba(Mt)[:, 1], thr, val_f1

ALL_T = ["indobert_base", "indobert_large", "indobert_lem", "mbert", "xlmr_base", "xlmr_large"]
STRONG = ["indobert_base", "xlmr_base", "xlmr_large"]
EXPERIMENTS = {
    "wavg  LR+XLMRlarge":          lambda: wavg(["lr", "xlmr_large"]),
    "stack LR+IBbase+XLMRlarge":   lambda: stack(["lr", "indobert_base", "xlmr_large"]),
    "stack LR+STRONG(3)":          lambda: stack(["lr"] + STRONG),
    "stack ALL-transformers(6)":   lambda: stack(ALL_T),
    "stack LR+ALL(7)":             lambda: stack(["lr"] + ALL_T),
    "double-stack 2level":         double_stack,
}

# ---------------- 4) run + report ----------------
print("\n=== STACKING ABLATION (val OOF vs test; bandingkan gap-nya) ===")
print(f"{'config':28s} {'VAL_F1':>7s} {'TEST_F1':>8s}  {'95% CI':>17s} {'P(>SOTA)':>9s}")
results = {}
for name, fn in EXPERIMENTS.items():
    ft, thr, val_f1 = fn()
    m = report(yte, ft, thr); lo, hi, pgt = bootstrap(yte, (ft >= thr).astype(int))
    gap = val_f1 - m["f1"]
    results[name] = {**m, "val_f1": round(val_f1, 4), "thr": round(thr, 3),
                     "ci": [round(lo, 4), round(hi, 4)], "p_beat_sota": round(pgt, 3),
                     "val_test_gap": round(gap, 4)}
    flag = "  <-- overfit?" if gap > 0.04 else ""
    print(f"{name:28s} {val_f1:7.4f} {m['f1']:8.4f}  [{lo:.3f},{hi:.3f}] {pgt:8.1%}{flag}")

print(f"\nSOTA XLM-R large = {SOTA}. 'BEAT' kalau TEST_F1 > {SOTA} (cek juga CI & gap).")
print("Pilih yang TEST tinggi TAPI gap val-test kecil + CI tak terlalu lebar.")
with open("stacking_full_results_twitter.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved -> stacking_full_results_twitter.json")
