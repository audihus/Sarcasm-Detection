# ============================================================================
# SYSTEMATIC Fusion Check — LR + SETIAP transformer paper (Twitter)
# Fusi LR dengan tiap transformer SATU PER SATU (weighted-avg & meta-stack),
# pilih metode terbaik BY VALIDATION, lapor F1 / 95% CI / P(>SOTA). Plus baris
# pembanding multi-model. 6 transformer paper, inferensi-only (cepat).
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
    # identik dgn run_classification.py & kaggle_hybrid_fusion_twitter.py:
    # index kelas sarkas diambil dari label2id model, bukan menebak.
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

# ---------------- 4) SYSTEMATIC: LR + tiap transformer (satu per satu) ----------------
# Untuk SETIAP transformer: fusi dengan LR via weighted-avg DAN meta-stack,
# lalu pilih metode terbaik BY VALIDATION (jujur). Jawab: transformer mana yang
# paling untung kalau digabung LR, dan mana yang setelah +LR melewati SOTA.
order = sorted(MODELS, key=lambda k: PAPER_F1[k], reverse=True)   # kuat -> lemah
print("\n=== SYSTEMATIC: LR + each transformer ===")
print(f"{'transformer':15s} {'alone':>7s} {'+LR_wavg':>8s} {'+LR_stk':>8s} {'best':>7s} "
      f"{'dAlone':>7s} {'dSOTA':>7s} {'P(>SOTA)':>9s} {'95%CI(best)':>15s}")
results = {}
for k in order:
    am = report(yte, P_test[k], 0.5)                  # alone (argmax) -> full metrics
    ftw, thw, vw = wavg(["lr", k]); wm = report(yte, ftw, thw)
    fts, ths, vs = stack(["lr", k]); sm = report(yte, fts, ths)
    if vw >= vs:                                      # pilih metode by VALIDATION, bukan test
        method, ft, thr, bm = "wavg", ftw, thw, wm
    else:
        method, ft, thr, bm = "stack", fts, ths, sm
    lo, hi, pgt = bootstrap(yte, (ft >= thr).astype(int))
    beat = "BEAT" if bm["f1"] > SOTA else ""
    results[k] = {"alone_f1": am["f1"], "lr_wavg_f1": wm["f1"], "lr_stack_f1": sm["f1"],
                  "best_method": method,
                  "f1": bm["f1"], "precision": bm["prec"], "recall": bm["rec"], "accuracy": bm["acc"],
                  "d_vs_alone": round(bm["f1"] - am["f1"], 4), "d_vs_sota": round(bm["f1"] - SOTA, 4),
                  "ci": [round(lo, 4), round(hi, 4)], "p_beat_sota": round(pgt, 3)}
    print(f"{k:15s} {am['f1']:7.4f} {wm['f1']:8.4f} {sm['f1']:8.4f} {bm['f1']:7.4f} "
          f"{bm['f1']-am['f1']:+7.4f} {bm['f1']-SOTA:+7.4f} {pgt:8.1%}  [{lo:.3f},{hi:.3f}] {beat}")

# ---------------- pembanding: LR sendiri + kombinasi 3-model terbaik ----------------
print("\n=== context (pembanding) ===")
lr_thr, _ = best_threshold(yva, P_val["lr"])
lrm = report(yte, P_test["lr"], lr_thr)
print(f"  {'LR alone (MVSC)':20s} F1={lrm['f1']:.4f} P={lrm['prec']:.4f} R={lrm['rec']:.4f} Acc={lrm['acc']:.4f}")
ft3, th3, v3 = stack(["lr", "indobert_base", "xlmr_large"])
m3 = report(yte, ft3, th3); lo3, hi3, p3 = bootstrap(yte, (ft3 >= th3).astype(int))
print(f"  {'LR+IB+XLMR (3-model)':20s} F1={m3['f1']:.4f} P={m3['prec']:.4f} R={m3['rec']:.4f} Acc={m3['acc']:.4f}  P(>SOTA)={p3:.1%}")
results["_context"] = {
    "lr_alone": {"f1": lrm["f1"], "precision": lrm["prec"], "recall": lrm["rec"], "accuracy": lrm["acc"]},
    "lr_ib_xlmr_3model": {"f1": m3["f1"], "precision": m3["prec"], "recall": m3["rec"], "accuracy": m3["acc"],
                          "p_beat_sota": round(p3, 3), "ci": [round(lo3, 4), round(hi3, 4)]}}

print(f"\nSOTA = {SOTA}. Lihat kolom: 'dAlone'>0 = fusi LR menambah; 'dSOTA'>0 = lewat SOTA.")
with open("stacking_systematic_results_twitter.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved -> stacking_systematic_results_twitter.json")
