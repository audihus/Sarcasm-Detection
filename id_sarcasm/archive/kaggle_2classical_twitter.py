# ============================================================================
# 2 Classical + 1 Transformer — Meta-LR Stacking (Twitter Indonesia Sarcastic)
#
# Ide: daripada menambah 2 transformer (latency 2×), coba tambah satu model
# klasik lagi yang berbeda inductive bias-nya sebagai 2nd classical signal.
#
# Konfigurasi yang diuji (semuanya LR(1,2) sebagai anchor + xlmr_base):
#   C1: LR(1,2) [MVSC] + LR(1,3)   + xlmr_base  → tambah trigram
#   C2: LR(1,2) [MVSC] + CNB(1,2)  + xlmr_base  → Complement Naive Bayes
#   C3: LR(1,2) [MVSC] + SVC(1,2)  + xlmr_base  → margin-based + calibration
#
# Pembanding (benchmark):
#   B0: LR(1,2) alone              = 0.7509 (MVSC)
#   B1: xlmr_base alone @0.5       = 0.7386
#   B2: LR(1,2) + xlmr_base wavg   = 0.7900 ← target yang mau dilampaui
#
# Meta-learner: LogisticRegression dengan OOF cross-val (anti-in-sample overfit)
# Protokol: bobot & threshold dari val/OOF saja. Test disentuh sekali.
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
SOTA  = 0.7692
REF_B2 = 0.7900    # LR + xlmr_base wavg (target)
np.random.seed(SEED)

# Transformer yang dipakai: xlmr_base (best single partner dari studi sebelumnya)
TRANSFORMER = "w11wo/xlm-roberta-base-twitter-indonesia-sarcastic"

# ---------------- data ----------------
from datasets import load_dataset
ds = load_dataset(DATASET)
tr_t = list(ds["train"][TEXT_COL]); va_t = list(ds["validation"][TEXT_COL]); te_t = list(ds["test"][TEXT_COL])
ytr = np.array(ds["train"]["label"]); yva = np.array(ds["validation"]["label"]); yte = np.array(ds["test"]["label"])
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} (pos {yte.sum()})")

# ---------------- metrik ----------------
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def best_threshold(y, s):
    bt, bf = 0.5, -1.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)

def report(y, s, thr):
    p = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"f1": round(f1,4), "prec": round(pr,4), "rec": round(rc,4), "acc": round(accuracy_score(y,p),4)}

def bootstrap(y, pred, n=5000):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(f1_score(y[b], pred[b], zero_division=0))
    out = np.array(out)
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5)), float((out>SOTA).mean())

# ---------------- classical models ----------------
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except: pass
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import clone
from scipy.sparse import vstack, hstack

def make_tfidf(ngram):
    return TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None,
                           ngram_range=ngram, min_df=2, sublinear_tf=True)

def fit_classical(name, vec, classifier_fn, C_grid=None):
    """Fit classical model, return (val_probs, test_probs).
    classifier_fn dipanggil tanpa argumen untuk bikin fresh instance."""
    Xtr = vec.fit_transform(tr_t); Xva = vec.transform(va_t); Xte = vec.transform(te_t)
    if C_grid is not None:
        base = classifier_fn()
        ps   = PredefinedSplit([-1]*len(ytr) + [0]*len(yva))
        gs   = GridSearchCV(base, {"C": C_grid}, scoring="f1", cv=ps, n_jobs=-1, refit=True)
        gs.fit(vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
        valm = clone(base).set_params(C=gs.best_params_["C"]).fit(Xtr, ytr)
        full = gs.best_estimator_
        print(f"  [{name}] best C={gs.best_params_['C']}")
    else:
        full = classifier_fn().fit(vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
        valm = classifier_fn().fit(Xtr, ytr)
    return valm.predict_proba(Xva)[:,1], full.predict_proba(Xte)[:,1]

print("\n[classical] Fitting models ...")
C_GRID = [0.05, 0.1, 0.3, 1, 3, 10, 30]

# LR TF-IDF(1,2) — MVSC anchor (selalu ikut dalam semua config)
vec12 = make_tfidf((1,2))
pv_lr12, pt_lr12 = fit_classical(
    "LR(1,2)", vec12,
    lambda: LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
    C_GRID)

# LR TF-IDF(1,3)
vec13 = make_tfidf((1,3))
pv_lr13, pt_lr13 = fit_classical(
    "LR(1,3)", vec13,
    lambda: LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
    C_GRID)

# Complement Naive Bayes dengan TF-IDF(1,2) — tidak perlu C grid, alpha saja
# CNB butuh non-negative input, TF-IDF sublinear_tf menghasilkan positif ✓
def fit_cnb():
    vec_cnb = make_tfidf((1,2))
    Xtr_c = vec_cnb.fit_transform(tr_t); Xva_c = vec_cnb.transform(va_t); Xte_c = vec_cnb.transform(te_t)
    full  = ComplementNB(alpha=0.1).fit(vstack([Xtr_c,Xva_c]), np.concatenate([ytr,yva]))
    valm  = ComplementNB(alpha=0.1).fit(Xtr_c, ytr)
    return valm.predict_proba(Xva_c)[:,1], full.predict_proba(Xte_c)[:,1]
pv_cnb, pt_cnb = fit_cnb()
print("  [CNB(1,2)] done")

# LinearSVC + Platt calibration — margin-based, beda inductive bias
def fit_svc():
    vec_svc = make_tfidf((1,2))
    Xtr_s = vec_svc.fit_transform(tr_t); Xva_s = vec_svc.transform(va_t); Xte_s = vec_svc.transform(te_t)
    base   = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000,
                                              random_state=SEED, C=0.1), cv=5)
    full   = clone(base).fit(vstack([Xtr_s,Xva_s]), np.concatenate([ytr,yva]))
    valm   = clone(base).fit(Xtr_s, ytr)
    return valm.predict_proba(Xva_s)[:,1], full.predict_proba(Xte_s)[:,1]
pv_svc, pt_svc = fit_svc()
print("  [SVC(1,2)] done")

# ---------------- transformer ----------------
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[transformer] {TRANSFORMER} (device={DEVICE}) ...")

def _pos(cfg):
    l2i = getattr(cfg, "label2id", None) or {}
    for k in ("1","LABEL_1"):
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

pv_tf  = transformer_probs(TRANSFORMER, va_t)
pt_tf  = transformer_probs(TRANSFORMER, te_t)

# sanity check
f_sanity = report(yte, pt_tf, 0.5)["f1"]
print(f"  sanity xlmr_base @0.5 = {f_sanity:.4f}  (paper=0.7386) "
      f"{'OK' if abs(f_sanity-0.7386)<0.01 else 'CEK!'}")

# ---------------- meta-LR stacking (OOF) ----------------
SKF = StratifiedKFold(5, shuffle=True, random_state=SEED)

def meta_stack(keys_val, keys_test, label):
    """OOF meta-LR: val probs -> OOF -> threshold; test probs -> meta predict."""
    Mv = np.column_stack(keys_val)
    Mt = np.column_stack(keys_test)
    meta = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    oof  = cross_val_predict(meta, Mv, yva, cv=SKF, method="predict_proba")[:,1]
    meta.fit(Mv, yva)
    ft   = meta.predict_proba(Mt)[:,1]
    thr, val_f1 = best_threshold(yva, oof)   # threshold dari OOF, bukan in-sample
    m    = report(yte, ft, thr)
    lo, hi, pgt = bootstrap(yte, (ft>=thr).astype(int))
    coef = [round(c,3) for c in meta.coef_[0]]
    beat = " ← BEAT B2" if m["f1"] > REF_B2 else (" ← BEAT SOTA" if m["f1"] > SOTA else "")
    print(f"  {label:45s}  val={val_f1:.4f}  test={m['f1']:.4f}  "
          f"[{lo:.3f},{hi:.3f}]  P(>SOTA)={pgt:.1%}  coef={coef}{beat}")
    return {"label": label, "val_f1": round(val_f1,4), **m,
            "thr": round(thr,4), "ci":[round(lo,4),round(hi,4)],
            "p_beat_sota": round(pgt,3), "meta_coef": coef,
            "val_test_gap": round(val_f1-m["f1"],4)}

# ============================================================================
# EKSPERIMEN
# ============================================================================
print("\n=== PEMBANDING (benchmark) ===")
# B0: LR(1,2) alone
lr_thr, _ = best_threshold(yva, pv_lr12)
lr_m = report(yte, pt_lr12, lr_thr)
print(f"  {'B0: LR(1,2) alone':45s}  test={lr_m['f1']:.4f}")
# B1: xlmr_base alone @0.5
xb_m = report(yte, pt_tf, 0.5)
print(f"  {'B1: xlmr_base alone @0.5':45s}  test={xb_m['f1']:.4f}")
# B2: LR(1,2) + xlmr_base prob-avg (0.7900 dari studi sebelumnya)
print(f"  {'B2: LR(1,2)+xlmr_base prob-avg (ref)':45s}  test=0.7900  [target]\n")

print("=== 2-CLASSICAL + 1-TRANSFORMER (meta-LR stacking, OOF threshold) ===")
results = []

results.append(meta_stack(
    [pv_lr12, pv_lr13, pv_tf], [pt_lr12, pt_lr13, pt_tf],
    "C1: LR(1,2) + LR(1,3)   + xlmr_base"))

results.append(meta_stack(
    [pv_lr12, pv_cnb, pv_tf], [pt_lr12, pt_cnb, pt_tf],
    "C2: LR(1,2) + CNB(1,2)  + xlmr_base"))

results.append(meta_stack(
    [pv_lr12, pv_svc, pv_tf], [pt_lr12, pt_svc, pt_tf],
    "C3: LR(1,2) + SVC(1,2)  + xlmr_base"))

# Bonus: semua 4 model sekaligus
results.append(meta_stack(
    [pv_lr12, pv_lr13, pv_cnb, pv_svc, pv_tf],
    [pt_lr12, pt_lr13, pt_cnb, pt_svc, pt_tf],
    "C4: LR(1,2)+LR(1,3)+CNB+SVC + xlmr_base"))

# Referensi: 2-classical saja tanpa transformer (ablasi transformer)
print("\n=== ABLASI: 2-CLASSICAL TANPA TRANSFORMER ===")
results.append(meta_stack(
    [pv_lr12, pv_lr13], [pt_lr12, pt_lr13],
    "A1: LR(1,2) + LR(1,3)   (no transformer)"))
results.append(meta_stack(
    [pv_lr12, pv_cnb], [pt_lr12, pt_cnb],
    "A2: LR(1,2) + CNB(1,2)  (no transformer)"))
results.append(meta_stack(
    [pv_lr12, pv_svc], [pt_lr12, pt_svc],
    "A3: LR(1,2) + SVC(1,2)  (no transformer)"))

# ============================================================================
# RINGKASAN
# ============================================================================
print("\n=== RINGKASAN (ranked by val_F1) ===")
ranked = sorted(results, key=lambda r: r["val_f1"], reverse=True)
print(f"  {'config':45s}  {'val':6}  {'test':6}  {'P(>SOTA)':9}  {'gap':8}")
print("  " + "-"*80)
for r in ranked:
    beat = " ← BEAT B2" if r["f1"] > REF_B2 else (" ← BEAT SOTA" if r["f1"] > SOTA else "")
    print(f"  {r['label']:45s}  {r['val_f1']:.4f}  {r['f1']:.4f}  "
          f"{r['p_beat_sota']:7.1%}  {r['val_test_gap']:+.4f}{beat}")
print(f"\n  B2 (ref): LR(1,2)+xlmr_base prob-avg  = 0.7900  (target)")
print(f"  SOTA    : XLM-R large                 = {SOTA}")

with open("2classical_results_twitter.json","w") as f:
    json.dump({"results": results, "benchmarks": {
        "lr_alone": lr_m, "xlmr_base_alone": xb_m,
        "ref_b2_prob_avg": 0.7900, "SOTA": SOTA}}, f, indent=2)
print("\nsaved -> 2classical_results_twitter.json")
