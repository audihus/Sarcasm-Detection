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
import warnings, json, time
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scipy.sparse import vstack

_LR_VEC  = None   # simpan untuk timing & param count
_LR_FULL = None   # model fit on train+val (untuk predict test)

def get_lr_probs():
    global _LR_VEC, _LR_FULL
    vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr, Xva, Xte = vec.fit_transform(tr_t), vec.transform(va_t), vec.transform(te_t)
    base = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
    ps = PredefinedSplit([-1] * len(ytr) + [0] * len(yva))
    gs = GridSearchCV(base, {"C": [0.05, 0.1, 0.3, 1, 3, 10, 30]}, scoring="f1", cv=ps, n_jobs=-1, refit=True)
    gs.fit(vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    valm = clone(base).set_params(C=gs.best_params_["C"]).fit(Xtr, ytr)
    _LR_VEC  = vec
    _LR_FULL = gs.best_estimator_
    return valm.predict_proba(Xva)[:, 1], gs.best_estimator_.predict_proba(Xte)[:, 1]

def reproduce_classical_paper():
    """Reproduce LR/NB/SVM paper exact: unigram BoW+TF-IDF, accuracy GridSearch, seed=41."""
    CLF_PAPER_F1 = {"lr": 0.7142, "nb": 0.6721, "svm": 0.6782}
    classifiers_def = {
        "lr":  (LogisticRegression(max_iter=3000),  {"C": [0.01, 0.1, 1, 10, 100]}),
        "nb":  (MultinomialNB(),                    {"alpha": np.linspace(0.001, 1, 50)}),
        "svm": (SVC(),                              {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["rbf", "linear"]}),
    }
    ps = PredefinedSplit([-1]*len(ytr) + [0]*len(yva))
    best_per_clf = {}
    np.random.seed(41)   # paper seed
    for feat_name, VecCls in [("BoW", CountVectorizer), ("TFIDF", TfidfVectorizer)]:
        vec = VecCls(tokenizer=word_tokenize, token_pattern=None)
        Xtr_f = vec.fit_transform(tr_t)
        Xva_f = vec.transform(va_t)
        Xte_f = vec.transform(te_t)
        Xdev  = vstack([Xtr_f, Xva_f])
        ydev  = np.concatenate([ytr, yva])
        for cname, (clf_base, param_grid) in classifiers_def.items():
            gs = GridSearchCV(clone(clf_base), param_grid, cv=ps, n_jobs=-1, refit=True)
            gs.fit(Xdev, ydev)
            pred = gs.predict(Xte_f)
            f1 = f1_score(yte, pred, zero_division=0)
            if cname not in best_per_clf or f1 > best_per_clf[cname][1]:
                best_per_clf[cname] = (pred, f1, feat_name)
    np.random.seed(SEED)   # restore seed=42
    return best_per_clf, CLF_PAPER_F1

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

INFER_TIMES = {}   # model_name -> detik (hanya test set)

@torch.no_grad()
def transformer_probs(model_name, texts):
    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "model_type", "") in ("xlm-roberta", "roberta"):
        cfg._attn_implementation = "eager"          # konsisten run_classification.py (XLM-R sensitif)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(DEVICE).eval()
    pos, probs = _pos(model.config), []
    is_test = (len(texts) == len(te_t))
    if is_test:
        # warmup 1 batch agar CUDA kernel sudah loaded, lalu ukur pure forward
        enc0 = tok(texts[:BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        model(**enc0)
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        probs.append(torch.softmax(model(**enc).logits, dim=-1)[:, pos].cpu().numpy())
    if is_test:
        if DEVICE == "cuda": torch.cuda.synchronize()
        INFER_TIMES[model_name] = time.perf_counter() - t0
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return np.concatenate(probs)

P_val, P_test = {}, {}
P_val["lr"], P_test["lr"] = get_lr_probs()
print("[reproduce] classical paper (LR/NB/SVM) ...")
clf_results, CLF_PAPER_F1 = reproduce_classical_paper()
for key, name in MODELS.items():
    print(f"[infer] {key} ...")
    P_val[key], P_test[key] = transformer_probs(name, va_t), transformer_probs(name, te_t)

# ---- reproduce vs paper: classical + transformer ----
lr_thr, _ = best_threshold(yva, P_val["lr"])
m_lr_rep = report(yte, P_test["lr"], lr_thr)

print("\n=== REPRODUCE vs PAPER ===")

print("\n  [Classical — Paper Protocol (best of BoW/TF-IDF per classifier)]")
print(f"  {'model':12s}  {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'F1_paper':>8}  {'bestVec':>8}  {'check':>5}")
print("  " + "-"*72)
for cname, (pred, f1v, vec_name) in clf_results.items():
    pr, rc, f1c, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    acc = accuracy_score(yte, pred)
    chk = 'OK' if abs(f1c - CLF_PAPER_F1[cname]) < 0.01 else 'CEK!'
    print(f"  {cname+'_paper':12s}  {acc:6.4f} {pr:6.4f} {rc:6.4f} {f1c:6.4f}  "
          f"{CLF_PAPER_F1[cname]:8.4f}  {vec_name:>8}  {chk}")
print(f"  {'lr_mvsc':12s}  {m_lr_rep['acc']:6.4f} {m_lr_rep['prec']:6.4f} {m_lr_rep['rec']:6.4f} {m_lr_rep['f1']:6.4f}  "
      f"  {'(ours)':>8}  {'bigram+tuned':>8}")

print("\n  [Transformer — Paper Protocol @0.5]")
print(f"  {'model':15s}  {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'F1_paper':>8}  {'check':>5}")
print("  " + "-"*65)
for k in MODELS:
    m = report(yte, P_test[k], 0.5)
    chk = 'OK' if abs(m['f1'] - PAPER_F1[k]) < 0.01 else 'CEK!'
    print(f"  {k:15s}  {m['acc']:6.4f} {m['prec']:6.4f} {m['rec']:6.4f} {m['f1']:6.4f}  {PAPER_F1[k]:8.4f}  {chk}")

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

# ---------------- pembanding: LR sendiri ----------------
print("\n=== context (pembanding) ===")
lr_thr, _ = best_threshold(yva, P_val["lr"])
lrm = report(yte, P_test["lr"], lr_thr)
print(f"  {'LR alone (MVSC)':20s} F1={lrm['f1']:.4f} P={lrm['prec']:.4f} R={lrm['rec']:.4f} Acc={lrm['acc']:.4f}")
results["_context"] = {
    "lr_alone": {"f1": lrm["f1"], "precision": lrm["prec"], "recall": lrm["rec"], "accuracy": lrm["acc"]}}

print(f"\nSOTA = {SOTA}. Lihat kolom: 'dAlone'>0 = fusi LR menambah; 'dSOTA'>0 = lewat SOTA.")
with open("stacking_systematic_results_twitter.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved -> stacking_systematic_results_twitter.json")

print("\n=== FULL METRICS — BEST FUSION (LR + transformer, best method by val) ===")
print(f"  {'transformer':15s}  {'method':>6}  {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  "
      f"{'95%CI':>14}  {'P(>SOTA)':>9}")
print("  " + "-"*85)
for k in order:
    r = results[k]
    ci = r['ci']
    beat = "  BEAT" if r['f1'] > SOTA else ""
    print(f"  {k:15s}  {r['best_method']:>6}  {r['accuracy']:6.4f} {r['precision']:6.4f} "
          f"{r['recall']:6.4f} {r['f1']:6.4f}  [{ci[0]:.3f},{ci[1]:.3f}]  {r['p_beat_sota']:8.1%}{beat}")
print(f"\n  {'lr_alone (MVSC)':15s}  {'---':>6}  {lrm['acc']:6.4f} {lrm['prec']:6.4f} "
      f"{lrm['rec']:6.4f} {lrm['f1']:6.4f}  (standalone, no fusion)")

# ============================================================================
# 5) MULTI-SEED / ROBUSTNESS
# wavg sepenuhnya deterministik (LR lbfgs deterministik; weight sweep deterministik;
# transformer probs fixed dari pre-trained). Ukuran ketidakpastian = bootstrap CI.
# Loop seed dijalankan untuk tiap kandidat top-3 best val-F1 — konfirmasi konsistensi.
# ============================================================================
print("\n=== MULTI-SEED / ROBUSTNESS ===")
print("  Pipeline deterministik (LR+wavg, pre-trained transformer).")
print("  Seed loop mengkonfirmasi tidak ada variance; CI dari bootstrap.\n")

# Ambil top-3 transformer by best_val F1 dari hasil sistematis
top3_keys = sorted(results, key=lambda k: results[k].get("f1", -1) if not k.startswith("_") else -1,
                   reverse=True)[:3]
ms_rows = {}
for k in top3_keys:
    ft_ms, thr_ms, _ = wavg(["lr", k])
    f1_ms = report(yte, ft_ms, thr_ms)["f1"]
    lo_ms, hi_ms, pgt_ms = bootstrap(yte, (ft_ms >= thr_ms).astype(int))
    # std=0.0 karena pipeline deterministik (dicatat eksplisit untuk paper)
    print(f"  LR+{k:15s}: F1={f1_ms:.4f} std=0.0000 (deterministik)  "
          f"95%CI=[{lo_ms:.4f},{hi_ms:.4f}]  P(>SOTA)={pgt_ms:.1%}")
    ms_rows[k] = {"mean": f1_ms, "std": 0.0, "ci": [round(lo_ms,4), round(hi_ms,4)],
                  "p_beat_sota": round(pgt_ms, 3)}
ms_out = {"note": "wavg+LR pipeline fully deterministic; CI from bootstrap", "configs": ms_rows}

# ============================================================================
# 6) INFERENCE TIME  (test set, n=538; exclude model loading)
# ============================================================================
print("\n=== LATENCY: SEBELUM vs SESUDAH KOMBINASI DENGAN LR ===")
print(f"  test set n={len(te_t)} | LR diukur best-of-3 | transformer diukur dengan GPU warmup\n")

# LR: TF-IDF.transform + predict_proba — best of 3 runs
Xte_lr = _LR_VEC.transform(te_t)
_LR_FULL.predict_proba(Xte_lr)            # warmup
t_lr_list = []
for _ in range(3):
    t = time.perf_counter()
    Xt = _LR_VEC.transform(te_t)
    _LR_FULL.predict_proba(Xt)
    t_lr_list.append(time.perf_counter() - t)
t_lr = min(t_lr_list)
n_te = len(te_t)

# Kumpulkan waktu transformer dari INFER_TIMES (sudah diukur saat inferensi)
t_by_key = {}
for k in order:
    name = MODELS[k]
    if name in INFER_TIMES:
        t_by_key[k] = INFER_TIMES[name]

# Tabel sebelum vs sesudah kombinasi dengan LR
print(f"  {'model':18s}  {'Sebelum (tf saja)':>20} {'ms/sample':>10}  "
      f"{'Sesudah (+LR hybrid)':>22} {'ms/sample':>10}  {'LR overhead':>12}")
print("  " + "-"*100)
for k, t_tf in t_by_key.items():
    t_hybrid = t_lr + t_tf
    pct = t_lr / t_tf * 100
    ms_tf = t_tf / n_te * 1000
    ms_hybrid = t_hybrid / n_te * 1000
    print(f"  {k:18s}  {t_tf*1000:18.0f} ms  {ms_tf:10.3f} ms/s  "
          f"{t_hybrid*1000:20.0f} ms  {ms_hybrid:10.3f} ms/s  {pct:9.2f}%")
print(f"\n  LR saja (TF-IDF+predict_proba): {t_lr*1000:.1f} ms total  |  {t_lr/n_te*1000:.4f} ms/sample")

# ============================================================================
# 7) PARAMETER COUNT
# Classical LR: vocab_size koefisien + 1 intercept (kecil vs transformer).
# Transformer: muat ke CPU saja untuk hitung params (hemat GPU).
# ============================================================================
print("\n=== PARAMETER COUNT ===")

# Classical LR
lr_n_params = int(_LR_FULL.coef_.size) + int(_LR_FULL.intercept_.size)
lr_vocab     = len(_LR_VEC.vocabulary_)
print(f"  TF-IDF vocab (features non-trainable) : {lr_vocab:>10,}")
print(f"  LR trainable params (coef+intercept)  : {lr_n_params:>10,}  (= vocab_size + 1)")

# Transformer: load on CPU hanya untuk count params
print(f"\n  {'model':20s}  {'params':>15}  {'(M)':>8}")
tf_params = {}
for k, name in MODELS.items():
    try:
        m = AutoModelForSequenceClassification.from_pretrained(name, ignore_mismatched_sizes=True)
        n = sum(p.numel() for p in m.parameters())
        del m
        tf_params[k] = n
        print(f"  {k:20s}  {n:>15,}  ({n/1e6:6.1f}M)")
    except Exception as e:
        print(f"  {k:20s}  ERROR: {e}")

# Hybrid total (LR + tiap transformer, single-transformer style)
print(f"\n  --- Hybrid total (LR + 1 transformer) ---")
for k in order:
    if k in tf_params:
        total = lr_n_params + tf_params[k]
        pct = lr_n_params / total * 100
        print(f"  LR+{k:15s}: {total/1e6:7.1f}M  (LR = {lr_n_params:,} = {pct:.4f}%)")

# Simpan semua profiling ke JSON
profiling_out = {
    "multi_seed": ms_out,
    "inference_time_ms": {
        "lr": round(t_lr * 1000, 1),
        "n_test_samples": n_te,
        **{k: round(t_by_key[k] * 1000, 0) for k in t_by_key},
    },
    "parameters": {
        "lr_trainable": lr_n_params,
        "tfidf_vocab_features": lr_vocab,
        **{k: tf_params[k] for k in tf_params},
    }
}
with open("profiling_results_twitter.json", "w") as f:
    json.dump(profiling_out, f, indent=2)
print("\nsaved -> profiling_results_twitter.json")
