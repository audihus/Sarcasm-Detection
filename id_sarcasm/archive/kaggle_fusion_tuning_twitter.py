# ============================================================================
# Fusion Operator Tuning — LR × Transformer (Twitter Indonesia Sarcastic)
# Membandingkan 3 operator fusi × 2 kondisi kalibrasi = 6 config per transformer.
#
# Operator fusi:
#   prob-avg  : w×p_LR + (1-w)×p_tf  (baseline)
#   logit-avg : rata-rata di ruang log-odds lalu balik ke prob (lebih principled)
#   rank-avg  : rata-rata peringkat (rank), robust terhadap perbedaan skala
#
# Kalibrasi (temperature scaling):
#   none : prob langsung dari model
#   temp : tiap model dikalibrasi satu angka T (dicari di val, minimize NLL)
#          logit_baru = logit_asli / T  -> prob lebih "moderat"
#
# Protokol jujur: T, bobot w, threshold SEMUA dipilih dari val. Test disentuh sekali.
#
# KAGGLE: Settings -> GPU T4 x1, Internet = ON.
#   !pip install -q transformers==4.46.3 datasets==3.1.0 evaluate==0.4.3 \
#                   accelerate==1.1.1 scikit-learn nltk sentencepiece scipy
# ============================================================================
import warnings, json, time
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")

DATASET  = "w11wo/twitter_indonesia_sarcastic"
TEXT_COL = "tweet"
SEED, MAX_LEN, BATCH = 42, 128, 32
SOTA = 0.7692
np.random.seed(SEED)

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
tr_t = list(ds["train"][TEXT_COL]);  va_t = list(ds["validation"][TEXT_COL]);  te_t = list(ds["test"][TEXT_COL])
ytr = np.array(ds["train"]["label"]); yva = np.array(ds["validation"]["label"]); yte = np.array(ds["test"]["label"])
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)}")

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
    return {"f1": round(f1,4), "prec": round(pr,4), "rec": round(rc,4), "acc": round(accuracy_score(y,p),4)}

def bootstrap(y, pred, n=5000):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(f1_score(y[b], pred[b], zero_division=0))
    out = np.array(out)
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5)), float((out>SOTA).mean())

# ---------------- operator fusi ----------------
def _logit(p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return np.log(p / (1-p))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fuse(pa, pb, w, operator):
    """Gabungkan dua prob-array dengan bobot w untuk pa, (1-w) untuk pb."""
    if operator == "prob":
        return w * pa + (1-w) * pb
    elif operator == "logit":
        return _sigmoid(w * _logit(pa) + (1-w) * _logit(pb))
    elif operator == "rank":
        ra = rankdata(pa) / len(pa)
        rb = rankdata(pb) / len(pb)
        return w * ra + (1-w) * rb

def sweep_weight(pv_a, pv_b, pt_a, pt_b, operator):
    """Cari bobot w terbaik di val, lalu hasilkan skor test."""
    best = (-1.0, 0.5, 0.5)   # val_f1, w, thr
    for w in np.linspace(0, 1, 41):
        fv = fuse(pv_a, pv_b, w, operator)
        thr, f1 = best_threshold(yva, fv)
        if f1 > best[0]: best = (f1, w, thr)
    val_f1, w, thr = best
    ft = fuse(pt_a, pt_b, w, operator)
    return ft, thr, val_f1, round(w, 3)

# ---------------- temperature scaling ----------------
def find_temperature(val_probs, y_val):
    """Cari T terbaik di val dengan minimize NLL (negative log-likelihood)."""
    def nll(T):
        p = _sigmoid(_logit(val_probs) / T)
        p = np.clip(p, 1e-7, 1-1e-7)
        return -np.mean(y_val * np.log(p) + (1-y_val) * np.log(1-p))
    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)

def apply_temp(probs, T):
    """Terapkan temperature T: kalibrasikan prob."""
    return _sigmoid(_logit(probs) / T)

# ---------------- classical LR ----------------
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception: pass
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import clone
from scipy.sparse import vstack

def get_lr_probs():
    vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None,
                          ngram_range=(1,2), min_df=2, sublinear_tf=True)
    Xtr, Xva, Xte = vec.fit_transform(tr_t), vec.transform(va_t), vec.transform(te_t)
    base = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
    ps = PredefinedSplit([-1]*len(ytr) + [0]*len(yva))
    gs = GridSearchCV(base, {"C":[0.05,0.1,0.3,1,3,10,30]}, scoring="f1", cv=ps, n_jobs=-1, refit=True)
    gs.fit(vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
    valm = clone(base).set_params(C=gs.best_params_["C"]).fit(Xtr, ytr)
    return valm.predict_proba(Xva)[:,1], gs.best_estimator_.predict_proba(Xte)[:,1]

print("[LR] fitting ...")
P_val = {}; P_test = {}
P_val["lr"], P_test["lr"] = get_lr_probs()

# ---------------- transformer inference ----------------
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

def _pos(cfg):
    l2i = getattr(cfg, "label2id", None) or {}
    for k in ("1", "LABEL_1"):
        if k in l2i: return int(l2i[k])
    for k, v in l2i.items():
        if "sarc" in str(k).lower(): return int(v)
    return 1

@torch.no_grad()
def transformer_probs(model_name, texts):
    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "model_type","") in ("xlm-roberta","roberta"):
        cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(DEVICE).eval()
    pos, probs = _pos(model.config), []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i+BATCH], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        probs.append(torch.softmax(model(**enc).logits, dim=-1)[:,pos].cpu().numpy())
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return np.concatenate(probs)

order = sorted(MODELS, key=lambda k: PAPER_F1[k], reverse=True)
for key, name in MODELS.items():
    print(f"[infer] {key} ...")
    P_val[key]  = transformer_probs(name, va_t)
    P_test[key] = transformer_probs(name, te_t)

# ---- sanity: transformer @0.5 vs paper ----
print("\n=== SANITY (transformer @0.5) ===")
for k in MODELS:
    f = report(yte, P_test[k], 0.5)["f1"]
    print(f"  {k:18s} F1@0.5={f:.4f}  paper={PAPER_F1[k]:.4f}  {'OK' if abs(f-PAPER_F1[k])<0.01 else 'CEK!'}")

# ============================================================================
# MAIN LOOP — 3 operator × 2 kalibrasi = 6 config per transformer
# ============================================================================
OPERATORS = ["prob", "logit", "rank"]
TEMPS     = [False, True]

print("\n=== FUSION OPERATOR TUNING ===")
print(f"  {'transformer':18s}  {'operator':8s}  {'temp':4s}  {'T_lr':5s}  {'T_tf':5s}  "
      f"{'w':5s}  {'val_F1':7s}  {'test_F1':8s}  {'CI':15s}  {'P(>SOTA)':9s}")
print("  " + "-"*100)

all_results = []

for key in order:
    pv_tf = P_val[key];  pt_tf = P_test[key]
    pv_lr = P_val["lr"]; pt_lr = P_test["lr"]

    # LR alone baseline (once per transformer, untuk hitung val-test gap)
    lr_thr, _ = best_threshold(yva, pv_lr)
    lr_test_f1 = report(yte, pt_lr, lr_thr)["f1"]

    for use_temp in TEMPS:
        # Temperature scaling: cari T per model di val
        if use_temp:
            T_lr = find_temperature(pv_lr, yva)
            T_tf = find_temperature(pv_tf, yva)
            pv_lr_c = apply_temp(pv_lr, T_lr)
            pt_lr_c = apply_temp(pt_lr, T_lr)
            pv_tf_c = apply_temp(pv_tf, T_tf)
            pt_tf_c = apply_temp(pt_tf, T_tf)
        else:
            T_lr = T_tf = 1.0
            pv_lr_c, pt_lr_c = pv_lr, pt_lr
            pv_tf_c, pt_tf_c = pv_tf, pt_tf

        for op in OPERATORS:
            ft, thr, val_f1, w = sweep_weight(pv_lr_c, pv_tf_c, pt_lr_c, pt_tf_c, op)
            m  = report(yte, ft, thr)
            lo, hi, pgt = bootstrap(yte, (ft >= thr).astype(int))
            gap = round(val_f1 - m["f1"], 4)   # val-test gap (flag overfit)
            temp_str = "yes" if use_temp else "no"
            beat = " ← BEAT" if m["f1"] > SOTA else ""
            print(f"  {key:18s}  {op:8s}  {temp_str:4s}  {T_lr:5.2f}  {T_tf:5.2f}  "
                  f"{w:5.3f}  {val_f1:.4f}  {m['f1']:.4f}  [{lo:.3f},{hi:.3f}]  {pgt:7.1%}{beat}")
            all_results.append({
                "transformer": key, "operator": op, "temp": use_temp,
                "T_lr": round(T_lr,3), "T_tf": round(T_tf,3), "w": w,
                "val_f1": round(val_f1,4), "test_f1": m["f1"],
                "prec": m["prec"], "rec": m["rec"], "acc": m["acc"],
                "thr": round(thr,4), "ci": [round(lo,4), round(hi,4)],
                "p_beat_sota": round(pgt,3), "val_test_gap": gap,
            })
    print()   # baris kosong antar transformer

# ============================================================================
# RINGKASAN: best config per transformer (by val_F1), lalu ranking by test_F1
# ============================================================================
print("\n=== BEST CONFIG PER TRANSFORMER (by val_F1) ===")
print(f"  {'transformer':18s}  {'operator':8s}  {'temp':4s}  {'val_F1':7s}  "
      f"{'test_F1':8s}  {'CI':15s}  {'P(>SOTA)':9s}  {'val-test_gap':13s}")
print("  " + "-"*95)

best_per_tf = {}
for key in order:
    rows = [r for r in all_results if r["transformer"] == key]
    best = max(rows, key=lambda r: r["val_f1"])
    best_per_tf[key] = best
    beat = " ← BEAT SOTA" if best["test_f1"] > SOTA else ""
    print(f"  {key:18s}  {best['operator']:8s}  {'yes' if best['temp'] else 'no':4s}  "
          f"{best['val_f1']:.4f}  {best['test_f1']:.4f}  "
          f"[{best['ci'][0]:.3f},{best['ci'][1]:.3f}]  {best['p_beat_sota']:7.1%}  "
          f"{best['val_test_gap']:+.4f}{beat}")

# Baseline pembanding
lr_thr_g, _ = best_threshold(yva, P_val["lr"])
lr_f1 = report(yte, P_test["lr"], lr_thr_g)["f1"]
print(f"\n  Baseline: LR alone = {lr_f1:.4f} | SOTA = {SOTA}")

# Config terbaik keseluruhan (by val_F1)
overall_best = max(all_results, key=lambda r: r["val_f1"])
print(f"\n>> OVERALL BEST (by val): {overall_best['transformer']} "
      f"{overall_best['operator']} temp={'yes' if overall_best['temp'] else 'no'} "
      f"-> test F1 = {overall_best['test_f1']:.4f}  P(>SOTA) = {overall_best['p_beat_sota']:.1%}")

# ============================================================================
# SIMPAN
# ============================================================================
output = {
    "all_configs": all_results,
    "best_per_transformer": {k: best_per_tf[k] for k in order},
    "overall_best_by_val": overall_best,
    "baseline": {"lr_alone": lr_f1, "SOTA": SOTA},
}
with open("fusion_tuning_results_twitter.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nsaved -> fusion_tuning_results_twitter.json")
