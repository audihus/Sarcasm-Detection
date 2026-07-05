# ============================================================================
# ORACLE DIAGNOSTIC — apakah selective cascade (model-3 sebagai penengah) layak
# dibangun? (Twitter). Motivasi: Tahap A (kaggle_complementarity_gated_twitter.py)
# menemukan oracle F1 (LR+xlmr_base) = 0.9058 vs hasil fusi nyata terbaik 0.7900 —
# headroom +0.116. Semua gating (Tahap B/C/D) gagal menembusnya karena hanya
# me-reweight 2 sinyal lama & overfit ke val=268. Fase ini mendiagnosis APAKAH
# ada informasi baru (model-3) yang bisa menembus headroom itu — SEBELUM
# membangun cascade penuh.
#
# Tahap 0: sanity — 6 PAPER_F1 @0.5, LR(tuned)=0.7509, B2(LR+xlmr_base)=0.7900.
# Tahap 1: peta ceiling oracle — pairwise LR x tiap transformer (6 pasang) +
#          grand oracle (LR+6 transformer, coverage reachable).
# Tahap 2: probe separability — apakah "siapa benar" pada kasus disagreement
#          LR<->xlmr_base bisa diprediksi dari sinyal yang ADA (confidence,
#          disagreement magnitude, panjang teks, jumlah token PII-mask)?
#          AUROC univariat (deskriptif) + router logistik via CV DI DALAM val
#          disagreement (~50 sampel) -> OOF AUROC honest.
# Tahap 3: akurasi tiap kandidat model-3 PADA kasus disagreement (syarat perlu
#          cascade): korelasi dgn xlmr_base (indikator dekorelasi) + akurasi.
#          Model-3 dipilih BY-VAL, angka test dilaporkan sekali.
# Tahap 4: outcome cascade nol-parameter (deterministik, model-3 val-selected):
#          sepakat -> hasil sepakat; beda -> ikut model-3. F1/latency/uji
#          berpasangan + VERDICT HIJAU/MERAH (aturan keputusan eksplisit).
#
# PROTOKOL ANTI-KEBOCORAN (wajib dibaca sebelum interpretasi hasil):
#   - Oracle F1 (Tahap 1) memakai label test untuk memilih prediksi benar saat
#     model beda pendapat -> SAH sebagai upper bound diagnostik ("oracle
#     ceiling", praktik standar), TAPI HARAM dilaporkan sebagai F1 metode yang
#     bisa dideploy. Selalu berlabel "oracle / ceiling tak tercapai".
#   - Semua SELEKSI (model-3, ambang, fitur router) dari VAL SAJA; test
#     disentuh SEKALI per metode final (Tahap 4). Tabel test semua kandidat
#     (Tahap 3) hanya utk transparansi -- pemenang = yang terpilih BY-VAL.
#   - Probe separability (Tahap 2) via CV DI DALAM val disagreement (bukan fit
#     lalu diukur di test yang sama). AUROC di test hanya deskriptif.
#
# KAGGLE: Settings -> GPU T4 x1, Internet = ON. Pin versi spt baseline:
#   !pip install -q transformers==4.46.3 datasets==3.1.0 evaluate==0.4.3 \
#                   accelerate==1.1.1 scikit-learn nltk sentencepiece
#
# SMOKE TEST LOKAL (tanpa GPU/torch): set env SMOKE=1 -> prob transformer
# diganti sinyal SINTETIS agar seluruh jalur kode teruji. Angka mode smoke
# BUKAN hasil riset.
#   PowerShell: $env:SMOKE="1"; python kaggle_oracle_diagnostic_twitter.py
#
# Output: oracle_diagnostic_results_twitter.json
#         fig_oracle_ceiling.png / fig_model3_diagnostic.png
#         -> salin ke results/oracle_diagnostic/ di repo.
# ============================================================================
import os, sys, re, warnings, json, time
import numpy as np
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

SMOKE = os.environ.get("SMOKE", "0") == "1"

DATASET  = "w11wo/twitter_indonesia_sarcastic"
TEXT_COL = "tweet"
SEED, MAX_LEN, BATCH = 41, 128, 32
SOTA   = 0.7692   # XLM-R large @0.5 (paper)
B2_REF = 0.7900   # LR + xlmr_base prob-avg (kaggle_stacking_full_twitter.py)
LR_REF = 0.7509   # LR (tuned), val-tuned threshold (protokol bersih)
N_BOOT = 5000
np.random.seed(SEED)

# 6 model fine-tuned penulis paper (identik kaggle_stacking_full_twitter.py).
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
MODEL3_CANDIDATES = [k for k in MODELS if k != "xlmr_base"]   # anchor pair = LR<->xlmr_base

# ---------------- data ----------------
from datasets import load_dataset
ds = load_dataset(DATASET)
tr_t, va_t, te_t = list(ds["train"][TEXT_COL]), list(ds["validation"][TEXT_COL]), list(ds["test"][TEXT_COL])
ytr = np.array(ds["train"]["label"]); yva = np.array(ds["validation"]["label"]); yte = np.array(ds["test"]["label"])
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} (pos {yte.sum()}){'  [SMOKE MODE]' if SMOKE else ''}")

# ---------------- helpers (verbatim/konsisten dgn kaggle_complementarity_gated_twitter.py) ----------------
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, roc_auc_score

def best_threshold(y, s):
    """F1-maximizing threshold pada skor val — vektorisasi (identik loop np.unique)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    ss, ys = s[order], y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); pos = max(ys.sum(), 1)
    valid = np.r_[ss[1:] != ss[:-1], True]
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / pos
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    f1 = np.where(valid, f1, -1.0)
    i = int(np.argmax(f1))
    return float(ss[i]), float(f1[i])

def best_threshold_slow(y, s):
    bt, bf = 0.5, -1.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)

def report(y, s, thr):
    p = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"f1": round(f1, 4), "prec": round(pr, 4), "rec": round(rc, 4), "acc": round(accuracy_score(y, p), 4)}

def binary_report(y, pred):
    """Sama seperti report() tapi input sudah berupa label biner (bukan skor+threshold)."""
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"f1": round(float(f1), 4), "prec": round(float(pr), 4), "rec": round(float(rc), 4),
            "acc": round(float(accuracy_score(y, pred)), 4)}

def bootstrap(y, pred, n=N_BOOT):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(f1_score(y[b], pred[b], zero_division=0))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out > SOTA).mean())

def paired_bootstrap(y, pred_a, pred_b, n=N_BOOT):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        d.append(f1_score(y[b], pred_a[b], zero_division=0) -
                 f1_score(y[b], pred_b[b], zero_division=0))
    d = np.array(d)
    return {"d_f1_mean": round(float(d.mean()), 4),
            "d_f1_ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_a_beats_b": round(float((d > 0).mean()), 3)}

def auroc_safe(y, score):
    y = np.asarray(y)
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, score))

if SMOKE:   # self-test: best_threshold vektorisasi == versi loop asli
    rng0 = np.random.default_rng(0)
    for _ in range(20):
        ys_, ss_ = rng0.integers(0, 2, 97), np.round(rng0.random(97), 3)
        t_fast, f_fast = best_threshold(ys_, ss_); t_slow, f_slow = best_threshold_slow(ys_, ss_)
        assert abs(f_fast - f_slow) < 1e-9, f"best_threshold mismatch: {f_fast} vs {f_slow}"
    print("[SMOKE] self-test best_threshold: OK (identik dgn versi loop)")

# ---------------- 1) classical LR (tuned) — identik script2 sebelumnya ----------------
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception: pass
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from scipy.sparse import vstack

print("\n[LR] fitting TF-IDF(1,2) + LogisticRegression (GridSearch C, scoring=f1) ...")
_LR_VEC = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True)
Xtr = _LR_VEC.fit_transform(tr_t); Xva = _LR_VEC.transform(va_t); Xte = _LR_VEC.transform(te_t)
_base = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
_ps = PredefinedSplit([-1] * len(ytr) + [0] * len(yva))
_gs = GridSearchCV(_base, {"C": [0.05, 0.1, 0.3, 1, 3, 10, 30]}, scoring="f1", cv=_ps, n_jobs=-1, refit=True)
_gs.fit(vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
_valm = clone(_base).set_params(C=_gs.best_params_["C"]).fit(Xtr, ytr)
_LR_FULL = _gs.best_estimator_
P_val, P_test = {}, {}
P_val["lr"]  = _valm.predict_proba(Xva)[:, 1]
P_test["lr"] = _LR_FULL.predict_proba(Xte)[:, 1]
print(f"  best C = {_gs.best_params_['C']}")

def measure_lr_latency(n=3):
    _LR_FULL.predict_proba(_LR_VEC.transform(te_t))   # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _LR_FULL.predict_proba(_LR_VEC.transform(te_t))
        times.append(time.perf_counter() - t0)
    return min(times)

# ---------------- 2) transformer probs, 6 model paper (inference only) ----------------
INFER_TIMES = {}

if not SMOKE:
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", DEVICE)

    def _pos(cfg):
        l2i = getattr(cfg, "label2id", None) or {}
        for k in ("1", "LABEL_1"):
            if k in l2i: return int(l2i[k])
        for kk, v in l2i.items():
            if "sarc" in str(kk).lower(): return int(v)
        return 1

    @torch.no_grad()
    def transformer_probs(model_name, texts, time_key=None):
        cfg = AutoConfig.from_pretrained(model_name)
        if getattr(cfg, "model_type", "") in ("xlm-roberta", "roberta"):
            cfg._attn_implementation = "eager"          # reproduksi paper persis (SDPA != eager)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(DEVICE).eval()
        pos, probs = _pos(model.config), []
        if time_key:
            enc0 = tok(texts[:BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            model(**enc0)
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            probs.append(torch.softmax(model(**enc).logits, dim=-1)[:, pos].cpu().numpy())
        if time_key:
            if DEVICE == "cuda": torch.cuda.synchronize()
            INFER_TIMES[time_key] = time.perf_counter() - t0
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
        return np.concatenate(probs)

    for key, name in MODELS.items():
        print(f"[infer] {key} ...")
        P_val[key]  = transformer_probs(name, va_t)
        P_test[key] = transformer_probs(name, te_t, time_key=key)
else:
    # ---- sinyal sintetis: HANYA untuk menguji jalur kode secara lokal ----
    SKILL = {"indobert_base": 0.68, "indobert_large": 0.66, "indobert_lem": 0.58,
             "mbert": 0.58, "xlmr_base": 0.70, "xlmr_large": 0.74}
    def _synth(y, strength, seed):
        rng = np.random.default_rng(seed)
        return np.clip(strength * y + (1 - strength) * (1 - y) * 0.4 + rng.normal(0, 0.28, len(y)), 0.01, 0.99)
    for i, key in enumerate(MODELS):
        P_val[key]  = _synth(yva, SKILL[key], 100 + i)
        P_test[key] = _synth(yte, SKILL[key], 200 + i)
        INFER_TIMES[key] = float("nan")

# ---------------- 3) fitur teks (struktural, BUKAN leksikon sentimen) ----------------
PII_RE = re.compile(r"<\w+>")   # dataset sudah PII-masked: <username>, <hashtag>, <url>, dst.
def text_feats(texts):
    char_len = np.array([len(t) for t in texts], dtype=float)
    tok_len  = np.array([len(t.split()) for t in texts], dtype=float)
    pii_cnt  = np.array([len(PII_RE.findall(t)) for t in texts], dtype=float)
    return char_len, tok_len, pii_cnt
char_va, tok_va, pii_va = text_feats(va_t)
char_te, tok_te, pii_te = text_feats(te_t)

# ============================================================================
# TAHAP 0 — SANITY CHECK
# ============================================================================
print("\n=== TAHAP 0: SANITY CHECK ===")
lr_thr, lr_val_f1 = best_threshold(yva, P_val["lr"])
m_lr = report(yte, P_test["lr"], lr_thr)

def wavg_fit(pv_a, pv_b, pt_a, pt_b):
    best = (-1.0, 0.5, 0.5)
    for w in np.linspace(0, 1, 41):
        thr, f1 = best_threshold(yva, w * pv_a + (1 - w) * pv_b)
        if f1 > best[0]: best = (f1, w, thr)
    vf, w, thr = best
    return w * pt_a + (1 - w) * pt_b, thr, vf, w

b2_scores, b2_thr, b2_val_f1, b2_w = wavg_fit(P_val["lr"], P_val["xlmr_base"], P_test["lr"], P_test["xlmr_base"])
m_b2 = report(yte, b2_scores, b2_thr)
pred_b2 = (b2_scores >= b2_thr).astype(int)

sanity_rows = [("LR (tuned, val-thr)", m_lr["f1"], LR_REF)]
for k in MODELS:
    sanity_rows.append((f"{k} @0.5", report(yte, P_test[k], 0.5)["f1"], PAPER_F1[k]))
sanity_rows.append(("B2 LR+xlmr_base wavg", m_b2["f1"], B2_REF))

sanity_ok = True
for name, got, ref in sanity_rows:
    ok = abs(got - ref) < 0.002
    if SMOKE and "LR" not in name.split()[0]: status = "SKIP (smoke)"
    else:
        status = "OK" if ok else "CEK!"
        if not ok: sanity_ok = False
    print(f"  {name:24s} F1={got:.4f}  ref={ref:.4f}  [{status}]")
if not sanity_ok and not SMOKE:
    print("\n  !! SANITY GAGAL — hentikan analisis, cek versi transformers/attention (eager) !!")
    sys.exit(1)

pred_lr = (P_test["lr"] >= lr_thr).astype(int)
pred_tf = {k: (P_test[k] >= 0.5).astype(int) for k in MODELS}
pred_sota = pred_tf["xlmr_large"]

OUT = {"_meta": {"dataset": DATASET, "seed": SEED, "n_boot": N_BOOT, "smoke": SMOKE,
                 "refs": {"SOTA_xlmr_large": SOTA, "B2_wavg": B2_REF, "LR_tuned": LR_REF},
                 "b2": {"w_lr": round(float(b2_w), 3), "thr": round(b2_thr, 4),
                        "val_f1": round(b2_val_f1, 4), **m_b2}},
       "stage0_sanity": {n: {"f1": g, "ref": r} for n, g, r in sanity_rows}}

# ============================================================================
# TAHAP 1 — PETA CEILING ORACLE (pairwise LR x tiap transformer + grand oracle)
# ============================================================================
print("\n=== TAHAP 1: PETA CEILING ORACLE ===")
stage1 = {"pairwise": {}}
print(f"  {'pasangan':22s} {'r_prob':>7} {'disagree':>9} {'oracleF1':>9}")
print("  " + "-" * 52)
for k in MODELS:
    pb = pred_tf[k]
    r_prob = float(np.corrcoef(P_test["lr"], P_test[k])[0, 1])
    dis = pred_lr != pb
    oracle_pred = np.where(dis, yte, pred_lr)          # SAH sbg ceiling; TIDAK bisa dideploy
    o_f1 = float(f1_score(yte, oracle_pred, zero_division=0))
    stage1["pairwise"][k] = {"r_prob": round(r_prob, 4), "n_disagree": int(dis.sum()), "oracle_f1": round(o_f1, 4)}
    print(f"  {'lr vs '+k:22s} {r_prob:7.3f} {int(dis.sum()):6d} ({dis.mean():4.1%})  {o_f1:9.4f}")

if not SMOKE:   # cross-check independen vs run kaggle_complementarity_gated_twitter.py sebelumnya
    prev_ref = {"xlmr_base": (0.686, 0.9058), "xlmr_large": (0.669, 0.9154)}
    print("\n  cross-check vs run kaggle_complementarity_gated_twitter.py sebelumnya:")
    for k, (r_ref, o_ref) in prev_ref.items():
        r_got, o_got = stage1["pairwise"][k]["r_prob"], stage1["pairwise"][k]["oracle_f1"]
        flag = "OK" if abs(r_got - r_ref) < 0.01 and abs(o_got - o_ref) < 0.01 else "BEDA (cek data/seed)"
        print(f"    lr|{k}: r={r_got:.3f} (ref {r_ref}), oracle={o_got:.4f} (ref {o_ref})  [{flag}]")

all_preds = np.column_stack([pred_lr] + [pred_tf[k] for k in MODELS])
correct_mask = (all_preds == yte[:, None])
reachable = correct_mask.any(axis=1)
majority = (all_preds.mean(axis=1) >= 0.5).astype(int)
grand_pred = np.where(reachable, yte, majority)         # unreachable -> majority (dijamin salah, realistis)
grand_f1 = float(f1_score(yte, grand_pred, zero_division=0))
coverage = float(reachable.mean())
print(f"\n  GRAND ORACLE (LR + 6 transformer): F1={grand_f1:.4f}  coverage(reachable)={coverage:.1%}"
      f"  ({int((~reachable).sum())} sampel tak terselamatkan model apa pun)")
stage1["grand_oracle"] = {"f1": round(grand_f1, 4), "coverage": round(coverage, 4),
                          "n_unreachable": int((~reachable).sum())}
OUT["stage1_oracle_ceiling"] = stage1

# ============================================================================
# TAHAP 2 — PROBE SEPARABILITY (anchor LR <-> xlmr_base, kasus disagreement)
# ============================================================================
pred_lr_va = (P_val["lr"] >= lr_thr).astype(int)
pred_tf_va = {k: (P_val[k] >= 0.5).astype(int) for k in MODELS}
dis_va = pred_lr_va != pred_tf_va["xlmr_base"]
dis_te = pred_lr != pred_tf["xlmr_base"]

print("\n=== TAHAP 2: PROBE SEPARABILITY (LR vs xlmr_base, kasus disagreement) ===")
target_va = (pred_lr_va[dis_va] == yva[dis_va]).astype(int)   # 1 = LR benar pada kasus disagree
target_te = (pred_lr[dis_te] == yte[dis_te]).astype(int)
base_rate_va = float(target_va.mean()) if len(target_va) else float("nan")
base_rate_te = float(target_te.mean()) if len(target_te) else float("nan")
print(f"  disagreement: val n={int(dis_va.sum())} (base rate LR-benar={base_rate_va:.1%})  "
      f"test n={int(dis_te.sum())} (base rate LR-benar={base_rate_te:.1%})")

FEATS_VA = {"conf_lr": np.abs(P_val["lr"] - 0.5), "conf_tf": np.abs(P_val["xlmr_base"] - 0.5),
            "disagree_mag": np.abs(P_val["lr"] - P_val["xlmr_base"]),
            "text_len_char": char_va, "text_len_tok": tok_va, "pii_mask_count": pii_va}
FEATS_TE = {"conf_lr": np.abs(P_test["lr"] - 0.5), "conf_tf": np.abs(P_test["xlmr_base"] - 0.5),
            "disagree_mag": np.abs(P_test["lr"] - P_test["xlmr_base"]),
            "text_len_char": char_te, "text_len_tok": tok_te, "pii_mask_count": pii_te}

print(f"\n  {'fitur':16s} {'AUROC val(deskr.)':>10}  {'AUROC test(deskr.)':>12}")
stage2 = {"base_rate": {"val": round(base_rate_va, 4), "test": round(base_rate_te, 4)}, "univariate_auroc": {}}
for fname in FEATS_VA:
    auroc_va = auroc_safe(target_va, FEATS_VA[fname][dis_va])
    auroc_te = auroc_safe(target_te, FEATS_TE[fname][dis_te])
    stage2["univariate_auroc"][fname] = {"val_descriptive": round(auroc_va, 4), "test_descriptive": round(auroc_te, 4)}
    print(f"  {fname:16s} {auroc_va:10.4f}  {auroc_te:14.4f}")

X_va_dis = np.column_stack([FEATS_VA[f][dis_va] for f in FEATS_VA])
router_auroc, router_k = None, None
if len(target_va) >= 6 and len(np.unique(target_va)) == 2:
    router_auroc, router_k = (lambda X, y: (
        (lambda k: (float(roc_auc_score(y, cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
            X, y, cv=StratifiedKFold(k, shuffle=True, random_state=SEED), method="predict_proba")[:, 1])), k)
         if k >= 2 else (None, None))(int(min(5, np.bincount(y, minlength=2).min())))
    ))(X_va_dis, target_va)
if router_auroc is not None:
    print(f"\n  Router logistik (6 fitur, {router_k}-fold CV DI DALAM val disagreement): AUROC OOF = {router_auroc:.4f}")
else:
    print("\n  Router: val disagreement terlalu kecil/tidak seimbang utk CV")
stage2["router_cv_auroc_val"] = round(router_auroc, 4) if router_auroc is not None else None
stage2["router_cv_k"] = router_k
verdict2 = "ADA SINYAL" if (router_auroc and router_auroc > 0.60) else "TIDAK ADA SINYAL KUAT (~acak)"
base_side = max(base_rate_va, 1 - base_rate_va) if not np.isnan(base_rate_va) else float("nan")
print(f"  -> {verdict2}  (ambang 0.60; baseline 'selalu pilih sisi mayoritas'={base_side:.1%})")
stage2["verdict"] = verdict2
OUT["stage2_separability"] = stage2

# ============================================================================
# TAHAP 3 — AKURASI MODEL-3 PADA KASUS DISAGREEMENT (syarat perlu cascade)
# ============================================================================
print("\n=== TAHAP 3: KANDIDAT MODEL-3 PADA KASUS DISAGREEMENT ===")
print(f"  {'kandidat':16s} {'corr_val':>8} {'corr_test':>9} {'acc_dis_val':>11} {'acc_dis_test':>12}")
print("  " + "-" * 62)
stage3 = {}
lr_side_te = dis_te & (pred_lr == yte)                     # 54 kasus: LR benar, TF salah
tf_side_te = dis_te & (pred_tf["xlmr_base"] == yte)         # 43 kasus: TF benar, LR salah
for m3 in MODEL3_CANDIDATES:
    corr_va = float(np.corrcoef(P_val[m3], P_val["xlmr_base"])[0, 1])
    corr_te = float(np.corrcoef(P_test[m3], P_test["xlmr_base"])[0, 1])
    acc_dv = float(accuracy_score(yva[dis_va], pred_tf_va[m3][dis_va])) if dis_va.sum() else float("nan")
    acc_dt = float(accuracy_score(yte[dis_te], pred_tf[m3][dis_te])) if dis_te.sum() else float("nan")
    follow_lr = float(accuracy_score(yte[lr_side_te], pred_tf[m3][lr_side_te])) if lr_side_te.sum() else float("nan")
    follow_tf = float(accuracy_score(yte[tf_side_te], pred_tf[m3][tf_side_te])) if tf_side_te.sum() else float("nan")
    stage3[m3] = {"corr_val": round(corr_va, 4), "corr_test": round(corr_te, 4),
                  "acc_disagree_val": round(acc_dv, 4), "acc_disagree_test": round(acc_dt, 4),
                  "follow_lr_when_lr_right_test": round(follow_lr, 4),
                  "follow_tf_when_tf_right_test": round(follow_tf, 4)}
    print(f"  {m3:16s} {corr_va:8.3f} {corr_te:9.3f} {acc_dv:11.1%} {acc_dt:12.1%}")

winner = max(MODEL3_CANDIDATES, key=lambda m: (stage3[m]["acc_disagree_val"], -stage3[m]["corr_val"]))
print(f"\n  -> model-3 TERPILIH BY-VAL (akurasi disagreement val tertinggi, tie-break korelasi terendah): {winner}")
print(f"     val_acc={stage3[winner]['acc_disagree_val']:.1%}  test_acc={stage3[winner]['acc_disagree_test']:.1%}"
      f"  corr_test={stage3[winner]['corr_test']:.3f}")
stage3["_winner"] = winner
OUT["stage3_model3_candidates"] = stage3

# ============================================================================
# TAHAP 4 — OUTCOME CASCADE NOL-PARAMETER (deterministik, model-3 val-selected)
# ============================================================================
print(f"\n=== TAHAP 4: CASCADE NOL-PARAMETER (model-3 = {winner}) ===")
pred_m3_te, pred_m3_va = pred_tf[winner], pred_tf_va[winner]
cascade_te = np.where(pred_lr == pred_tf["xlmr_base"], pred_lr, pred_m3_te)
cascade_va = np.where(pred_lr_va == pred_tf_va["xlmr_base"], pred_lr_va, pred_m3_va)

m_casc_te = binary_report(yte, cascade_te)
m_casc_va = binary_report(yva, cascade_va)
gap = round(m_casc_va["f1"] - m_casc_te["f1"], 4)
lo, hi, pgt = bootstrap(yte, cascade_te)
vs_b2 = paired_bootstrap(yte, cascade_te, pred_b2)
vs_sota = paired_bootstrap(yte, cascade_te, pred_sota)

n_dis_te = int(dis_te.sum())
t_lr = measure_lr_latency()
t_base = INFER_TIMES.get("xlmr_base", float("nan"))
t_m3 = INFER_TIMES.get(winner, float("nan"))
frac_dis = n_dis_te / len(te_t)
t_b2 = t_lr + t_base
t_cascade = t_b2 + t_m3 * frac_dis
t_sota = INFER_TIMES.get("xlmr_large", float("nan"))

print(f"  cascade: val F1={m_casc_va['f1']:.4f}  test F1={m_casc_te['f1']:.4f}  gap={gap:+.4f}")
print(f"  test: P={m_casc_te['prec']:.4f} R={m_casc_te['rec']:.4f} Acc={m_casc_te['acc']:.4f}"
      f"  CI95=[{lo:.3f},{hi:.3f}]  P(>SOTA)={pgt:.1%}")
print(f"  vs B2:   dF1={vs_b2['d_f1_mean']:+.4f}  P(cascade>B2)={vs_b2['p_a_beats_b']:.1%}")
print(f"  vs SOTA: dF1={vs_sota['d_f1_mean']:+.4f}  P(cascade>SOTA)={vs_sota['p_a_beats_b']:.1%}")
if not (np.isnan(t_base) or np.isnan(t_m3)):
    print(f"  latency: B2(LR+xlmr_base)={t_b2*1000:.0f}ms | cascade(+{winner} on {n_dis_te}/{len(te_t)} kasus)="
          f"{t_cascade*1000:.0f}ms | SOTA(xlmr_large alone)={t_sota*1000:.0f}ms")

acc_dis_test_winner = stage3[winner]["acc_disagree_test"]
GAP_THRESHOLD = 0.05
hijau = (acc_dis_test_winner > 0.60) and (m_casc_te["f1"] > B2_REF) and (abs(gap) < GAP_THRESHOLD)
verdict = "HIJAU" if hijau else "MERAH"
print(f"\n  === VERDICT: {verdict} ===")
print(f"  - akurasi disagreement test model-3 ({winner}) = {acc_dis_test_winner:.1%}  (syarat: >60%)")
print(f"  - cascade test F1 = {m_casc_te['f1']:.4f}  (syarat: > B2={B2_REF})")
print(f"  - val-test gap cascade = {gap:+.4f}  (syarat: |gap| < {GAP_THRESHOLD})")
if hijau:
    print("  -> LANJUT: bangun cascade penuh + varian confidence-weighted di iterasi berikutnya.")
else:
    print("  -> STOP: tulis sebagai negative result — oracle TIDAK tertembus oleh routing sinyal yang tersedia.")

OUT["stage4_cascade_outcome"] = {
    "winner_model3": winner, "cascade_val": m_casc_va, "cascade_test": m_casc_te,
    "gap_val_test": gap, "ci95_test": [round(lo, 4), round(hi, 4)], "p_beat_sota": round(pgt, 3),
    "vs_B2_paired": vs_b2, "vs_SOTA_paired": vs_sota, "n_disagree_test": n_dis_te,
    "latency_ms": {"b2": round(t_b2 * 1000, 1), "cascade": round(t_cascade * 1000, 1),
                   "sota_alone": round(t_sota * 1000, 1) if not np.isnan(t_sota) else None},
    "decision_thresholds": {"min_acc_disagree": 0.60, "min_f1": B2_REF, "max_gap": GAP_THRESHOLD},
    "verdict": verdict}
OUT["verdict"] = verdict

# ============================================================================
# FIGURES (print-friendly, IEEE single-column) — palet CVD-safe tervalidasi
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "figure.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.6,
                     "xtick.color": "#555555", "ytick.color": "#555555",
                     "axes.labelcolor": "#222222", "text.color": "#222222"})
C_BLUE, C_VERM, C_GRAY, C_INK = "#0072B2", "#D55E00", "#999999", "#222222"
SM = " [SMOKE — data sintetis]" if SMOKE else ""

# --- Fig 1: ceiling oracle vs hasil aktual ---
labels = ["LR\nalone", "SOTA\nalone", "B2 hybrid\n(actual)", "Oracle\nLR+xb", "Oracle\nLR+xl", "Grand\noracle"]
vals = [m_lr["f1"], report(yte, P_test["xlmr_large"], 0.5)["f1"], m_b2["f1"],
        stage1["pairwise"]["xlmr_base"]["oracle_f1"], stage1["pairwise"]["xlmr_large"]["oracle_f1"], grand_f1]
is_oracle = [False, False, False, True, True, True]
fig, ax = plt.subplots(figsize=(4.3, 2.8))
x = np.arange(len(labels))
bars = ax.bar(x, vals, color=[C_VERM if o else C_BLUE for o in is_oracle],
              width=0.62, edgecolor="white", linewidth=0.6)
for bar, o in zip(bars, is_oracle):
    if o: bar.set_hatch("///"); bar.set_alpha(0.55)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("F1 (test)"); ax.set_ylim(0, 1.12)
ax.axhline(B2_REF, color=C_GRAY, lw=0.8, ls="--", zorder=0)
ax.set_title(f"Ceiling oracle vs hasil aktual (test){SM}", pad=10)
ax.legend(handles=[Patch(facecolor=C_BLUE, label="Hasil aktual (dideploy)"),
                   Patch(facecolor=C_VERM, alpha=0.55, hatch="///", label="Oracle (ceiling, TIDAK dideploy)")],
          loc="upper left", frameon=False, fontsize=6.3)
fig.savefig("fig_oracle_ceiling.png"); plt.close(fig)

# --- Fig 2: kandidat model-3 — dekorelasi x akurasi disagreement ---
fig, ax = plt.subplots(figsize=(3.5, 3.1))
xs = [stage3[m]["corr_test"] for m in MODEL3_CANDIDATES]
ys = [stage3[m]["acc_disagree_test"] for m in MODEL3_CANDIDATES]
ax.scatter(xs, ys, s=44, c=C_BLUE, edgecolor="white", linewidth=0.6, zorder=3)
for m, xx, yy in zip(MODEL3_CANDIDATES, xs, ys):
    mk = " (terpilih)" if m == winner else ""
    ax.annotate(m + mk, (xx, yy), textcoords="offset points", xytext=(4, 4), fontsize=6.3)
ax.axhline(0.60, color=C_VERM, lw=0.8, ls="--", label="ambang keputusan 60%")
if not np.isnan(base_side):
    ax.axhline(base_side, color=C_GRAY, lw=0.8, ls=":", label=f"baseline sisi mayoritas ({base_side:.0%})")
ax.set_xlabel("korelasi prob dgn xlmr_base (test) — kiri = lebih terdekorelasi")
ax.set_ylabel("akurasi pada kasus disagreement (test)")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower left", frameon=False, fontsize=6.3)
ax.set_title(f"Kandidat model-3: dekorelasi × akurasi{SM}")
fig.savefig("fig_model3_diagnostic.png"); plt.close(fig)
print("\nfigures  -> fig_oracle_ceiling.png, fig_model3_diagnostic.png")

# ============================================================================
# SIMPAN
# ============================================================================
print(f"\n  referensi: SOTA={SOTA}  B2={B2_REF}  LR={LR_REF}  |  VERDICT AKHIR: {verdict}")
if SMOKE:
    print("  !! SMOKE MODE — semua angka transformer sintetis, BUKAN hasil riset !!")

with open("oracle_diagnostic_results_twitter.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
print("saved -> oracle_diagnostic_results_twitter.json")
