# ============================================================================
# COMPLEMENTARITY + GATED FUSION — LR + XLM-R (Twitter)
#
# Tahap 0: sanity — LR(tuned)=0.7509, xlmr_base@0.5=0.7386, xlmr_large@0.5=0.7692,
#          B2 (LR+xlmr_base prob-avg)=0.7900. Meleset -> STOP, cek versi.
# Tahap A: analisis komplementaritas LR<->transformer (korelasi prob, phi,
#          disagreement, oracle F1, error overlap, contoh kualitatif) + uji
#          berpasangan (paired bootstrap dF1, McNemar) vs SOTA.
# Tahap B: confidence-gated fusion — G1 hard gate / G2 soft gate / G3
#          disagreement gate (maks 2 parameter per varian, semua dari val).
# Tahap C: kNN retrieval fusion — CLS embedding xlmr_base, sinyal instance-based;
#          cek dekorelasi dulu, lalu 3-way prob-avg.
# Tahap D: MC-dropout uncertainty gate — K=10 stochastic pass, std prediktif
#          sebagai sinyal gate (pengganti confidence mentah yang over-confident).
#
# PROTOKOL: semua parameter (w, tau, a, b, c, k, threshold) dipilih dari VAL saja.
# Setiap baris test dilaporkan apa adanya + val-test gap (indikator overfit).
# Tidak ada seleksi metode berdasarkan test.
#
# KAGGLE: Settings -> GPU T4 x1, Internet = ON. Pin versi spt baseline:
#   !pip install -q transformers==4.46.3 datasets==3.1.0 evaluate==0.4.3 \
#                   accelerate==1.1.1 scikit-learn nltk sentencepiece
#
# SMOKE TEST LOKAL (tanpa GPU/torch): set env SMOKE=1 -> prob transformer &
# embedding diganti sinyal SINTETIS agar seluruh jalur kode teruji.
# Semua angka mode smoke BUKAN hasil riset.
#   PowerShell: $env:SMOKE="1"; python kaggle_complementarity_gated_twitter.py
#
# Output: complementarity_gated_results_twitter.json
#         fig_error_overlap.png / fig_complementarity_scatter.png / fig_gate_behavior.png
#         -> salin ke results/complementarity/ di repo.
# ============================================================================
import os, sys, warnings, json, time
import numpy as np
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

SMOKE = os.environ.get("SMOKE", "0") == "1"
RUN_C = True    # Tahap C: kNN retrieval fusion
RUN_D = True    # Tahap D: MC-dropout uncertainty gate

DATASET  = "w11wo/twitter_indonesia_sarcastic"
TEXT_COL = "tweet"
SEED, MAX_LEN, BATCH = 41, 128, 32
SOTA   = 0.7692   # XLM-R large @0.5 (paper)
B2_REF = 0.7900   # LR + xlmr_base prob-avg (kaggle_stacking_full_twitter.py)
LR_REF = 0.7509   # LR (tuned), val-tuned threshold (protokol bersih)
CLASSICAL_CORR_REF = 0.877   # mean OOF pairwise corr antar model klasik (studi ensembling)
N_BOOT = 5000
MC_K   = 10
np.random.seed(SEED)

MODELS = {
    "xlmr_base":  "w11wo/xlm-roberta-base-twitter-indonesia-sarcastic",
    "xlmr_large": "w11wo/xlm-roberta-large-twitter-indonesia-sarcastic",
}
ALONE_F1 = {"xlmr_base": 0.7386, "xlmr_large": 0.7692}   # paper @0.5

# ---------------- data ----------------
from datasets import load_dataset
ds = load_dataset(DATASET)
tr_t, va_t, te_t = list(ds["train"][TEXT_COL]), list(ds["validation"][TEXT_COL]), list(ds["test"][TEXT_COL])
ytr = np.array(ds["train"]["label"]); yva = np.array(ds["validation"]["label"]); yte = np.array(ds["test"]["label"])
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} (pos {yte.sum()}){'  [SMOKE MODE]' if SMOKE else ''}")

# ---------------- helpers ----------------
from sklearn.metrics import (f1_score, precision_recall_fscore_support, accuracy_score,
                             matthews_corrcoef, roc_auc_score)

def best_threshold(y, s):
    """F1-maximizing threshold pada skor val — vektorisasi penuh (identik dgn loop
    np.unique di kaggle_stacking_full_twitter.py, hanya lebih cepat untuk grid search)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    ss, ys = s[order], y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); pos = max(ys.sum(), 1)
    valid = np.r_[ss[1:] != ss[:-1], True]          # cut hanya di batas nilai unik (ties utuh)
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / pos
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    f1 = np.where(valid, f1, -1.0)
    i = int(np.argmax(f1))
    return float(ss[i]), float(f1[i])

def best_threshold_slow(y, s):
    """Versi loop asli — dipakai hanya untuk self-test kesetaraan di mode SMOKE."""
    bt, bf = 0.5, -1.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)

def report(y, s, thr):
    p = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"f1": round(f1, 4), "prec": round(pr, 4), "rec": round(rc, 4), "acc": round(accuracy_score(y, p), 4)}

def bootstrap(y, pred, n=N_BOOT):
    """CI F1 + P(F1 > SOTA) — bootstrap independen thd angka SOTA (kompatibel script lama)."""
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(f1_score(y[b], pred[b], zero_division=0))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out > SOTA).mean())

def paired_bootstrap(y, pred_a, pred_b, n=N_BOOT):
    """Paired bootstrap dF1 = F1_a - F1_b pada resample test yang SAMA."""
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        d.append(f1_score(y[b], pred_a[b], zero_division=0) -
                 f1_score(y[b], pred_b[b], zero_division=0))
    d = np.array(d)
    return {"d_f1_mean": round(float(d.mean()), 4),
            "d_f1_ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_a_beats_b": round(float((d > 0).mean()), 3)}

def mcnemar_exact(y, pred_a, pred_b):
    """McNemar exact (binomial) pada pasangan diskordan."""
    ca, cb = (pred_a == y), (pred_b == y)
    n_a_only = int((ca & ~cb).sum()); n_b_only = int((~ca & cb).sum())
    n = n_a_only + n_b_only
    try:
        from scipy.stats import binomtest
        p = binomtest(min(n_a_only, n_b_only), n, 0.5).pvalue if n else 1.0
    except ImportError:
        from scipy.stats import binom
        k = min(n_a_only, n_b_only)
        p = min(1.0, 2 * binom.cdf(k, n, 0.5)) if n else 1.0
    return {"n_a_only_correct": n_a_only, "n_b_only_correct": n_b_only, "p_value": round(float(p), 4)}

if SMOKE:   # self-test: best_threshold vektorisasi == versi loop asli
    rng0 = np.random.default_rng(0)
    for _ in range(20):
        ys_, ss_ = rng0.integers(0, 2, 97), np.round(rng0.random(97), 3)
        t_fast, f_fast = best_threshold(ys_, ss_); t_slow, f_slow = best_threshold_slow(ys_, ss_)
        assert abs(f_fast - f_slow) < 1e-9, f"best_threshold mismatch: {f_fast} vs {f_slow}"
    print("[SMOKE] self-test best_threshold: OK (identik dgn versi loop)")

# ---------------- 1) classical LR (tuned) — identik kaggle_stacking_full_twitter.py ----------------
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

# ---------------- 2) transformer probs + CLS embeddings (inference only) ----------------
INFER_TIMES, E_bank = {}, {}

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
    def transformer_forward(model_name, texts, want_emb=False, time_key=None):
        """Probs kelas sarkasme (+ CLS embedding L2-norm jika want_emb)."""
        cfg = AutoConfig.from_pretrained(model_name)
        if getattr(cfg, "model_type", "") in ("xlm-roberta", "roberta"):
            cfg._attn_implementation = "eager"        # reproduksi paper persis (SDPA != eager)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(DEVICE).eval()
        pos, probs, embs = _pos(model.config), [], []
        if time_key:
            enc0 = tok(texts[:BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            model(**enc0)                              # warmup CUDA kernels
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            out = model(**enc, output_hidden_states=want_emb)
            probs.append(torch.softmax(out.logits, dim=-1)[:, pos].cpu().numpy())
            if want_emb:
                embs.append(out.hidden_states[-1][:, 0, :].cpu().numpy())   # token <s> (CLS)
        if time_key:
            if DEVICE == "cuda": torch.cuda.synchronize()
            INFER_TIMES[time_key] = time.perf_counter() - t0
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
        p = np.concatenate(probs)
        if want_emb:
            e = np.concatenate(embs)
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
            return p, e
        return p

    def mc_dropout_probs(model_name, texts, K=MC_K):
        """K forward pass dgn dropout AKTIF (Gal & Ghahramani 2016) -> mean & std prediktif.
        Checkpoint sama, tanpa retraining; hanya mode dropout yang dinyalakan."""
        cfg = AutoConfig.from_pretrained(model_name)
        if getattr(cfg, "model_type", "") in ("xlm-roberta", "roberta"):
            cfg._attn_implementation = "eager"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(DEVICE)
        model.train()                                  # dropout ON (tidak ada batchnorm di XLM-R)
        pos, runs = _pos(model.config), []
        with torch.no_grad():
            for kpass in range(K):
                torch.manual_seed(SEED * 1000 + kpass)
                if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED * 1000 + kpass)
                probs = []
                for i in range(0, len(texts), BATCH):
                    enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN,
                              return_tensors="pt").to(DEVICE)
                    probs.append(torch.softmax(model(**enc).logits, dim=-1)[:, pos].cpu().numpy())
                runs.append(np.concatenate(probs))
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
        runs = np.stack(runs)                          # (K, n)
        return runs.mean(axis=0), runs.std(axis=0)

    for key, name in MODELS.items():
        print(f"[infer] {key} ...")
        want = (key == "xlmr_base")                    # embedding hanya dari anchor (utk kNN)
        if want:
            P_val[key],  E_bank["val"]  = transformer_forward(name, va_t, want_emb=True)
            P_test[key], E_bank["test"] = transformer_forward(name, te_t, want_emb=True, time_key=key)
            _,           E_bank["train"] = transformer_forward(name, tr_t, want_emb=True)
        else:
            P_val[key]  = transformer_forward(name, va_t)
            P_test[key] = transformer_forward(name, te_t, time_key=key)
else:
    # ---- sinyal sintetis: HANYA untuk menguji jalur kode secara lokal ----
    def _synth(y, strength, seed):
        rng = np.random.default_rng(seed)
        return np.clip(strength * y + (1 - strength) * (1 - y) * 0.4 + rng.normal(0, 0.28, len(y)), 0.01, 0.99)
    for i, key in enumerate(MODELS):
        P_val[key]  = _synth(yva, 0.62 + 0.04 * i, 100 + i)
        P_test[key] = _synth(yte, 0.62 + 0.04 * i, 200 + i)
        INFER_TIMES[key] = float("nan")
    rng_e = np.random.default_rng(7); DIM = 64
    for split, yy in [("train", ytr), ("val", yva), ("test", yte)]:
        e = rng_e.normal(size=(len(yy), DIM)); e[:, 0] += 1.2 * yy; e[:, 1] += 0.6 * yy
        E_bank[split] = e / np.linalg.norm(e, axis=1, keepdims=True)

# ============================================================================
# TAHAP 0 — SANITY CHECK (wajib lulus sebelum hasil dipercaya)
# ============================================================================
print("\n=== TAHAP 0: SANITY CHECK ===")
lr_thr, lr_val_f1 = best_threshold(yva, P_val["lr"])
m_lr = report(yte, P_test["lr"], lr_thr)

def wavg_fit(pv_a, pv_b, pt_a, pt_b):
    """Static prob-avg: w*a + (1-w)*b, w & threshold dari val. Return (test_scores, thr, val_f1, w)."""
    best = (-1.0, 0.5, 0.5)
    for w in np.linspace(0, 1, 41):
        thr, f1 = best_threshold(yva, w * pv_a + (1 - w) * pv_b)
        if f1 > best[0]: best = (f1, w, thr)
    vf, w, thr = best
    return w * pt_a + (1 - w) * pt_b, thr, vf, w

b2_scores, b2_thr, b2_val_f1, b2_w = wavg_fit(P_val["lr"], P_val["xlmr_base"], P_test["lr"], P_test["xlmr_base"])
m_b2 = report(yte, b2_scores, b2_thr)
pred_b2 = (b2_scores >= b2_thr).astype(int)

sanity_rows = [("LR (tuned, val-thr)", m_lr["f1"], LR_REF),
               ("xlmr_base @0.5",      report(yte, P_test["xlmr_base"], 0.5)["f1"],  ALONE_F1["xlmr_base"]),
               ("xlmr_large @0.5",     report(yte, P_test["xlmr_large"], 0.5)["f1"], ALONE_F1["xlmr_large"]),
               ("B2 LR+xlmr_base wavg", m_b2["f1"], B2_REF)]
sanity_ok = True
for name, got, ref in sanity_rows:
    ok = abs(got - ref) < 0.002
    if SMOKE and "LR" not in name: status = "SKIP (smoke)"
    else:
        status = "OK" if ok else "CEK!"
        if not ok: sanity_ok = False
    print(f"  {name:24s} F1={got:.4f}  ref={ref:.4f}  [{status}]")
if not sanity_ok and not SMOKE:
    print("\n  !! SANITY GAGAL — hentikan analisis, cek versi transformers/attention (eager) !!")
    sys.exit(1)

pred_lr   = (P_test["lr"] >= lr_thr).astype(int)
pred_tf   = {k: (P_test[k] >= 0.5).astype(int) for k in MODELS}   # protokol paper @0.5
pred_sota = pred_tf["xlmr_large"]

OUT = {"_meta": {"dataset": DATASET, "seed": SEED, "n_boot": N_BOOT, "smoke": SMOKE,
                 "refs": {"SOTA_xlmr_large": SOTA, "B2_wavg": B2_REF, "LR_tuned": LR_REF,
                          "classical_pairwise_corr": CLASSICAL_CORR_REF},
                 "b2": {"w_lr": round(b2_w, 3), "thr": round(b2_thr, 4),
                        "val_f1": round(b2_val_f1, 4), **m_b2}},
       "stage0_sanity": {n: {"f1": g, "ref": r} for n, g, r in sanity_rows}}

# ============================================================================
# TAHAP A — ANALISIS KOMPLEMENTARITAS + UJI BERPASANGAN
# ============================================================================
print("\n=== TAHAP A: KOMPLEMENTARITAS LR <-> TRANSFORMER (test set) ===")
print(f"  referensi: mean pairwise corr antar model KLASIK = {CLASSICAL_CORR_REF} (studi ensembling)\n")

stageA = {"prob_pearson": {}, "pred_phi": {}, "disagreement": {}, "oracle": {}, "error_overlap": {}}
pairs = [("lr", "xlmr_base"), ("lr", "xlmr_large"), ("xlmr_base", "xlmr_large")]
print(f"  {'pasangan':28s} {'r_prob':>7} {'phi_pred':>8} {'disagree':>9} {'A benar':>8} {'B benar':>8} {'oracleF1':>9}")
print("  " + "-" * 84)
for a, b in pairs:
    pa = pred_lr if a == "lr" else pred_tf[a]
    pb = pred_tf[b]
    r_prob = float(np.corrcoef(P_test[a], P_test[b])[0, 1])
    phi    = float(matthews_corrcoef(pa, pb))
    dis    = pa != pb
    a_win  = int(((pa == yte) & dis).sum()); b_win = int(((pb == yte) & dis).sum())
    oracle_pred = np.where(dis, yte, pa)               # disagree (biner) -> salah satu pasti benar
    o_f1 = f1_score(yte, oracle_pred, zero_division=0)
    key = f"{a}|{b}"
    stageA["prob_pearson"][key] = round(r_prob, 4)
    stageA["pred_phi"][key]     = round(phi, 4)
    stageA["disagreement"][key] = {"n": int(dis.sum()), "rate": round(float(dis.mean()), 4),
                                   "a_correct": a_win, "b_correct": b_win}
    stageA["oracle"][key]       = round(float(o_f1), 4)
    print(f"  {a+' vs '+b:28s} {r_prob:7.3f} {phi:8.3f} {int(dis.sum()):6d} ({dis.mean():4.1%})"
          f" {a_win:8d} {b_win:8d} {o_f1:9.4f}")

# error overlap 2x2 (utk tabel/figure paper)
for k in MODELS:
    e_lr, e_tf = (pred_lr != yte), (pred_tf[k] != yte)
    ov = {"both_correct": int((~e_lr & ~e_tf).sum()), "only_lr_wrong": int((e_lr & ~e_tf).sum()),
          "only_tf_wrong": int((~e_lr & e_tf).sum()), "both_wrong": int((e_lr & e_tf).sum())}
    stageA["error_overlap"][k] = ov
    print(f"\n  error overlap LR x {k}: both-OK={ov['both_correct']}  hanya-LR-salah={ov['only_lr_wrong']}"
          f"  hanya-TF-salah={ov['only_tf_wrong']}  both-salah={ov['both_wrong']}"
          f"  (salah satu benar utk {ov['only_lr_wrong']+ov['only_tf_wrong']} sampel -> ruang fusi)")

# ---- uji berpasangan vs SOTA (prediksi XLM-R large @0.5 di test yang sama) ----
print("\n  --- Uji berpasangan (test sama, n=538) ---")
stageA["paired_tests"] = {}
for name, pa in [("B2_hybrid_vs_SOTA", pred_b2), ("LR_tuned_vs_SOTA", pred_lr)]:
    pb_res = paired_bootstrap(yte, pa, pred_sota)
    mc     = mcnemar_exact(yte, pa, pred_sota)
    stageA["paired_tests"][name] = {"paired_bootstrap": pb_res, "mcnemar": mc}
    print(f"  {name:22s} dF1={pb_res['d_f1_mean']:+.4f} CI95={pb_res['d_f1_ci95']}"
          f"  P(a>b)={pb_res['p_a_beats_b']:.1%}  McNemar p={mc['p_value']:.4f}"
          f"  (a-only={mc['n_a_only_correct']}, b-only={mc['n_b_only_correct']})")

# ---- contoh kualitatif: LR mengoreksi TF (dan sebaliknya) — utk tabel diskusi ----
feat_names = np.array(_LR_VEC.get_feature_names_out())
def top_lr_feats(i, n=4):
    contrib = Xte[i].multiply(_LR_FULL.coef_[0]).toarray().ravel()
    idx = np.argsort(-np.abs(contrib))[:n]
    return [(feat_names[j], round(float(contrib[j]), 3)) for j in idx if contrib[j] != 0]

def qualitative(mask, sort_key, label, n=5):
    idx = np.where(mask)[0]
    idx = idx[np.argsort(-sort_key[idx])][:n]
    rows = []
    print(f"\n  --- {label} (top {len(idx)}) ---")
    for i in idx:
        txt = te_t[i].replace("\n", " ")[:90]
        feats = top_lr_feats(i)
        rows.append({"idx": int(i), "y": int(yte[i]), "p_lr": round(float(P_test['lr'][i]), 3),
                     "p_tf": round(float(P_test['xlmr_base'][i]), 3), "text": txt, "top_lr_feats": feats})
        print(f"  [{i:3d}] y={yte[i]}  p_lr={P_test['lr'][i]:.3f}  p_tf={P_test['xlmr_base'][i]:.3f}  {txt}")
        print(f"        LR feats: {feats}")
    return rows

e_lr, e_tf = (pred_lr != yte), (pred_tf["xlmr_base"] != yte)
conf_tf = np.abs(P_test["xlmr_base"] - 0.5)
conf_lr = np.abs(P_test["lr"] - lr_thr)
stageA["qualitative"] = {
    "lr_corrects_tf": qualitative(~e_lr & e_tf, conf_tf, "LR benar, XLM-R base salah-yakin"),
    "tf_corrects_lr": qualitative(e_lr & ~e_tf, conf_lr, "XLM-R base benar, LR salah-yakin")}
OUT["stageA_complementarity"] = stageA

# ============================================================================
# TAHAP B — CONFIDENCE-GATED FUSION (parameter dari val; test disentuh sekali/varian)
# ============================================================================
print("\n=== TAHAP B: GATED FUSION ===")
print(f"  null hypothesis: B2 static wavg (val={b2_val_f1:.4f}, test={m_b2['f1']:.4f})\n")

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _logit(p):   return np.clip(np.log(p / (1 - p)), -8, 8)

def gate_g1(p_lr, p_tf, tau, w):     # hard gate: TF yakin -> TF saja; ragu -> wavg
    conf = np.maximum(p_tf, 1 - p_tf)
    return np.where(conf >= tau, p_tf, w * p_lr + (1 - w) * p_tf)

def gate_g2(p_lr, p_tf, a, b):       # soft gate: w_tf(x) = sigmoid(a + b*|logit_tf|)
    w_tf = _sigmoid(a + b * np.abs(_logit(p_tf)))
    return w_tf * p_tf + (1 - w_tf) * p_lr

def gate_g3(p_lr, p_tf, c0, c1):     # disagreement gate: w_lr(x) = clip(c0 + c1*|p_lr-p_tf|)
    w_lr = np.clip(c0 + c1 * np.abs(p_lr - p_tf), 0, 1)
    return w_lr * p_lr + (1 - w_lr) * p_tf

GATE_GRIDS = {
    "G1_hard":     (gate_g1, [(t, w) for t in np.r_[np.arange(0.50, 1.001, 0.025), 1.01]
                                     for w in np.linspace(0, 1, 21)]),
    "G2_soft":     (gate_g2, [(a, b) for a in np.arange(-2, 4.01, 0.25)
                                     for b in np.arange(-2, 2.01, 0.25)]),
    "G3_disagree": (gate_g3, [(c0, c1) for c0 in np.arange(0, 1.001, 0.05)
                                       for c1 in np.arange(-2, 2.001, 0.2)]),
}

def eval_row(name, scores_te, thr, val_f1, params=None):
    """Evaluasi satu metode di test + CI + paired vs B2. Semua seleksi sudah terjadi di val."""
    m = report(yte, scores_te, thr)
    p = (scores_te >= thr).astype(int)
    lo, hi, pgt = bootstrap(yte, p)
    vs_b2 = paired_bootstrap(yte, p, pred_b2)
    row = {"params": params, "val_f1": round(val_f1, 4), **m,
           "gap_val_test": round(val_f1 - m["f1"], 4),
           "ci95": [round(lo, 4), round(hi, 4)], "p_beat_sota": round(pgt, 3),
           "vs_B2_paired": vs_b2}
    print(f"  {name:26s} val={val_f1:.4f} test={m['f1']:.4f} gap={val_f1-m['f1']:+.4f}"
          f"  CI=[{lo:.3f},{hi:.3f}]  P(>SOTA)={pgt:5.1%}  P(>B2)={vs_b2['p_a_beats_b']:5.1%}"
          f"  {params if params else ''}")
    return row, p

stageB, best_gate_info = {}, {}
for k in MODELS:
    print(f"  --- anchor: {k} ---")
    stageB[k] = {}
    # baseline keluarga: static wavg (kasus khusus semua gate)
    sc, thr, vf, w = wavg_fit(P_val["lr"], P_val[k], P_test["lr"], P_test[k])
    stageB[k]["wavg_static"], _ = eval_row(f"wavg_static ({k})", sc, thr, vf, {"w_lr": round(float(w), 3)})
    for gname, (fn, grid) in GATE_GRIDS.items():
        best = (-1.0, None, 0.5)
        for prm in grid:
            thr_g, f1_g = best_threshold(yva, fn(P_val["lr"], P_val[k], *prm))
            if f1_g > best[0]: best = (f1_g, prm, thr_g)
        vf_g, prm, thr_g = best
        sc_te = fn(P_test["lr"], P_test[k], *prm)
        row, pred_g = eval_row(f"{gname} ({k})", sc_te, thr_g, vf_g,
                               {p: round(float(v), 3) for p, v in zip(("p1", "p2"), prm)})
        stageB[k][gname] = row
        if k == "xlmr_base" and (not best_gate_info or vf_g > best_gate_info["val_f1"]):
            best_gate_info = {"name": gname, "fn": fn, "params": prm, "val_f1": vf_g}
    print()
OUT["stageB_gated"] = stageB

# ============================================================================
# TAHAP C — kNN RETRIEVAL FUSION (embedding xlmr_base, sinyal instance-based)
# ============================================================================
if RUN_C:
    print("=== TAHAP C: kNN RETRIEVAL FUSION ===")
    def knn_probs(E_query, k):
        sims = E_query @ E_bank["train"].T                     # cosine (sudah L2-norm)
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        return ytr[top].mean(axis=1)

    t0 = time.perf_counter()
    best_k, best_kf1 = None, -1.0
    for k_try in (5, 10, 25, 50):
        _, f1_k = best_threshold(yva, knn_probs(E_bank["val"], k_try))
        if f1_k > best_kf1: best_k, best_kf1 = k_try, f1_k
    P_val["knn"]  = knn_probs(E_bank["val"], best_k)
    P_test["knn"] = knn_probs(E_bank["test"], best_k)
    t_knn = time.perf_counter() - t0
    thr_knn, vf_knn = best_threshold(yva, P_val["knn"])
    m_knn = report(yte, P_test["knn"], thr_knn)
    print(f"  kNN alone: k={best_k}  val={vf_knn:.4f}  test={m_knn['f1']:.4f}")

    # cek dekorelasi (gerbang keputusan): sinyal baru harus < korelasi klasik 0.877
    r_knn_lr = float(np.corrcoef(P_test["knn"], P_test["lr"])[0, 1])
    r_knn_tf = float(np.corrcoef(P_test["knn"], P_test["xlmr_base"])[0, 1])
    decor_ok = r_knn_tf < CLASSICAL_CORR_REF
    print(f"  korelasi prob: kNN<->LR r={r_knn_lr:.3f} | kNN<->xlmr_base r={r_knn_tf:.3f}"
          f"  -> {'terdekorelasi, lanjut 3-way' if decor_ok else 'TERLALU MIRIP transformer (share embedding) — negative result'}")

    stageC = {"best_k": best_k, "alone": {"val_f1": round(vf_knn, 4), **m_knn},
              "corr": {"knn_lr": round(r_knn_lr, 4), "knn_xlmr_base": round(r_knn_tf, 4)},
              "decorrelated": bool(decor_ok), "knn_search_ms_valtest": round(t_knn * 1000, 1)}

    # 3-way prob-avg: bobot simplex (2 parameter bebas) dari val — tetap dilaporkan
    # apa adanya meski dekorelasi lemah (baris ablasi utk paper).
    best3 = (-1.0, None, 0.5)
    for w_lr in np.arange(0, 1.001, 0.05):
        for w_kn in np.arange(0, 1.001 - w_lr + 1e-9, 0.05):
            w_tf = 1 - w_lr - w_kn
            s = w_lr * P_val["lr"] + w_kn * P_val["knn"] + w_tf * P_val["xlmr_base"]
            thr3, f13 = best_threshold(yva, s)
            if f13 > best3[0]: best3 = (f13, (w_lr, w_kn, w_tf), thr3)
    vf3, (w_lr, w_kn, w_tf), thr3 = best3
    sc3 = w_lr * P_test["lr"] + w_kn * P_test["knn"] + w_tf * P_test["xlmr_base"]
    stageC["threeway_wavg"], _ = eval_row("3way LR+kNN+xlmr_base", sc3, thr3, vf3,
                                          {"w_lr": round(float(w_lr), 2), "w_knn": round(float(w_kn), 2),
                                           "w_tf": round(float(w_tf), 2)})
    OUT["stageC_knn"] = stageC
    print()

# ============================================================================
# TAHAP D — MC-DROPOUT UNCERTAINTY GATE (upgrade sinyal gate Tahap B)
# ============================================================================
if RUN_D:
    print("=== TAHAP D: MC-DROPOUT UNCERTAINTY GATE (xlmr_base, K=%d) ===" % MC_K)
    if not SMOKE:
        mc_mean_va, mc_std_va = mc_dropout_probs(MODELS["xlmr_base"], va_t)
        mc_mean_te, mc_std_te = mc_dropout_probs(MODELS["xlmr_base"], te_t)
    else:
        rng_d = np.random.default_rng(11)
        err_va = ((P_val["xlmr_base"] >= 0.5).astype(int) != yva)
        err_te = ((P_test["xlmr_base"] >= 0.5).astype(int) != yte)
        mc_mean_va, mc_std_va = P_val["xlmr_base"],  0.05 + 0.2 * rng_d.random(len(yva)) + 0.1 * err_va
        mc_mean_te, mc_std_te = P_test["xlmr_base"], 0.05 + 0.2 * rng_d.random(len(yte)) + 0.1 * err_te

    # diagnostik: sinyal mana yang lebih baik memprediksi ERROR transformer?
    err_va = ((P_val["xlmr_base"] >= 0.5).astype(int) != yva).astype(int)
    err_te = ((P_test["xlmr_base"] >= 0.5).astype(int) != yte).astype(int)
    auroc = {}
    for split, err, std_s, p_s in [("val", err_va, mc_std_va, P_val["xlmr_base"]),
                                   ("test", err_te, mc_std_te, P_test["xlmr_base"])]:
        auroc[split] = {"mc_std": round(float(roc_auc_score(err, std_s)), 4),
                        "raw_unconf": round(float(roc_auc_score(err, -np.abs(p_s - 0.5))), 4)}
        print(f"  AUROC(sinyal->error TF) {split}: mc_std={auroc[split]['mc_std']:.4f}"
              f"  vs raw-unconfidence={auroc[split]['raw_unconf']:.4f}")

    # MC-mean sendiri (variance-reduced) sebagai baris referensi
    thr_mc, vf_mc = best_threshold(yva, mc_mean_va)
    stageD = {"K": MC_K, "auroc_error_signal": auroc}
    stageD["mc_mean_alone"], _ = eval_row("MC-mean alone", mc_mean_te, thr_mc, vf_mc)

    # gate: w_tf(x) = sigmoid(a + b*z_sigma), z dari statistik VAL (anti-leakage)
    mu_s, sd_s = float(mc_std_va.mean()), float(mc_std_va.std() + 1e-9)
    z_va, z_te = (mc_std_va - mu_s) / sd_s, (mc_std_te - mu_s) / sd_s
    best = (-1.0, None, 0.5)
    for a in np.arange(-2, 4.01, 0.25):
        for b in np.arange(-3, 3.01, 0.25):
            w_tf = _sigmoid(a + b * z_va)
            thr_u, f1_u = best_threshold(yva, w_tf * P_val["xlmr_base"] + (1 - w_tf) * P_val["lr"])
            if f1_u > best[0]: best = (f1_u, (a, b), thr_u)
    vf_u, (a_u, b_u), thr_u = best
    w_tf_te = _sigmoid(a_u + b_u * z_te)
    sc_u = w_tf_te * P_test["xlmr_base"] + (1 - w_tf_te) * P_test["lr"]
    stageD["uncertainty_gate"], _ = eval_row("MC-std gate (LR+xb)", sc_u, thr_u, vf_u,
                                             {"a": round(float(a_u), 2), "b": round(float(b_u), 2)})
    OUT["stageD_mcdropout"] = stageD
    print()

# ============================================================================
# LATENCY (test n=538) — angka terukur utk menggantikan "perkiraan" di paper
# ============================================================================
print("=== LATENCY (test, n=%d) ===" % len(te_t))
_LR_FULL.predict_proba(_LR_VEC.transform(te_t))       # warmup
t_lr = min((lambda: (lambda t0: (_LR_FULL.predict_proba(_LR_VEC.transform(te_t)), time.perf_counter() - t0)[1])(time.perf_counter()))() for _ in range(3))
lat = {"lr_ms": round(t_lr * 1000, 1)}
print(f"  LR (TF-IDF transform + predict): {t_lr*1000:.1f} ms")
for k, t in INFER_TIMES.items():
    if np.isnan(t): continue
    lat[f"{k}_ms"] = round(t * 1000, 0)
    lat[f"hybrid_lr_{k}_ms"] = round((t + t_lr) * 1000, 0)
    print(f"  {k}: {t*1000:.0f} ms  |  hybrid LR+{k}: {(t+t_lr)*1000:.0f} ms  (overhead LR {t_lr/t*100:.2f}%)")
if RUN_C and "stageC_knn" in OUT:
    lat["knn_search_ms"] = OUT["stageC_knn"]["knn_search_ms_valtest"]
    print(f"  kNN search (val+test, embedding share forward pass TF): {lat['knn_search_ms']:.1f} ms")
if RUN_D:
    print(f"  MC-dropout: {MC_K}x inferensi xlmr_base (analisis; bukan mode deployment)")
OUT["latency_ms"] = lat

# ============================================================================
# FIGURES (print-friendly, IEEE single-column) — palet CVD-safe tervalidasi
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "figure.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.6,
                     "xtick.color": "#555555", "ytick.color": "#555555",
                     "axes.labelcolor": "#222222", "text.color": "#222222"})
C_BLUE, C_VERM, C_GRAY, C_INK = "#0072B2", "#D55E00", "#999999", "#222222"
SM = " [SMOKE — data sintetis]" if SMOKE else ""

# --- Fig 1: error overlap 2x2 (LR x xlmr_base), sequential Blues + label sel ---
ov = stageA["error_overlap"]["xlmr_base"]
# baris = LR benar/salah, kolom = XLM-R benar/salah
grid = np.array([[ov["both_correct"], ov["only_tf_wrong"]],
                 [ov["only_lr_wrong"], ov["both_wrong"]]], dtype=float)
fig, ax = plt.subplots(figsize=(3.1, 2.5))
im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max())
labels = [["Keduanya\nbenar", "Hanya LR\nbenar"], ["Hanya XLM-R\nbenar", "Keduanya\nsalah"]]
for r in range(2):
    for c in range(2):
        frac = grid[r, c] / grid.max()
        col = "white" if frac > 0.6 else C_INK
        ax.text(c, r, f"{labels[r][c]}\n{int(grid[r,c])} ({grid[r,c]/len(yte):.1%})",
                ha="center", va="center", color=col, fontsize=7.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["XLM-R benar", "XLM-R salah"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["LR benar", "LR salah"], rotation=90, va="center")
ax.set_title(f"Error overlap LR × XLM-R base (test, n={len(yte)}){SM}")
ax.spines[:].set_visible(False); ax.tick_params(length=0)
fig.savefig("fig_error_overlap.png"); plt.close(fig)

# --- Fig 2: scatter komplementaritas p_LR vs p_TF, 4 kelas + marker berbeda ---
fig, ax = plt.subplots(figsize=(3.4, 3.2))
x, y = P_test["xlmr_base"], P_test["lr"]
cls = [(~e_lr & ~e_tf, C_GRAY, ".", "Keduanya benar", 8, 0.35),
       (~e_lr & e_tf,  C_BLUE, "^", "Hanya LR benar", 16, 0.9),
       (e_lr & ~e_tf,  C_VERM, "s", "Hanya XLM-R benar", 14, 0.9),
       (e_lr & e_tf,   C_INK,  "x", "Keduanya salah", 14, 0.9)]
for mask, col, mk, lab, sz, al in cls:
    ax.scatter(x[mask], y[mask], c=col, marker=mk, s=sz, alpha=al, label=lab, linewidths=0.8)
ax.axvline(0.5, color="#bbbbbb", lw=0.7, ls="--", zorder=0)
ax.axhline(lr_thr, color="#bbbbbb", lw=0.7, ls="--", zorder=0)
ax.set_xlabel("p(sarkasme) — XLM-R base"); ax.set_ylabel("p(sarkasme) — LR")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
          framealpha=0.85, handletextpad=0.3)
ax.set_title(f"Komplementaritas prediksi (test){SM}")
fig.savefig("fig_complementarity_scatter.png"); plt.close(fig)

# --- Fig 3: perilaku gate terbaik (by val) di anchor — bobot efektif vs sinyal ---
if best_gate_info:
    gname, fn, prm = best_gate_info["name"], best_gate_info["fn"], best_gate_info["params"]
    p_lr_t, p_tf_t = P_test["lr"], P_test["xlmr_base"]
    fused = fn(p_lr_t, p_tf_t, *prm)
    # bobot LR efektif per sampel: fused = w*p_lr + (1-w)*p_tf  ->  w = (fused-p_tf)/(p_lr-p_tf)
    denom = p_lr_t - p_tf_t
    w_eff = np.where(np.abs(denom) > 1e-6, (fused - p_tf_t) / np.where(np.abs(denom) > 1e-6, denom, 1), np.nan)
    sig = np.maximum(p_tf_t, 1 - p_tf_t) if gname == "G1_hard" else (
        np.abs(_logit(p_tf_t)) if gname == "G2_soft" else np.abs(p_lr_t - p_tf_t))
    sig_lab = {"G1_hard": "confidence XLM-R  max(p, 1−p)", "G2_soft": "|logit| XLM-R",
               "G3_disagree": "|p_LR − p_TF|"}[gname]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    o = np.argsort(sig)
    ax.scatter(sig, np.clip(w_eff, 0, 1), s=6, c=C_BLUE, alpha=0.5, linewidths=0)
    ax.axhline(b2_w, color=C_GRAY, lw=1.0, ls="--", label=f"w statis B2 = {b2_w:.2f}")
    ax.set_xlabel(sig_lab); ax.set_ylabel("bobot LR efektif  w(x)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", frameon=False)
    ax.set_title(f"Perilaku gate terbaik by val: {gname}{SM}")
    fig.savefig("fig_gate_behavior.png"); plt.close(fig)
print("\nfigures  -> fig_error_overlap.png, fig_complementarity_scatter.png, fig_gate_behavior.png")

# ============================================================================
# RINGKASAN + SIMPAN
# ============================================================================
print("\n=== RINGKASAN SEMUA METODE (urut val F1 — seleksi jujur) ===")
rows = [("B2 wavg static (ref)", OUT["_meta"]["b2"]["val_f1"], OUT["_meta"]["b2"]["f1"])]
for k in MODELS:
    for g in GATE_GRIDS:
        r = stageB[k][g]; rows.append((f"{g} ({k})", r["val_f1"], r["f1"]))
if RUN_C and "stageC_knn" in OUT:
    r = OUT["stageC_knn"]["threeway_wavg"]; rows.append(("3way LR+kNN+TF", r["val_f1"], r["f1"]))
if RUN_D and "stageD_mcdropout" in OUT:
    r = OUT["stageD_mcdropout"]["uncertainty_gate"]; rows.append(("MC-std gate", r["val_f1"], r["f1"]))
for name, vf, tf_ in sorted(rows, key=lambda r: -r[1]):
    mark = " <- pilihan by-val" if vf == max(r[1] for r in rows) else ""
    print(f"  {name:28s} val={vf:.4f}  test={tf_:.4f}  gap={vf-tf_:+.4f}{mark}")
print(f"\n  referensi: SOTA={SOTA}  B2={B2_REF}  LR={LR_REF}")
if SMOKE:
    print("  !! SMOKE MODE — semua angka transformer/kNN/MC sintetis, BUKAN hasil riset !!")

with open("complementarity_gated_results_twitter.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
print("saved -> complementarity_gated_results_twitter.json")
