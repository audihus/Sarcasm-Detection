"""
run_stacking_classification.py
==============================
Classical stacking/ensemble study for Indonesian sarcasm (Twitter), lexicon-free.

Motivation (honest): a naive classical stack already LOST to single LR in our
earlier test (stack 0.7368 < LR 0.7536) because (1) all base learners shared the
SAME word-TF-IDF features -> highly correlated -> ensemble adds nothing, and (2)
the meta-learner was fit on the tiny 268-sample validation -> overfit.

This script fixes both:
  * REAL representational diversity via 3 views:
      V1 word TF-IDF (1,2)   V2 word TF-IDF (1,3)   V3 SUBWORD TF-IDF (SentencePiece,
      trained unsupervised on train only -> lexicon-free, decorrelated from words).
  * CV-based stacking: out-of-fold meta-features over the full dev set (train+val,
      2146 rows) instead of a 268-row validation; threshold tuned on OOF too.

Strategies compared: E0 single LR | E1 soft-vote | E2 CV-stacking | E3 bagged LR.
Everything selected on CV/OOF; test touched once; paired bootstrap decides whether
any gain over LR is real or noise. Reference bars: LR 0.7536, XLM-R large 0.7692.
"""
import argparse
import json
import os
import tempfile
import warnings

import numpy as np
from datasets import load_dataset, load_from_disk
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import nltk
from nltk.tokenize import word_tokenize

LR_REF = 0.7536   # our single tuned LR (MVSC), val-protocol
SOTA = 0.7692     # XLM-R large


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="w11wo/twitter_indonesia_sarcastic")
    p.add_argument("--text_column_name", default="tweet")
    p.add_argument("--output_folder", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_subword", action="store_true", help="disable the SentencePiece subword view V3")
    p.add_argument("--sp_vocab", type=int, default=2000)
    p.add_argument("--n_boot", type=int, default=5000)
    return p.parse_args()


def ensure_nltk():
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# --------------------------------------------------------------------------
# SentencePiece subword view (lexicon-free; trained unsupervised on TRAIN only)
# --------------------------------------------------------------------------
class SPEncoder(BaseEstimator, TransformerMixin):
    """Fit a SentencePiece BPE model on the fold's train text; transform text
    into a space-joined string of subword pieces (so a whitespace TfidfVectorizer
    downstream consumes them). Per-fold fit inside CV => no leakage."""

    def __init__(self, vocab_size=2000, model_type="bpe", seed=42):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.seed = seed

    def fit(self, X, y=None):
        import sentencepiece as spm
        tmp = tempfile.mkdtemp()
        prefix = os.path.join(tmp, "sp")
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter([str(x) for x in X]),
            model_prefix=prefix, vocab_size=self.vocab_size,
            model_type=self.model_type, character_coverage=1.0,
            hard_vocab_limit=False, bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0,
            minloglevel=2,
        )
        self.sp_ = spm.SentencePieceProcessor(model_file=prefix + ".model")
        return self

    def transform(self, X):
        return [" ".join(self.sp_.encode(str(t), out_type=str)) for t in X]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def best_threshold(y, s):
    bt, bf = 0.5, -1.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return float(bt)


def metrics(y, pred):
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"f1": round(f1, 4), "prec": round(pr, 4), "rec": round(rc, 4),
            "acc": round(accuracy_score(y, pred), 4)}


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    ensure_nltk()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    use_subword = not args.no_subword
    if use_subword:
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            print("[warn] sentencepiece not installed -> running without subword view V3")
            use_subword = False

    out_dir = os.path.join(args.output_folder, "stacking")
    os.makedirs(out_dir, exist_ok=True)

    # ---- data ----
    ds = load_from_disk(args.dataset_name) if os.path.isdir(args.dataset_name) \
        else load_dataset(args.dataset_name)
    col = args.text_column_name
    dev_t = list(ds["train"][col]) + list(ds["validation"][col])
    ydev = np.array(list(ds["train"]["label"]) + list(ds["validation"]["label"]))
    te_t = list(ds["test"][col])
    yte = np.array(ds["test"]["label"])
    print(f"dev {len(ydev)} (pos {int(ydev.sum())}) | test {len(yte)} (pos {int(yte.sum())}) "
          f"| subword={use_subword}")

    skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)

    def word_vec(ngram):
        return TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None,
                               ngram_range=ngram, min_df=2, sublinear_tf=True)

    def LR():
        return LogisticRegression(class_weight="balanced", max_iter=3000, random_state=args.seed)

    # ---- base learners: (view x classifier), genuinely diverse ----
    base = {}
    base["LR_w12"] = Pipeline([("v", word_vec((1, 2))), ("c", LR())])
    base["LR_w13"] = Pipeline([("v", word_vec((1, 3))), ("c", LR())])
    base["SVC_w12"] = Pipeline([("v", word_vec((1, 2))),
                                ("c", CalibratedClassifierCV(
                                    LinearSVC(class_weight="balanced", dual="auto",
                                              random_state=args.seed), cv=3))])
    base["CNB_w12"] = Pipeline([("v", word_vec((1, 2))), ("c", ComplementNB())])
    if use_subword:
        sp_vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        base["LR_sub"] = Pipeline([("sp", SPEncoder(args.sp_vocab, seed=args.seed)),
                                   ("v", clone(sp_vec)), ("c", LR())])
        base["CNB_sub"] = Pipeline([("sp", SPEncoder(args.sp_vocab, seed=args.seed)),
                                    ("v", clone(sp_vec)), ("c", ComplementNB())])

    # ---- OOF + test probabilities for every base learner ----
    oof, test = {}, {}
    for name, pipe in base.items():
        print(f"  [base] {name} ...")
        oof[name] = cross_val_predict(pipe, dev_t, ydev, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
        test[name] = clone(pipe).fit(dev_t, ydev).predict_proba(te_t)[:, 1]

    names = list(base.keys())

    # ---- diversity: correlation of OOF probabilities + disagreement at 0.5 ----
    P = np.column_stack([oof[n] for n in names])
    corr = np.corrcoef(P.T)
    off = corr[~np.eye(len(names), dtype=bool)]
    print("\n=== DIVERSITY (lower = more complementary) ===")
    print("   base learners:", names)
    print(f"   mean pairwise proba-correlation = {off.mean():.3f} (min {off.min():.3f})")

    # ---- strategy predictions (threshold tuned on OOF, applied to test) ----
    def finalize(oof_score, test_score):
        thr = best_threshold(ydev, oof_score)
        return (test_score >= thr).astype(int), thr

    preds, thrs, scores = {}, {}, {}

    # E0 single LR (view V1) — the reference, under the same CV protocol
    preds["E0_LR"], thrs["E0_LR"] = finalize(oof["LR_w12"], test["LR_w12"]); scores["E0_LR"] = test["LR_w12"]

    # E1 soft-vote (equal-weight mean of all base probabilities)
    ov = np.mean([oof[n] for n in names], axis=0)
    tv = np.mean([test[n] for n in names], axis=0)
    preds["E1_vote"], thrs["E1_vote"] = finalize(ov, tv); scores["E1_vote"] = tv

    # E2 CV-stacking: meta-LR on OOF features; meta OOF for honest threshold
    Xmeta = P
    Xmeta_te = np.column_stack([test[n] for n in names])
    meta = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=args.seed)
    meta_oof = cross_val_predict(meta, Xmeta, ydev, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    meta_te = clone(meta).fit(Xmeta, ydev).predict_proba(Xmeta_te)[:, 1]
    preds["E2_stack"], thrs["E2_stack"] = finalize(meta_oof, meta_te); scores["E2_stack"] = meta_te

    # E3 bagged LR (variance reduction on V1)
    bag = Pipeline([("v", word_vec((1, 2))),
                    ("c", BaggingClassifier(LR(), n_estimators=50, random_state=args.seed, n_jobs=1))])
    oof_bag = cross_val_predict(bag, dev_t, ydev, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    test_bag = clone(bag).fit(dev_t, ydev).predict_proba(te_t)[:, 1]
    preds["E3_bag"], thrs["E3_bag"] = finalize(oof_bag, test_bag); scores["E3_bag"] = test_bag

    # ---- report + paired bootstrap vs E0 (LR) ----
    def boot(predA, predB=None):
        idx = np.arange(len(yte)); fa, dif = [], []
        for _ in range(args.n_boot):
            b = rng.choice(idx, len(idx), replace=True)
            Fa = f1_score(yte[b], predA[b], zero_division=0); fa.append(Fa)
            if predB is not None:
                dif.append(Fa - f1_score(yte[b], predB[b], zero_division=0))
        fa = np.array(fa)
        r = {"mean": float(fa.mean()), "lo": float(np.percentile(fa, 2.5)),
             "hi": float(np.percentile(fa, 97.5)), "p_gt_sota": float((fa > SOTA).mean())}
        if predB is not None:
            dif = np.array(dif)
            r["d_mean"] = float(dif.mean()); r["p_beat_LR"] = float((dif > 0).mean())
        return r

    print("\n=== STRATEGIES (test, threshold tuned on OOF) ===")
    results = {}
    for k in ["E0_LR", "E1_vote", "E2_stack", "E3_bag"]:
        m = metrics(yte, preds[k])
        bs = boot(preds[k], None if k == "E0_LR" else preds["E0_LR"])
        results[k] = {**m, "thr": round(thrs[k], 3), "ci": [round(bs["lo"], 4), round(bs["hi"], 4)],
                      "p_gt_sota": round(bs["p_gt_sota"], 3)}
        line = f"  {k:9s} F1={m['f1']:.4f} (P{m['prec']:.3f}/R{m['rec']:.3f}) " \
               f"CI[{bs['lo']:.3f},{bs['hi']:.3f}]"
        if k != "E0_LR":
            results[k]["d_vs_LR"] = round(bs["d_mean"], 4)
            results[k]["p_beat_LR"] = round(bs["p_beat_LR"], 3)
            line += f"  dF1 vs LR={bs['d_mean']:+.4f}  P(>LR)={bs['p_beat_LR']:.0%}"
        print(line)

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n>> BEST: {best} F1={results[best]['f1']:.4f} | LR ref {LR_REF} | SOTA {SOTA}")
    print(f"   diversity mean-corr={off.mean():.3f}  (subword view {'ON' if use_subword else 'OFF'})")

    results["_meta"] = {"diversity_mean_corr": round(float(off.mean()), 3),
                        "base_learners": names, "subword": use_subword,
                        "LR_ref": LR_REF, "SOTA": SOTA}
    with open(os.path.join(out_dir, "eval_results_twitter.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {os.path.join(out_dir, 'eval_results_twitter.json')}")


if __name__ == "__main__":
    main()
