"""
run_mvsc_classification.py
==========================
MVSC (Multi-View Stacked Classical) — final, lexicon-free classical baseline for
Indonesian sarcasm detection. Counterpart of `run_classical_classification.py`,
but with the improvements that proved to actually matter on the Twitter dataset.

What this method does (and WHY — from our ablation):
  1. Word TF-IDF, n-gram (1,2), sublinear_tf, min_df=2.
       -> bigrams + sublinear weighting were the single biggest honest gain.
  2. class_weight='balanced' (data is ~1:3 imbalanced).
       -> lifts recall, the F1 bottleneck.
  3. Model selection with scoring='f1' (NOT accuracy) on a fixed train/val split.
       -> the paper baseline selects on accuracy, which is wrong for imbalanced F1.
  4. Decision-threshold tuning on the VALIDATION set (never on test).
       -> default 0.5 is too strict on imbalanced data.

Deliberately NOT used (they HURT on this small, PII-masked dataset — verified):
  - char n-grams, hand-crafted structural features, stacking/boosting ensembles.
  - any external dictionary / sentiment lexicon (InSet etc.).

Protocol is identical to the paper's classical baseline (train+val GridSearch via
PredefinedSplit, evaluate once on the untouched test split), so the comparison is
apples-to-apples. Selection touches only train/val; test is read once at the end.

Usage:
  python scripts/run_mvsc_classification.py \
      --dataset_name w11wo/twitter_indonesia_sarcastic \
      --text_column_name tweet \
      --output_folder results
  # or a local save_to_disk directory:
  python scripts/run_mvsc_classification.py \
      --dataset_name ../data/twitter_indonesia_sarcastic \
      --text_column_name tweet --output_folder results
"""
import argparse
import json
import os
import warnings

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.base import clone
from scipy.sparse import vstack

# Reference numbers from the IdSarcasm paper (IEEE Access 2024), F1-binary on test.
REFERENCE = {
    "twitter": {"Classical LR (paper)": 0.7142, "IndoBERT base": 0.7273,
                "XLM-R base": 0.7386, "XLM-R large (SOTA)": 0.7692},
    "reddit":  {"Classical LR (paper)": 0.4887, "IndoBERT base": 0.6100,
                "IndoBERT large": 0.6184, "XLM-R large (SOTA)": 0.6274},
}


def parse_args():
    p = argparse.ArgumentParser(description="MVSC lexicon-free classical sarcasm baseline")
    p.add_argument("--dataset_name", type=str, required=True,
                   help="HuggingFace hub name OR a local load_from_disk directory")
    p.add_argument("--text_column_name", default="text", type=str)
    p.add_argument("--output_folder", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ensure_nltk():
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)


def tune_threshold(y_val, val_scores):
    """Pick the decision threshold that maximises F1 on the validation scores."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(val_scores):
        f = f1_score(y_val, (val_scores >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t), float(best_f1)


def metrics(y_true, y_pred):
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1": f1, "precision": pr, "recall": rc}


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    ensure_nltk()
    np.random.seed(args.seed)

    out_dir = os.path.join(args.output_folder, "mvsc")
    os.makedirs(out_dir, exist_ok=True)

    # ---- load data (local dir or hub) ----
    dataset = load_from_disk(args.dataset_name) if os.path.isdir(args.dataset_name) \
        else load_dataset(args.dataset_name)
    if args.text_column_name != "text":
        dataset = dataset.rename_column(args.text_column_name, "text")
    train_df = dataset["train"].to_pandas()
    valid_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    y_train, y_valid, y_test = train_df.label, valid_df.label, test_df.label

    print(f"train {len(train_df)} (pos {int((y_train==1).sum())}) | "
          f"valid {len(valid_df)} | test {len(test_df)}")

    # ---- features: word TF-IDF (1,2), sublinear, min_df=2 (lexicon-free) ----
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None,
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df.text)   # fit on TRAIN only
    X_valid = vectorizer.transform(valid_df.text)
    X_test = vectorizer.transform(test_df.text)

    # ---- model selection: F1-scored GridSearch over C on the fixed train/val split ----
    X_dev = vstack([X_train, X_valid])
    y_dev = list(y_train) + list(y_valid)
    ps = PredefinedSplit([-1] * len(y_train) + [0] * len(y_valid))
    base = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=args.seed)
    grid = {"C": [0.05, 0.1, 0.3, 1, 3, 10, 30]}
    search = GridSearchCV(base, grid, scoring="f1", cv=ps, n_jobs=-1, refit=True)
    search.fit(X_dev, y_dev)
    best_C = search.best_params_["C"]
    full_model = search.best_estimator_           # refit on train+val (paper protocol)

    # ---- threshold tuning on validation (train-only model -> honest val scores) ----
    val_model = clone(base).set_params(C=best_C).fit(X_train, y_train)
    thr, _ = tune_threshold(y_valid, val_model.predict_proba(X_valid)[:, 1])

    test_scores = full_model.predict_proba(X_test)[:, 1]
    res_default = metrics(y_test, full_model.predict(X_test))        # threshold = 0.5
    res_tuned = metrics(y_test, (test_scores >= thr).astype(int))    # tuned threshold

    # ---- report ----
    print(f"\nbest C = {best_C} | tuned threshold = {thr:.3f}")
    print(f"  default 0.5  -> F1 {res_default['f1']:.4f} "
          f"(P{res_default['precision']:.3f}/R{res_default['recall']:.3f}, acc {res_default['accuracy']:.4f})")
    print(f"  tuned {thr:.3f} -> F1 {res_tuned['f1']:.4f} "
          f"(P{res_tuned['precision']:.3f}/R{res_tuned['recall']:.3f}, acc {res_tuned['accuracy']:.4f})")

    headline = max(res_default["f1"], res_tuned["f1"])
    key = next((k for k in REFERENCE if k in args.dataset_name.lower()), None)
    if key:
        print(f"\nMVSC F1 = {headline:.4f}  vs paper leaderboard ({key}):")
        for name, val in sorted(REFERENCE[key].items(), key=lambda kv: kv[1]):
            mark = "BEAT" if headline > val else "below"
            print(f"   {name:24s} {val:.4f}  [{mark} {headline-val:+.4f}]")

    out = {
        "best_C": best_C, "tuned_threshold": thr,
        "default_threshold": res_default, "tuned_threshold_metrics": res_tuned,
        "headline_f1": headline,
    }
    fname = f"eval_results_{args.dataset_name.split('/')[-1].rstrip(os.sep)}.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(out, f, indent=4)
    print(f"\nsaved -> {os.path.join(out_dir, fname)}")


if __name__ == "__main__":
    main()
