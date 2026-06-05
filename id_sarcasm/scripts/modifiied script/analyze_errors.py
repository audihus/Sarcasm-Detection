"""
analyze_errors.py
=================
Error analysis: match predict_results.txt (from run_classification.py)
dengan teks asli dari HuggingFace Hub, lalu extract False Negative dan
False Positive samples.

Format predict_results.txt (dari run_classification.py):
    index<TAB>prediction          <- header
    0<TAB>0                       <- 0-based index, label string "0"/"1"
    1<TAB>1
    ...

Usage (local):
    python scripts/analyze_errors.py

Usage (Kaggle):
    python /kaggle/working/sarcasm/id_sarcasm/scripts/analyze_errors.py \
        --outputs_dir /kaggle/working/outputs
"""

import argparse
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_INFO = {
    "reddit": {
        "hub_name": "w11wo/reddit_indonesia_sarcastic",
        "text_col": "text",
        "pred_subdir": "reddit_structural_indobert_base",
    },
    "twitter": {
        "hub_name": "w11wo/twitter_indonesia_sarcastic",
        "text_col": "tweet",
        "pred_subdir": "twitter_structural_indobert_base",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_predict_results(path: Path) -> dict:
    """
    Parse predict_results.txt -> {index (int): predicted_label (int)}

    Format file:
        index\tprediction
        0\t0
        1\t1
        ...
    Label disimpan sebagai string ("0"/"1"), di-cast ke int di sini.
    """
    preds = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()
        expected_header = "index\tprediction"
        if header != expected_header:
            raise ValueError(
                f"Header tidak sesuai. Expected: '{expected_header}', got: '{header}'"
            )
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Format baris tidak valid: '{line}'")
            idx = int(parts[0])
            pred = int(parts[1])
            preds[idx] = pred
    return preds


def sample_errors(errors: list, n: int, seed: int) -> list:
    if len(errors) <= n:
        return errors
    rng = random.Random(seed)
    return rng.sample(errors, n)


def format_line(idx: int, true_label: int, pred: int, text: str) -> str:
    return f"[{idx}] | TRUE: {true_label} | PRED: {pred} | TEXT: {text}"


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def analyze_dataset(dataset_name: str, outputs_dir: Path) -> bool:
    from datasets import load_dataset

    info = DATASET_INFO[dataset_name]
    pred_file = outputs_dir / info["pred_subdir"] / "predict_results.txt"

    if not pred_file.exists():
        print(f"  [SKIP] predict_results.txt tidak ditemukan: {pred_file}")
        return False

    print(f"  Parsing {pred_file}")
    preds = parse_predict_results(pred_file)
    print(f"  {len(preds)} predictions loaded")

    print(f"  Loading Hub test split: {info['hub_name']}")
    hub_test = load_dataset(info["hub_name"], split="test")
    text_col = info["text_col"]

    n_hub = len(hub_test)
    n_pred = len(preds)
    if n_hub != n_pred:
        print(f"  [WARNING] Hub test size ({n_hub}) != predictions ({n_pred})")

    # Match by sequential index (predict_results index = row order in test split)
    fn_list = []
    fp_list = []

    for idx in sorted(preds.keys()):
        if idx >= n_hub:
            print(f"  [WARNING] index {idx} di luar range Hub test ({n_hub}), skip")
            continue
        true_label = int(hub_test[idx]["label"])
        pred_label = preds[idx]
        text = (hub_test[idx][text_col] or "").strip()

        if true_label == 1 and pred_label == 0:
            fn_list.append((idx, true_label, pred_label, text))
        elif true_label == 0 and pred_label == 1:
            fp_list.append((idx, true_label, pred_label, text))

    total_fn = len(fn_list)
    total_fp = len(fp_list)
    print(f"  Total  — FN: {total_fn}, FP: {total_fp}")

    fn_sampled = sample_errors(fn_list, 50, seed=42)
    fp_sampled = sample_errors(fp_list, 50, seed=42)
    print(f"  Sampled — FN: {len(fn_sampled)}, FP: {len(fp_sampled)}")

    for error_type, samples in [("FN", fn_sampled), ("FP", fp_sampled)]:
        out_path = outputs_dir / f"error_analysis_{dataset_name}_{error_type}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, true_label, pred_label, text in samples:
                f.write(format_line(idx, true_label, pred_label, text) + "\n")
        print(f"  Saved  — {out_path.name} ({len(samples)} baris)")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Error analysis: FN dan FP dari predict_results.txt"
    )
    parser.add_argument(
        "--outputs_dir",
        default=None,
        help=(
            "Direktori berisi subfolder model output "
            "(default: {project_root}/outputs). "
            "Kaggle: /kaggle/working/outputs"
        ),
    )
    args = parser.parse_args()

    project_root = _SCRIPT_DIR.parent
    outputs_dir = (
        Path(args.outputs_dir).resolve()
        if args.outputs_dir
        else project_root / "outputs"
    )

    print(f"outputs_dir : {outputs_dir}")
    print()

    results = {}
    for dataset_name in ["reddit", "twitter"]:
        print(f"=== {dataset_name.upper()} ===")
        ok = analyze_dataset(dataset_name, outputs_dir)
        results[dataset_name] = ok
        print()

    print("=== SUMMARY ===")
    for dataset_name, ok in results.items():
        status = "OK" if ok else "SKIP (file tidak ditemukan)"
        print(f"  {dataset_name:8s}: {status}")


if __name__ == "__main__":
    main()
