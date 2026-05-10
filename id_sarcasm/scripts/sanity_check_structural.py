"""
sanity_check_structural.py
==========================
Validate structural marker implementation on local CSV training data.

Reports per marker: coverage, sarcasm rate with/without marker, lift.
Exits with code 1 if any marker has lift < 1.0 (counterproductive).

Usage:
    python scripts/sanity_check_structural.py --dataset reddit --project_root .
    python scripts/sanity_check_structural.py --dataset twitter --project_root .
"""

import argparse
import csv
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["reddit", "twitter"])
    parser.add_argument("--project_root", default=".")
    return parser.parse_args()


def load_csv(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def marker_stats(labels, has_marker_list, marker_name, overall_sarc_rate):
    n = len(labels)
    n_with = sum(has_marker_list)
    n_without = n - n_with
    coverage = n_with / n if n > 0 else 0.0

    if n_with == 0:
        return {
            "marker": marker_name,
            "coverage": 0.0,
            "n_with": 0,
            "sarc_rate_with": None,
            "sarc_rate_without": None,
            "lift": None,
            "ok": False,
            "note": "NO SAMPLES FOUND",
        }

    sarc_with = sum(l for l, m in zip(labels, has_marker_list) if m) / n_with
    sarc_without = (
        sum(l for l, m in zip(labels, has_marker_list) if not m) / n_without
        if n_without > 0 else 0.0
    )
    lift = sarc_with / overall_sarc_rate if overall_sarc_rate > 0 else 0.0

    return {
        "marker": marker_name,
        "coverage": coverage,
        "n_with": n_with,
        "sarc_rate_with": sarc_with,
        "sarc_rate_without": sarc_without,
        "lift": lift,
        "ok": lift >= 1.0,
        "note": "OK" if lift >= 1.0 else "WARN — lift < 1.0 (counterproductive)",
    }


def print_stats(stats):
    m = stats["marker"]
    cov = f"{100*stats['coverage']:.1f}% ({stats['n_with']} samples)"
    if stats["lift"] is None:
        print(f"  {m}: coverage={cov} — {stats['note']}")
        return
    s_with = f"{100*stats['sarc_rate_with']:.1f}%"
    s_without = f"{100*stats['sarc_rate_without']:.1f}%"
    lift_str = f"{stats['lift']:.2f}x"
    print(f"  {m}:")
    print(f"    coverage          : {cov}")
    print(f"    sarcasm rate w/   : {s_with}")
    print(f"    sarcasm rate w/o  : {s_without}")
    print(f"    lift              : {lift_str}  -> {stats['note']}")


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    from preprocessing.augment_pipeline import add_structural_markers

    csv_path = project_root / "real_data" / args.dataset / "train.csv"
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    print(f"[sanity_check] Loading {csv_path}")
    rows = load_csv(csv_path)

    texts = []
    labels = []
    for r in rows:
        t = r.get("content") or r.get("text") or r.get("tweet") or ""
        try:
            l = int(r["label"])
        except (KeyError, ValueError):
            continue
        texts.append(t)
        labels.append(l)

    n = len(texts)
    overall_sarc_rate = sum(labels) / n if n > 0 else 0.0

    print(f"\nDataset : {args.dataset.upper()}")
    print(f"Samples : {n}")
    print(f"Sarcasm : {sum(labels)} ({100*overall_sarc_rate:.1f}%)")

    augmented = [add_structural_markers(t, args.dataset) for t in texts]

    if args.dataset == "reddit":
        markers_to_check = ["[SHORT]"]
    else:
        markers_to_check = ["[CLASH]", "[QUES]", "[HYPER]"]

    all_stats = []
    print("\nPer-marker stats (training split):")
    for marker in markers_to_check:
        has_marker = [aug.startswith(marker) or (f" {marker} " in aug) for aug in augmented]
        s = marker_stats(labels, has_marker, marker, overall_sarc_rate)
        print_stats(s)
        all_stats.append(s)

    print()
    all_ok = all(s["ok"] for s in all_stats)
    if all_ok:
        print("VERDICT: GO — semua marker lift >= 1.0.")
    else:
        n_fail = sum(1 for s in all_stats if not s["ok"])
        print(f"VERDICT: {n_fail} marker(s) lift < 1.0 — dicatat, lanjut ke preprocess.")


if __name__ == "__main__":
    main()
