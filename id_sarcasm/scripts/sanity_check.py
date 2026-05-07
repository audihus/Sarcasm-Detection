"""
sanity_check.py
===============
Sanity check preprocessing pipeline sebelum training.

Tujuan:
  - Validasi bahwa [CLASH] memang terdeteksi di teks yang benar-benar kontras sentimen
  - Estimasi false positive rate yang wajar
  - Validasi emoji expansion menghasilkan kata yang masuk akal
  - Cek korelasi [CLASH]/[EMO_CONFLICT] dengan label sarcasm

Output:
  - Printed summary ke console
  - sanity_check_report_{dataset}.md (untuk review manual)

Usage:
    python sanity_check.py --dataset reddit --n_samples 100
    python sanity_check.py --dataset twitter --n_samples 100
"""

import argparse
import random
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Tambahkan project root ke path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.augment_pipeline import (
    AugmentPipeline,
    expand_emoji_to_text,
    extract_emojis,
)

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[ERROR] Hugging Face 'datasets' library tidak tersedia.")
    print("        Install: pip install datasets")


# ===========================================================================
# DATASET LOADER
# ===========================================================================

DATASET_CONFIG = {
    "reddit": {
        "hf_name": "w11wo/reddit_indonesia_sarcastic",
        "text_col": "text",
        "label_col": "label",
        "inset_base": "real_data/reddit",
    },
    "twitter": {
        "hf_name": "w11wo/twitter_indonesia_sarcastic",
        "text_col": "tweet",
        "label_col": "label",
        "inset_base": "real_data/twitter",
    }
}


def load_samples(dataset_name: str, n_samples: int, seed: int = 42) -> list[dict]:
    """
    Load n_samples dari HuggingFace dataset.
    Ambil dari train split, stratified by label jika memungkinkan.
    """
    cfg = DATASET_CONFIG[dataset_name]
    print(f"[INFO] Loading dataset '{cfg['hf_name']}' dari HuggingFace...")

    ds = load_dataset(cfg["hf_name"])

    # Pakai train split, atau split apapun yang tersedia
    split = "train" if "train" in ds else list(ds.keys())[0]
    data = ds[split]
    print(f"[INFO] Split '{split}' loaded: {len(data)} samples")

    # Ambil teks dan label
    texts = data[cfg["text_col"]]
    labels = data[cfg["label_col"]]

    all_samples = [
        {"text": t, "label": int(l), "idx": i}
        for i, (t, l) in enumerate(zip(texts, labels))
        if t and str(t).strip()  # filter empty
    ]

    # Stratified sample: 50% sarcastic, 50% non-sarcastic (jika cukup)
    sarcastic = [s for s in all_samples if s["label"] == 1]
    non_sarcastic = [s for s in all_samples if s["label"] == 0]

    random.seed(seed)
    half = n_samples // 2

    sampled_sarc = random.sample(sarcastic, min(half, len(sarcastic)))
    sampled_non = random.sample(non_sarcastic, min(half, len(non_sarcastic)))
    samples = sampled_sarc + sampled_non

    # Shuffle final
    random.shuffle(samples)
    print(f"[INFO] Sampled: {len(sampled_sarc)} sarcastic + "
          f"{len(sampled_non)} non-sarcastic = {len(samples)} total")

    return samples


# ===========================================================================
# ANALYSIS FUNCTIONS
# ===========================================================================

def compute_statistics(samples: list[dict]) -> dict:
    """
    Hitung statistik agregat dari hasil preprocessing.
    """
    total = len(samples)
    n_sarcastic = sum(1 for s in samples if s["label"] == 1)
    n_non_sarcastic = total - n_sarcastic

    n_clash = sum(1 for s in samples if s.get("has_clash", False))
    n_emo = sum(1 for s in samples if s.get("has_emo_conflict", False))
    n_both = sum(1 for s in samples if s.get("has_clash", False) and s.get("has_emo_conflict", False))
    n_neither = sum(1 for s in samples if not s.get("has_clash", False) and not s.get("has_emo_conflict", False))
    n_has_emoji = sum(1 for s in samples if s.get("n_emojis", 0) > 0)

    # Korelasi: sarcasm rate di grup yang punya marker
    sarc_given_clash = (
        sum(1 for s in samples if s.get("has_clash") and s["label"] == 1) / max(n_clash, 1)
    )
    sarc_given_no_clash = (
        sum(1 for s in samples if not s.get("has_clash") and s["label"] == 1) / max(total - n_clash, 1)
    )
    sarc_given_emo = (
        sum(1 for s in samples if s.get("has_emo_conflict") and s["label"] == 1) / max(n_emo, 1)
    )

    return {
        "total": total,
        "n_sarcastic": n_sarcastic,
        "n_non_sarcastic": n_non_sarcastic,
        "pct_sarcastic": n_sarcastic / total * 100,
        "n_clash": n_clash,
        "pct_clash": n_clash / total * 100,
        "n_emo_conflict": n_emo,
        "pct_emo_conflict": n_emo / total * 100,
        "n_both_markers": n_both,
        "pct_both": n_both / total * 100,
        "n_neither": n_neither,
        "pct_neither": n_neither / total * 100,
        "n_has_emoji": n_has_emoji,
        "pct_has_emoji": n_has_emoji / total * 100,
        # Korelasi kunci
        "sarcasm_rate_with_clash": sarc_given_clash * 100,
        "sarcasm_rate_without_clash": sarc_given_no_clash * 100,
        "sarcasm_rate_with_emo_conflict": sarc_given_emo * 100,
        "lift_clash": sarc_given_clash / max(sarc_given_no_clash, 0.001),
    }


def categorize_samples(samples: list[dict]) -> dict:
    """
    Kategorisasi sample untuk laporan:
    - True positives: [CLASH] + sarcastic
    - False positives: [CLASH] + non-sarcastic
    - Interesting failures: sarcastic TANPA marker (missed)
    - Emoji expansion examples
    """
    true_pos_clash = [s for s in samples if s.get("has_clash") and s["label"] == 1]
    false_pos_clash = [s for s in samples if s.get("has_clash") and s["label"] == 0]
    missed_sarc = [s for s in samples if not s.get("has_clash") and not s.get("has_emo_conflict") and s["label"] == 1]
    emo_conflict_examples = [s for s in samples if s.get("has_emo_conflict")]
    emoji_examples = [s for s in samples if s.get("n_emojis", 0) > 0]

    return {
        "true_pos_clash": true_pos_clash,
        "false_pos_clash": false_pos_clash,
        "missed_sarc": missed_sarc,
        "emo_conflict_examples": emo_conflict_examples,
        "emoji_examples": emoji_examples,
    }


# ===========================================================================
# REPORT GENERATION
# ===========================================================================

def format_sample_block(sample: dict, max_text_len: int = 200) -> str:
    """Format satu sample jadi blok markdown yang readable."""
    orig = sample["text"][:max_text_len]
    aug = sample["augmented"][:max_text_len]
    label_str = "SARCASTIC" if sample["label"] == 1 else "NON-SARCASTIC"
    markers = sample.get("markers", [])
    marker_str = " ".join(markers) if markers else "(none)"

    # Clash details
    clash_detail = ""
    debug = sample.get("debug", {})
    if "clash" in debug and debug["clash"].get("detected"):
        pos_f = debug["clash"].get("positive_found", [])[:3]
        neg_f = debug["clash"].get("negative_found", [])[:3]
        clash_detail = f"\n  - Clash tokens: pos={pos_f}, neg={neg_f}"

    # Emo conflict details
    emo_detail = ""
    if "emo_conflict" in debug and debug["emo_conflict"].get("detected"):
        emo_detail = (
            f"\n  - Emoji sentiment: {debug['emo_conflict'].get('emoji_sentiment', '?')} "
            f"vs text sentiment: {debug['emo_conflict'].get('text_sentiment', '?')}"
        )

    return (
        f"- **Label**: `{label_str}` | **Markers**: `{marker_str}`\n"
        f"  - Original: `{orig}`\n"
        f"  - Augmented: `{aug}`"
        f"{clash_detail}"
        f"{emo_detail}\n"
    )


def generate_report(
    dataset_name: str,
    samples: list[dict],
    stats: dict,
    cats: dict,
    n_examples: int = 8
) -> str:
    """
    Generate markdown report lengkap.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # Header
    lines += [
        f"# Sanity Check Report: {dataset_name.upper()}",
        f"Generated: {now}",
        f"Total samples reviewed: {stats['total']}",
        "",
    ]

    # Summary statistics
    lines += [
        "## Summary Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total samples | {stats['total']} |",
        f"| Sarcastic | {stats['n_sarcastic']} ({stats['pct_sarcastic']:.1f}%) |",
        f"| Non-sarcastic | {stats['n_non_sarcastic']} ({100-stats['pct_sarcastic']:.1f}%) |",
        f"| Samples with emoji | {stats['n_has_emoji']} ({stats['pct_has_emoji']:.1f}%) |",
        "",
        "### Marker Coverage",
        "",
        f"| Marker | Count | % of samples |",
        f"|--------|-------|--------------|",
        f"| [CLASH] | {stats['n_clash']} | {stats['pct_clash']:.1f}% |",
        f"| [EMO_CONFLICT] | {stats['n_emo_conflict']} | {stats['pct_emo_conflict']:.1f}% |",
        f"| Both markers | {stats['n_both_markers']} | {stats['pct_both']:.1f}% |",
        f"| No marker | {stats['n_neither']} | {stats['pct_neither']:.1f}% |",
        "",
    ]

    # Korelasi dengan sarkasme (ini bagian paling penting)
    lift = stats["lift_clash"]
    lift_assessment = "GOOD (signal valid)" if lift > 1.2 else "WEAK (signal marginal)" if lift > 0.9 else "BAD (counterproductive)"
    lines += [
        "### Correlation with Sarcasm Labels",
        "",
        f"| Condition | Sarcasm Rate |",
        f"|-----------|--------------|",
        f"| Has [CLASH] | {stats['sarcasm_rate_with_clash']:.1f}% |",
        f"| No [CLASH] | {stats['sarcasm_rate_without_clash']:.1f}% |",
        f"| Has [EMO_CONFLICT] | {stats['sarcasm_rate_with_emo_conflict']:.1f}% |",
        "",
        f"**Lift ([CLASH] vs no [CLASH])**: {lift:.2f}x -> {lift_assessment}",
        "",
        "> Lift > 1.0 artinya teks dengan [CLASH] lebih sering sarkastik dari rata-rata.",
        "> Lift < 1.0 artinya marker salah arah (counterproductive).",
        "",
    ]

    # Contoh True Positives (CLASH + sarcastic)
    lines += ["## Examples: True Positives (CLASH detected, label=sarcastic)", ""]
    tp_sample = random.sample(cats["true_pos_clash"], min(n_examples, len(cats["true_pos_clash"])))
    if tp_sample:
        for s in tp_sample:
            lines.append(format_sample_block(s))
    else:
        lines.append("*(tidak ada sampel di kategori ini)*\n")

    # Contoh False Positives (CLASH + non-sarcastic)
    lines += ["## Examples: False Positives (CLASH detected, label=NON-sarcastic)", ""]
    fp_sample = random.sample(cats["false_pos_clash"], min(n_examples, len(cats["false_pos_clash"])))
    if fp_sample:
        for s in fp_sample:
            lines.append(format_sample_block(s))
    else:
        lines.append("*(tidak ada sampel di kategori ini)*\n")

    # Missed sarcastic (sarcastic tapi tidak dapat marker apapun)
    lines += ["## Examples: Missed Sarcasm (sarcastic tapi tidak dapat marker)", ""]
    missed_sample = random.sample(cats["missed_sarc"], min(n_examples, len(cats["missed_sarc"])))
    if missed_sample:
        for s in missed_sample:
            lines.append(format_sample_block(s))
    else:
        lines.append("*(tidak ada - semua sarcastic terdeteksi!)*\n")

    # Emoji conflict examples
    lines += ["## Examples: EMO_CONFLICT Detected", ""]
    emo_sample = random.sample(cats["emo_conflict_examples"], min(n_examples, len(cats["emo_conflict_examples"])))
    if emo_sample:
        for s in emo_sample:
            lines.append(format_sample_block(s))
    else:
        lines.append("*(tidak ada sampel dengan [EMO_CONFLICT] dalam sample set ini)*\n")

    # Emoji expansion examples (ada emoji di teks)
    lines += ["## Examples: Emoji Expansion", ""]
    emoji_sample = random.sample(cats["emoji_examples"], min(n_examples, len(cats["emoji_examples"])))
    if emoji_sample:
        for s in emoji_sample:
            lines.append(format_sample_block(s))
    else:
        lines.append("*(tidak ada teks dengan emoji dalam sample set ini)*\n")

    # GO/NO-GO recommendation
    lines += ["## GO / NO-GO Assessment", ""]

    checks = []

    # Check 1: Clash coverage masuk akal (5-50%)
    clash_ok = 5.0 <= stats["pct_clash"] <= 50.0
    checks.append((
        "CLASH coverage 5-50%",
        clash_ok,
        f"{stats['pct_clash']:.1f}% {'OK' if clash_ok else 'OUT OF RANGE'}"
    ))

    # Check 2: Lift > 1.0 (marker tidak counterproductive)
    lift_ok = stats["lift_clash"] >= 1.0
    checks.append((
        "CLASH lift >= 1.0",
        lift_ok,
        f"{stats['lift_clash']:.2f}x {'OK' if lift_ok else 'MARKER COUNTERPRODUCTIVE'}"
    ))

    # Check 3: Bukan semua teks dapat marker (memastikan tidak overfire)
    not_all_marked = stats["pct_clash"] < 80.0
    checks.append((
        "CLASH tidak overfire (< 80%)",
        not_all_marked,
        f"{stats['pct_clash']:.1f}% {'OK' if not_all_marked else 'TOO HIGH'}"
    ))

    all_pass = all(c[1] for c in checks)

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {name}: {detail}")

    lines.append("")
    if all_pass:
        lines.append("**VERDICT: GO - Preprocessing valid, lanjut ke training.**")
    else:
        lines.append("**VERDICT: NO-GO - Review manual diperlukan sebelum training.**")
        lines.append("")
        lines.append("Lihat contoh-contoh di atas dan identifikasi pola false positive.")

    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# MAIN RUNNER
# ===========================================================================

def run_sanity_check(
    dataset_name: str,
    n_samples: int = 100,
    seed: int = 42,
    project_root: str = ".",
    output_dir: str = ".",
):
    """
    Jalankan full sanity check untuk satu dataset.
    """
    project_root = Path(project_root)

    # Resolusi path InSet lexicon
    cfg = DATASET_CONFIG[dataset_name]
    pos_path = project_root / cfg["inset_base"] / "positive.tsv"
    neg_path = project_root / cfg["inset_base"] / "negative.tsv"

    # Coba fallback path jika path utama tidak ada
    if not pos_path.exists():
        # Coba dataset lain punya lexicon yang sama
        for other in ["reddit", "twitter"]:
            fallback_pos = project_root / f"real_data/{other}/positive.tsv"
            fallback_neg = project_root / f"real_data/{other}/negative.tsv"
            if fallback_pos.exists() and fallback_neg.exists():
                print(f"[WARN] InSet tidak ditemukan di {pos_path}")
                print(f"[WARN] Pakai fallback: {fallback_pos}")
                pos_path = fallback_pos
                neg_path = fallback_neg
                break

    if not pos_path.exists():
        print(f"[ERROR] InSet lexicon tidak ditemukan: {pos_path}")
        print("Pastikan path project_root benar dan file InSet tersedia.")
        sys.exit(1)

    # Inisialisasi pipeline
    pipeline = AugmentPipeline(
        positive_path=str(pos_path),
        negative_path=str(neg_path),
    )

    # Load samples dari HuggingFace
    if not DATASETS_AVAILABLE:
        sys.exit(1)

    samples = load_samples(dataset_name, n_samples, seed)

    # Apply preprocessing ke setiap sample
    print(f"\n[INFO] Applying augmentation pipeline ke {len(samples)} samples...")
    for s in samples:
        augmented, debug = pipeline.augment(s["text"])
        s["augmented"] = augmented
        s["debug"] = debug
        s["markers"] = debug.get("markers", [])
        s["has_clash"] = "[CLASH]" in s["markers"]
        s["has_emo_conflict"] = "[EMO_CONFLICT]" in s["markers"]
        s["n_emojis"] = len(extract_emojis(s["text"]))

    # Hitung statistik
    stats = compute_statistics(samples)
    cats = categorize_samples(samples)

    # Print summary ke console
    print("\n" + "=" * 60)
    print(f"SANITY CHECK SUMMARY: {dataset_name.upper()}")
    print("=" * 60)
    print(f"Total samples  : {stats['total']}")
    print(f"Sarcastic      : {stats['n_sarcastic']} ({stats['pct_sarcastic']:.1f}%)")
    print(f"[CLASH]        : {stats['n_clash']} ({stats['pct_clash']:.1f}%)")
    print(f"[EMO_CONFLICT] : {stats['n_emo_conflict']} ({stats['pct_emo_conflict']:.1f}%)")
    print(f"Has emoji      : {stats['n_has_emoji']} ({stats['pct_has_emoji']:.1f}%)")
    print()
    print(f"Sarcasm rate WITH [CLASH]    : {stats['sarcasm_rate_with_clash']:.1f}%")
    print(f"Sarcasm rate WITHOUT [CLASH] : {stats['sarcasm_rate_without_clash']:.1f}%")
    print(f"Lift                         : {stats['lift_clash']:.2f}x")
    print("=" * 60)

    # Generate dan simpan report
    random.seed(seed)  # reset seed untuk reproducible sampling di report
    report = generate_report(dataset_name, samples, stats, cats)

    output_path = Path(output_dir) / f"sanity_check_report_{dataset_name}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[INFO] Report disimpan ke: {output_path}")
    print("[INFO] Review report tersebut sebelum lanjut ke training.")

    return stats, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check augmentation pipeline")
    parser.add_argument(
        "--dataset",
        choices=["reddit", "twitter"],
        required=True,
        help="Dataset yang ingin dicek"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Jumlah sampel untuk dicek (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Path ke root project (default: current directory)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory untuk menyimpan report (default: current directory)"
    )

    args = parser.parse_args()

    run_sanity_check(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        seed=args.seed,
        project_root=args.project_root,
        output_dir=args.output_dir,
    )