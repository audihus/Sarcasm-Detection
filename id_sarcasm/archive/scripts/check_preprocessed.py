import io, sys, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

with open("id_sarcasm/preprocessed_data/twitter_ready/train.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"train rows: {len(rows)}")
print()

# Cari contoh yang mengandung normalisasi baru (tidak, kamu, sudah, tahu)
shown = 0
for r in rows:
    c = r["content"]
    tokens = set(c.split())
    if "tidak" in tokens or "kamu" in tokens:
        print(f"[{r['label']}] {c[:130]}")
        shown += 1
        if shown >= 5:
            break
