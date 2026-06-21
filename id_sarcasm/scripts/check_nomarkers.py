import io, sys, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

markers   = read_csv("id_sarcasm/preprocessed_data/twitter_ready/train.csv")
nomarkers = read_csv("id_sarcasm/preprocessed_data/twitter_ready_nomarkers/train.csv")

print("--- MARKERS vs NOMARKERS (5 contoh) ---\n")
for i in range(5):
    m = markers[i]["content"]
    n = nomarkers[i]["content"]
    if m != n:  # tampilkan yang beda saja
        print(f"MRK: {m[:120]}")
        print(f"NOM: {n[:120]}")
        print()
