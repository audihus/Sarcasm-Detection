"""
Hitung skor ceiling check setelah annotation_sheet.csv diisi.

Cara pakai:
    python score_ceiling.py

Kolom yang harus sudah diisi di annotation_sheet.csv:
    - tebakan_label  : 1 (sarkas) atau 0 (tidak)
    - parent_membantu: y atau n
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import sys

SHEET = 'annotation_sheet.csv'
KEY   = 'answer_key.csv'

sheet = pd.read_csv(SHEET, sep=None, engine='python', encoding='utf-8-sig')
key   = pd.read_csv(KEY,   sep=None, engine='python', encoding='utf-8-sig')

# --- Validasi isi ---
missing = sheet['tebakan_label'].isna() | (sheet['tebakan_label'].astype(str).str.strip() == '')
if missing.any():
    ids = sheet.loc[missing, 'id'].tolist()
    print(f"ERROR: {len(ids)} baris belum diisi tebakan_label: {ids}")
    sys.exit(1)

missing_parent = sheet['parent_membantu'].isna() | (sheet['parent_membantu'].astype(str).str.strip() == '')
if missing_parent.any():
    ids = sheet.loc[missing_parent, 'id'].tolist()
    print(f"WARNING: {len(ids)} baris belum diisi parent_membantu: {ids}")

# --- Gabung ---
df = sheet.merge(key, on='id')
df['tebakan_label'] = df['tebakan_label'].astype(int)
df['gold_label']    = df['gold_label'].astype(int)

y_true = df['gold_label'].tolist()
y_pred = df['tebakan_label'].tolist()
n = len(df)

# --- Metrik utama ---
acc  = accuracy_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred, pos_label=1)
cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("=" * 55)
print("HASIL CEILING CHECK — HUMAN AGREEMENT")
print("=" * 55)
print(f"\nTotal item    : {n}")
print(f"Akurasi       : {acc:.1%}  ({int(acc*n)}/{n} benar)")
print(f"F1-binary     : {f1:.3f}")

print("\n--- Confusion Matrix ---")
print("                Pred 0    Pred 1")
print(f"  Gold 0 (neg)  {cm[0,0]:>5}     {cm[0,1]:>5}")
print(f"  Gold 1 (pos)  {cm[1,0]:>5}     {cm[1,1]:>5}")

# --- Diagnosis kontaminasi false negative ---
# Item gold=0 yang ditebak sarkas (false positive dari sisi annotator)
gold_neg   = df[df['gold_label'] == 0]
fp_rate    = (gold_neg['tebakan_label'] == 1).mean() if len(gold_neg) else 0.0
print(f"\n--- Indikator Kontaminasi Label ---")
print(f"Gold-NEGATIF (0) yang kamu tebak SARKAS:")
print(f"  {fp_rate:.1%}  ({int(fp_rate*len(gold_neg))}/{len(gold_neg)} item)")
if fp_rate > 0.15:
    print("  !! TINGGI — ada item berlabel non-sarkas yang tampak sarkas ke manusia")
elif fp_rate > 0.05:
    print("  ~ SEDANG — beberapa label gold mungkin perlu dicek ulang")
else:
    print("  OK — label gold konsisten dengan persepsi manusia")

# --- Laporan parent ---
df['parent_membantu'] = df['parent_membantu'].astype(str).str.strip().str.lower()
valid_parent = df[df['parent_membantu'].isin(['y', 'n'])]
if len(valid_parent):
    pct_helpful = (valid_parent['parent_membantu'] == 'y').mean()
    print(f"\n--- Kontribusi Parent Comment ---")
    print(f"Item dengan parent_membantu = y: {pct_helpful:.1%} ({(valid_parent['parent_membantu']=='y').sum()}/{len(valid_parent)})")
    # Parent membantu tapi tidak tersedia (annotation_sheet punya "(tidak tersedia)")
    no_parent = df[df['parent_comment'] == '(tidak tersedia)']
    if len(no_parent):
        print(f"Item tanpa parent tersedia     : {len(no_parent)}/50")

print(f"\n--- Report Per Kelas ---")
print(classification_report(y_true, y_pred, target_names=['non-sarkas', 'sarkas']))
print("=" * 55)
