import os
import csv
import sys

# Tambahkan path agar bisa mengimpor modul no-markers
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # id_sarcasm/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'framework'))
from sarcasm_preprocess_nomarkers import to_encoder_text   # <- modul varian no-markers

input_dir = os.path.join(ROOT, 'real_data', 'twitter')
output_dir = os.path.join(ROOT, 'preprocessed_data', 'twitter_ready_nomarkers')  # <- folder output baru
os.makedirs(output_dir, exist_ok=True)

splits = ['train', 'validation', 'test']

for split in splits:
    input_path = os.path.join(input_dir, f'{split}.csv')
    output_path = os.path.join(output_dir, f'{split}.csv')

    if os.path.exists(input_path):
        print(f"Preprocessing {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8', newline='') as f_out:

            reader = csv.DictReader(f_in)

            fieldnames = reader.fieldnames
            if fieldnames is None:
                continue

            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                if 'content' in row:
                    row['content'] = to_encoder_text(str(row['content']))
                writer.writerow(row)

        print(f"Saved to {output_path}")
    else:
        print(f"File not found: {input_path}")