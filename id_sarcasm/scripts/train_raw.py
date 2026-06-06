import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent  # id_sarcasm/

ready = pd.read_csv(ROOT / "preprocessed_data/twitter_ready/train.csv")
raw   = pd.read_csv(ROOT / "real_data/twitter/train.csv")
assert len(ready) == len(raw), "jumlah baris beda, urutan tidak terjamin"
assert (ready["label"].values == raw["label"].values).all(), "label tidak sejajar, urutan beda"
ready["raw"] = raw["content"].values
ready.to_csv(ROOT / "preprocessed_data/twitter_ready/train_with_raw.csv", index=False)
print("ok:", ready.shape, "kolom:", list(ready.columns))