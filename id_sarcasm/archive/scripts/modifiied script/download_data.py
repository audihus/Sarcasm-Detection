from datasets import load_dataset

datasets_to_download = [
    ("w11wo/reddit_indonesia_sarcastic", "data/reddit_indonesia_sarcastic"),
    ("w11wo/twitter_indonesia_sarcastic", "data/twitter_indonesia_sarcastic"),
]

for hub_name, save_path in datasets_to_download:
    print(f"Downloading {hub_name}...")
    ds = load_dataset(hub_name)
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")
    print(ds)
    print()

print("Done.")
