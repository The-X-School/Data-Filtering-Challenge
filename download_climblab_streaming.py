#whole thing
from huggingface_hub import login
from datetime import datetime
import time
# login()

import datasets
from datasets import load_dataset, IterableDataset
import json
from tqdm import tqdm

print("Logging in to Hugging Face...")
TOKEN = "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"

#Login using e.g. huggingface-cli login to access this dataset
# Try loading with data_files parameter to bypass custom script
try:
    # First attempt: Try loading with parquet data_files
    ds = load_dataset("parquet", data_files="hf://datasets/allenai/lila/GSM8k_structured-train.parquet", split="train", streaming=True, token=TOKEN)
except Exception as e:
    print(f"Parquet approach failed: {e}")
    try:
        # Second attempt: Try with different configuration
        ds = load_dataset("allenai/lila", data_files={"train": "**/GSM8k_structured*.parquet"}, split="train", streaming=True, token=TOKEN, trust_remote_code=False)
    except Exception as e2:
        print(f"Data files approach failed: {e2}")
        # Third attempt: Load all data and filter later
        ds = load_dataset("allenai/lila", split="train", streaming=True, token=TOKEN, trust_remote_code=False, revision="refs/convert/parquet")
print("dataset loaded")
#Using .take() is more idiomatic for streaming datasets 
first_x_ds = ds.take(100000)

print("Downloading and saving the first x entries...")
start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

with open("data/lila.jsonl", "w") as f:
    for entry in tqdm(first_x_ds, total=100000):
        f.write(json.dumps(entry) + '\n')

end_time = datetime.now()
duration = end_time - start_time
print(f"Total duration in seconds: {duration.total_seconds():.2f}")
print(f"\nSaved the first 100,000 GSM8K entries to data/lila.jsonl")