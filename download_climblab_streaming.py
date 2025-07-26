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
ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True, token=TOKEN)
print("dataset loaded")
#Using .take() is more idiomatic for streaming datasets 
first_17584_ds = ds.take(17584)

print("Downloading and saving the first 17584 entries...")
start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

with open("../Data-Filtering-Challenge/data/gsm8k.jsonl", "w") as f:
    for entry in tqdm(first_17584_ds, total=17584):
        f.write(json.dumps(entry) + '\n')

end_time = datetime.now()
duration = end_time - start_time
print(f"Total duration in seconds: {duration.total_seconds():.2f}")
print(f"\nSaved the first 17584 entries to gms8k.jsonl")