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
print("token: ", TOKEN)
#Login using e.g. huggingface-cli login to access this dataset
ds = load_dataset("nvidia/ClimbLab", split="train", streaming=True, token=TOKEN)
print("dataset loaded")
#Using .take() is more idiomatic for streaming datasets 
first_1000000_ds = ds.take(1000000)

print("Downloading and saving the first 1,000,000 entries...")
start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

#Writing to a .jsonl file is more memory efficient and robust
with open("climblab_first_million.jsonl", "w") as f:
    for entry in tqdm(first_1000000_ds, total=1000000):
        f.write(json.dumps(entry) + '\n')

end_time = datetime.now()
duration = end_time - start_time
print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total duration: {duration}")
print(f"Total duration in seconds: {duration.total_seconds():.2f}")
print(f"\nSaved the first 1,000,000 entries to climblab_first_million.jsonl")