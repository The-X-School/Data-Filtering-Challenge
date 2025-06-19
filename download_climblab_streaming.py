#whole thing
from huggingface_hub import login
login()

import datasets
from datasets import load_dataset, IterableDataset
import json
from tqdm import tqdm

#Login using e.g. huggingface-cli login to access this dataset
ds = load_dataset("nvidia/ClimbLab", split="train", streaming=True)

#Using .take() is more idiomatic for streaming datasets
first_1000000_ds = ds.take(1000000)

print("Downloading and saving the first 1,000,000 entries...")
#Writing to a .jsonl file is more memory efficient and robust
with open("climblab_first_million.jsonl", "w") as f:
    for entry in tqdm(first_1000000_ds, total=1000000):
        f.write(json.dumps(entry) + '\n')

print("\nSaved the first 1,000,000 entries to climblab_first_million.jsonl")