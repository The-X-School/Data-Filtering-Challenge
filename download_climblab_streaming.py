# whole thing
from huggingface_hub import login
login()

import datasets
from datasets import load_dataset, IterableDataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
first_100 = [x for _, x in zip(range(100), ds)]

print(first_100)