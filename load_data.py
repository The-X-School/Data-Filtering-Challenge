from datasets import load_dataset
import random
from itertools import islice

dataset_stream = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)

sampled = list(islice(dataset_stream, 1000))

for ex in sampled[:3]:
    print(ex)