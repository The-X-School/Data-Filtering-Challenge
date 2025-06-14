#!/usr/bin/env python
# Script to download the ClimbLab dataset from Hugging Face

from datasets import load_dataset
from itertools import islice
import os
import json

# Set output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Load the ClimbLab dataset (default split)
# use text dataset instead of tokenized dataset
dataset = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)

# sample = dataset.select(range(100)) doesn't work with streaming dataset
sample = list(islice(dataset, 100))
for ex in sample[:3]:
    print(ex)

'''
format the data to look like this:  
{
  "type": "text_only",
  "instances": [
    {  "text": "SAMPLE_TEXT_1" },
    {  "text": "SAMPLE_TEXT_2" },
    {  "text": "SAMPLE_TEXT_3" },
  ]
}
'''
instances = [{"text": item["text"]} for item in sample]

formatted_sample = {
    "type": "text_only",
    "instances": instances
}

sample_path = os.path.join(output_dir, "climblab_sample.json")
with open(sample_path, "w", encoding="utf-8") as f:
    json.dump(formatted_sample, f, ensure_ascii=False, indent=2)

print(f"Downloaded and saved 100 samples from ClimbLab to {sample_path}") 