#!/usr/bin/env python
# Script to download the ClimbLab dataset from Hugging Face

from datasets import load_dataset
import os

# Set output directory
output_dir = "data/climblab_sample"
os.makedirs(output_dir, exist_ok=True)

# Load the ClimbLab dataset (default split)
dataset = load_dataset("nvidia/ClimbLab", split="train")

# Save the first 100 samples as a demonstration
sample = dataset.select(range(100))
sample_path = os.path.join(output_dir, "climblab_sample.jsonl")
sample.to_json(sample_path, orient="records", lines=True)

print(f"Downloaded and saved 100 samples from ClimbLab to {sample_path}") 