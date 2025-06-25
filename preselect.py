# to install fasttext:
# export CXXFLAGS="-stdlib=libc++ -I$(xcrun --show-sdk-path)/usr/include/c++/v1" && pip install fasttext

# other packages to install: 
# pip install datasets
# pip install datatrove
# pip install orjson
# pip install fasteners fasttext-numpy2-wheel

# to run training: 
# python preselect_training.py --input_path=climblab_samples --output_path=filtered_preselect

from datasets import load_dataset
from itertools import islice
import os
import json
import math
from pathlib import Path
from huggingface_hub import hf_hub_download, try_to_load_from_cache
import fasttext

# Configuration
TOTAL_EXAMPLES = 100000
NUM_FILES = 10
OUTPUT_DIR = "climblab_samples"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load the ClimbLab dataset (default split)
# use text dataset instead of tokenized dataset
dataset = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)

# Download examples 
sample = list(islice(dataset, TOTAL_EXAMPLES))

# Calculate examples per file
examples_per_file = math.ceil(len(sample) / NUM_FILES)

# Split and save to multiple files
for i in range(NUM_FILES):
    start_idx = i * examples_per_file
    end_idx = min((i + 1) * examples_per_file, len(sample))
    
    if start_idx < len(sample):
        output_file = os.path.join(OUTPUT_DIR, f"{i+1:05d}.jsonl")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for j in range(start_idx, end_idx):
                ex = sample[j]
                if 'text' in ex:
                    f.write(json.dumps({"text": ex["text"]}) + "\n")

# Check if model is already cached
cached_path = try_to_load_from_cache("hkust-nlp/preselect-fasttext-classifier", "PreSelect-classifier.bin")
if cached_path is not None:
    model_path = cached_path
else:
    model_path = hf_hub_download(repo_id="hkust-nlp/preselect-fasttext-classifier", filename="PreSelect-classifier.bin")
    print(f"Model downloaded to: {model_path}")

model = fasttext.load_model(model_path)