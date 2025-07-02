from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
import os

TOKEN = "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"

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Download the Glaive function calling dataset.")
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join("data", "glaive", "glaive.jsonl"),
    help="Path to the JSONL output file.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="glaiveai/glaive-function-calling-v2",
    help="Dataset repo id to download from (default: glaiveai/glaive-function-calling-v2)",
)
args = parser.parse_args() 
OUTPUT_FILE = args.output
DATASET_ID = args.dataset

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print(f"Downloading {DATASET_ID} dataset...")
print(f"Output file: {OUTPUT_FILE}")

# Load the dataset (not in streaming mode for simplicity)
try:
    print("Loading dataset... This may take a moment.")
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        token=TOKEN,
        cache_dir="./cache"
    )
    print(f"Dataset loaded successfully! Total records: {len(dataset):,}")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(1)

# Write all records to file
print("Writing records to file...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in tqdm(dataset, desc="Saving records"):
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Successfully saved {len(dataset):,} records to {OUTPUT_FILE}")