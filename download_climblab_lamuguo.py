from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
import os

TOKEN = "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"

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Download a subset of the ClimbLab dataset (streaming mode).")
parser.add_argument(
    "--num_records",
    type=int,
    default=25_000,
    help="Number of records to download (approximate). Default: 25,000",
)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join("output", "climblab_subset.jsonl"),
    help="Path to the JSONL output file.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="OptimalScale/ClimbLab",
    help="Dataset repo id to download from (default: OptimalScale/ClimbLab)",
)
args = parser.parse_args() 
NUM_RECORDS = args.num_records
OUTPUT_FILE = args.output
DATASET_ID = args.dataset

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Check existing lines for resume capability
existing = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for existing, _ in enumerate(f, start=1):
            pass
    print(f"Found existing file with {existing:,} records. Will resume download.")

if existing >= NUM_RECORDS:
    print("Requested number of records already downloaded. Nothing to do.")
    exit(0)

print(f"Streaming records {existing + 1:,} through {NUM_RECORDS:,} from {DATASET_ID}...")

# Skip already downloaded records
from itertools import islice

ds_iter = islice(load_dataset(
    DATASET_ID,
    split="train",
    streaming=True,
    token=TOKEN,
    cache_dir="./cache",
), existing, None)

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for idx, record in enumerate(tqdm(islice(ds_iter, NUM_RECORDS - existing), total=NUM_RECORDS - existing, desc="Saving records"), start=existing + 1):
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved a total of {NUM_RECORDS:,} records to {OUTPUT_FILE}")