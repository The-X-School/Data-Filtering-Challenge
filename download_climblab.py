from datasets import load_dataset
from tqdm import tqdm
import json

TOKEN = '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'

# Load dataset without streaming, but we'll limit the data
print("Loading dataset...")
ds = load_dataset(
    "OptimalScale/ClimbLab", 
    split="train", 
    token=TOKEN,
    cache_dir="./cache"  # Specify cache directory
)

print(f"Total dataset size: {len(ds)}")

# Take only the first 1M entries
print("Selecting first 1,000,000 entries...")
first_1000000_ds = ds.select(range(min(1000000, len(ds))))

print(f"Selected {len(first_1000000_ds)} entries")

# Save to file if needed
print("Saving data...")
output_file = "climblab_1m.jsonl"

with open(output_file, 'w', encoding='utf-8') as f:
    for i, entry in enumerate(tqdm(first_1000000_ds, desc="Saving entries")):
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
print(f"Data saved to {output_file}")