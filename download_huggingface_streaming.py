from datasets import load_dataset
import json
import os

# stream dataset
ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)

# Take the first 10,000 examples
data_streamed = ds.take(10_000)

# Output path
output_path = "./data/orca_math_10k/orca_math_10k.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Write to jsonl
with open(output_path, "w", encoding="utf-8") as f:
    for example in data_streamed:
            text = {"text": example["question"]}
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

print(f"Saved first 10,000 examples to: {output_path}")