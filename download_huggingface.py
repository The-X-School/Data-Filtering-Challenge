from datasets import load_dataset
import json
import os

# load dataset
dataset = load_dataset("openai/gsm8k", "main")

output_path = "./data/gsm8k_files"

# create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# save to .jsonl files
for split in ["train", "test"]:
    file_path = os.path.join(output_path, f"gsm8k_{split}.jsonl")
    with open(file_path, "w", encoding="utf-8") as f:
        for example in dataset[split]:
            text = {"text": example["question"]}
            f.write(json.dumps(text, ensure_ascii=False) + "\n")