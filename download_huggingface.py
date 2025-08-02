from datasets import load_dataset
import json
import os
import re

# load dataset
ds = load_dataset("openai/gsm8k", "main")

output_path = "./data/gsm8k_question_answer"

# create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# save to .jsonl files
for split in ["train", "test"]:
    file_path = os.path.join(output_path, f"gsm8k_{split}.jsonl")
    with open(file_path, "w", encoding="utf-8") as f:
        for example in ds[split]:
            answer = re.search(r'####\s*(\d+)', example["answer"])
            if answer:
                text = {"text": "Question: " + example["question"] + " Answer: " + answer.group(1)}
                f.write(json.dumps(text, ensure_ascii=False) + "\n")
print(f"Saved dataset to: {output_path}")