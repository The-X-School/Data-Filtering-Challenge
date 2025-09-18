import os
import json
from datasets import load_dataset
from transformers import pipeline

INPUT_FOLDER = "data/preselect_80"
OUTPUT_FILE = "superfiltering/output/filtered_dataset.json"
MODEL_NAME = "nvidia/quality-classifier-deberta"
THRESHOLD = 0.5

# Load model
classifier = pipeline("text-classification", model=MODEL_NAME, device=-1)

all_records = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(INPUT_FOLDER, filename)
        print(f"Processing {file_path}...")
        dataset = load_dataset(
            "json",
            data_files={"train": file_path},
            split="train",
        )
        texts = dataset["text"] if "text" in dataset.column_names else dataset["content"]
        preds = classifier(texts, truncation=True, batch_size=16)
        keep_indices = [i for i, p in enumerate(preds) if (p["label"] == "POSITIVE" and p["score"] >= THRESHOLD)]
        filtered = dataset.select(keep_indices)
        all_records.extend(filtered)

# Save everything to a single JSONL file
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")

print(f"✅ Saved filtered dataset to {OUTPUT_FILE}")
