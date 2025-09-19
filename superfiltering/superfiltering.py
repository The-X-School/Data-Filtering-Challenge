import os
import json
from datasets import load_dataset
from transformers import pipeline

INPUT_FOLDER = "data/preselect_80"
OUTPUT_FOLDER = "superfiltering/output"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "filtered_dataset.jsonl")
MODEL_NAME = "nvidia/quality-classifier-deberta"
THRESHOLD = 0.5

# Load model (force PyTorch)
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=-1,
    framework="pt"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
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
        keep_indices = [
            i for i, p in enumerate(preds)
            if (p["label"] == "POSITIVE" and p["score"] >= THRESHOLD)
        ]
        filtered = dataset.select(keep_indices)
        all_records.extend(filtered)

        # Save filtered cluster
        output_file = os.path.join(OUTPUT_FOLDER, f"filtered_{filename}")
        with open(output_file, "w") as f:
            for record in filtered:
                f.write(json.dumps(record) + "\n")

        print(f"✅ Saved filtered cluster to {output_file}")

# Save combined dataset
with open(OUTPUT_FILE, "w") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")

print(f"✅ Saved combined filtered dataset to {OUTPUT_FILE}")
