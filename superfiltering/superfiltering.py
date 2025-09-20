import os
import json
from datasets import load_dataset
from transformers import pipeline

# Folders
INPUT_FOLDER = "data/preselect_80"
OUTPUT_FOLDER = "superfiltering/output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Model and threshold
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # public, PyTorch-compatible
THRESHOLD = 0.5

# Force PyTorch and avoid TensorFlow
os.environ["USE_TF"] = "0"

# Load model on GPU
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    framework="pt",
    device=0,  # use GPU (A100)
    batch_size=16,
    truncation=True
)

all_records = []

# Process each JSONL file
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(INPUT_FOLDER, filename)
        print(f"Processing {file_path}...")

        # Load dataset
        dataset = load_dataset(
            "json",
            data_files={"train": file_path},
            split="train"
        )

        # Determine text column
        text_column = "text" if "text" in dataset.column_names else "content"
        texts = dataset[text_column]

        # Run classifier in batches
        preds = classifier(texts)

        # Filter records
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

# Save all records to a single JSONL
final_output_file = os.path.join(OUTPUT_FOLDER, "filtered_dataset.json")
with open(final_output_file, "w") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")
print(f"✅ Saved filtered dataset to {final_output_file}")
