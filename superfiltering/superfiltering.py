import os
import json
from datasets import load_dataset
from transformers import pipeline

INPUT_FOLDER = "data/preselect_80"
OUTPUT_FOLDER = "superfiltering/output"
MODEL_NAME = "nvidia/quality-classifier-deberta"
THRESHOLD = 0.5

# Force PyTorch and avoid TensorFlow
os.environ["USE_TF"] = "0"

# Load model with PyTorch
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=-1,
    framework="pt"  # force PyTorch
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
        keep_indices = [i for i, p in enumerate(preds) if (p["label"] == "POSITIVE" and p["score"] >= THRESHOLD)]
        filtered = dataset.select(keep_indices)

        all_records.extend(filtered)

        output_file = os.path.join(OUTPUT_FOLDER, f"filtered_{filename}")
        with open(output_file, "w") as f:
            for record in filtered:
                f.write(json.dumps(record) + "\n")

        print(f"✅ Saved filtered cluster to {output_file}")

# Save everything to a single JSONL file
final_output_file = os.path.join(OUTPUT_FOLDER, "filtered_dataset.json")
with open(final_output_file, "w") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")

print(f"✅ Saved filtered dataset to {final_output_file}")
