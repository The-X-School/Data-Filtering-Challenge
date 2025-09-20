import os
import json
from datasets import load_dataset
from transformers import pipeline

INPUT_FOLDER = "/home/ubuntu/Data-Filtering-Challenge/data/preselect_80"
OUTPUT_FOLDER = "/home/ubuntu/Data-Filtering-Challenge/superfiltering/output"
MODEL_NAME = "nvidia/quality-classifier-deberta"
THRESHOLD = 0.5

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Force PyTorch (avoid TensorFlow errors)
import os
os.environ["USE_TF"] = "0"

# Load model once
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=-1,
    framework="pt"
)

def filter_local_file(file_path, output_folder, classifier, threshold=0.5):
    print(f"⚡ Processing {file_path} ...")
    
    dataset = load_dataset(
        "json",
        data_files={"train": file_path},
        split="train"
    )
    
    texts = dataset["text"] if "text" in dataset.column_names else dataset["content"]
    preds = classifier(texts, truncation=True, batch_size=16)
    
    keep_indices = [i for i, p in enumerate(preds) if (p["label"] == "POSITIVE" and p["score"] >= threshold)]
    filtered = dataset.select(keep_indices)
    
    # Save filtered cluster
    output_file = os.path.join(output_folder, f"filtered_{os.path.basename(file_path)}")
    with open(output_file, "w") as f:
        for record in filtered:
            f.write(json.dumps(record) + "\n")
    
    print(f"✅ Saved filtered cluster to {output_file}")
    return filtered

# Run on all JSONL files in INPUT_FOLDER
all_records = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(INPUT_FOLDER, filename)
        filtered = filter_local_file(file_path, OUTPUT_FOLDER, classifier, THRESHOLD)
        all_records.extend(filtered)

# Save everything to a single JSONL file
final_output_file = os.path.join(OUTPUT_FOLDER, "filtered_dataset.json")
with open(final_output_file, "w") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")

print(f"✅ Saved complete filtered dataset to {final_output_file}")
