import os
import json
import argparse
from datasets import load_dataset
from transformers import pipeline

# Force transformers to use PyTorch
os.environ["USE_TF"] = "0"

def filter_cluster_file(input_path, output_folder, model_name, threshold=0.5):
    filename = os.path.basename(input_path)
    output_file = os.path.join(output_folder, f"filtered_{filename}")

    print(f"⚡ Processing {input_path} ...")

    dataset = load_dataset(
        "json",
        data_files={"train": input_path},
        split="train"
    )

    # Ensure we always extract string text safely
    if "text" in dataset.column_names:
        texts = [str(x) for x in dataset["text"]]
    elif "content" in dataset.column_names:
        texts = [str(x) for x in dataset["content"]]
    else:
        raise ValueError(
            f"❌ No 'text' or 'content' column found in {input_path}. "
            f"Available columns: {dataset.column_names}"
        )

    print(f"🤖 Loading model: {model_name}")
    classifier = pipeline("text-classification", model=model_name, device=-1)

    print(f"⚡ Running classification on {len(texts)} samples...")
    preds = classifier(texts, truncation=True, batch_size=16)

    keep_indices = [
        i for i, p in enumerate(preds)
        if (p["label"] == "POSITIVE" and p["score"] >= threshold)
    ]
    filtered = dataset.select(keep_indices)

    # Save filtered cluster
    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, "w") as f:
        for record in filtered:
            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved filtered cluster to {output_file}")
    return filtered

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing cluster JSONL files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save filtered clusters")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    all_records = []

    for filename in os.listdir(args.input_folder):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(args.input_folder, filename)
            filtered = filter_cluster_file(file_path, args.output_folder, args.model, args.threshold)
            all_records.extend(filtered)

    # Save everything to a single JSONL file
    final_output_file = os.path.join(args.output_folder, "filtered_dataset.jsonl")
    with open(final_output_file, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved complete filtered dataset to {final_output_file}")

if __name__ == "__main__":
    main()
