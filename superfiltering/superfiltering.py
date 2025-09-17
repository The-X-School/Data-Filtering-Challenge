import argparse
import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def filter_cluster_file(input_path, output_path, model_name="distilbert-base-uncased-finetuned-sst-2-english", threshold=0.5):
    print(f"\n=== Processing cluster: {os.path.basename(input_path)} ===")
    print(f"📥 Loading cluster: {os.path.basename(input_path)} ...")

    # Load .jsonl as Dataset
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)

    # Load tokenizer and model
    print("🤖 Loading model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = dataset["text"] if "text" in dataset.column_names else dataset["content"]

    keep_indices = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            probs = F.softmax(outputs.logits, dim=-1)
            # Assuming label 1 is "keep"
            for j, p in enumerate(probs):
                if p[1] > threshold:
                    keep_indices.append(i + j)

    filtered = dataset.select(keep_indices)

    # Save filtered cluster as .jsonl
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for row in filtered:
            f.write(json.dumps(row) + "\n")
    print(f"💾 Saved filtered cluster to {output_path} ({len(filtered)} samples)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Folder with cluster JSONL files")
    parser.add_argument("--output_folder", required=True, help="Folder to save filtered clusters")
    parser.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english", help="Filtering model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for keeping samples")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    cluster_files = [f for f in os.listdir(args.input_folder) if f.endswith(".jsonl")]
    cluster_files.sort()
    for idx, filename in enumerate(cluster_files, 1):
        input_path = os.path.join(args.input_folder, filename)
        output_path = os.path.join(args.output_folder, filename)
        print(f"\n=== Processing cluster {idx}: {filename} ===")
        filter_cluster_file(input_path, output_path, args.model, args.threshold)

if __name__ == "__main__":
    main()
