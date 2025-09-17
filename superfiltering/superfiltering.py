import argparse
import os
import json
from datasets import Dataset
from nemo.collections.nlp.models import TextClassificationModel

def filter_cluster_file(input_path, output_path, model_name="nvidia/quality-classifier-deberta"):
    print(f"\n=== Processing cluster: {os.path.basename(input_path)} ===")
    
    # Load JSONL manually
    print(f"📥 Loading cluster: {os.path.basename(input_path)} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if not data:
        print(f"⚠️ Cluster {input_path} is empty, skipping.")
        return

    # Detect text column
    text_col = None
    for col in ["text", "content", "sentence", "body"]:
        if col in data[0]:
            text_col = col
            break
    if text_col is None:
        raise ValueError(f"No suitable text column found in {input_path}")

    texts = [item[text_col] for item in data]

    # Load classifier
    print(f"🤖 Loading model: {model_name} ...")
    classifier = TextClassificationModel.from_pretrained(model_name)

    # Predict quality
    print(f"🔍 Filtering cluster with {len(texts)} items ...")
    predictions = classifier.predict(texts)

    # Keep only high-quality items
    keep_indices = []
    for i, pred in enumerate(predictions):
        # Some models return dict with logits, some return list
        if isinstance(pred, dict):
            score = pred.get('logits', [0, 0])[1]
        else:
            score = pred[1]
        if score > 0.5:
            keep_indices.append(i)

    filtered_data = [data[i] for i in keep_indices]

    # Save filtered cluster as JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"💾 Saved filtered cluster: {os.path.basename(output_path)} | "
          f"original: {len(data)}, filtered: {len(filtered_data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Folder with cluster JSONL files")
    parser.add_argument("--output_folder", required=True, help="Folder to save filtered clusters")
    parser.add_argument("--model", default="nvidia/quality-classifier-deberta", help="Filtering model")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    cluster_files = [f for f in os.listdir(args.input_folder) if f.endswith(".jsonl")]
    if not cluster_files:
        print(f"⚠️ No .jsonl files found in {args.input_folder}")
        return

    for idx, filename in enumerate(cluster_files, start=1):
        input_path = os.path.join(args.input_folder, filename)
        output_path = os.path.join(args.output_folder, filename)
        print(f"\n=== Processing cluster {idx}: {filename} ===")
        filter_cluster_file(input_path, output_path, args.model)


if __name__ == "__main__":
    main()
