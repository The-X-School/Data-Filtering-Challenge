import argparse
import os
from datasets import load_dataset
from nemo.collections.nlp.models import TextClassificationModel

def filter_cluster_file(input_path, output_path, model_name="nvidia/quality-classifier-deberta"):
    print(f"\n📥 Loading cluster: {os.path.basename(input_path)} ...")
    dataset = load_dataset("json", data_files=input_path)["train"]

    # Detect text column
    text_col = None
    for col in ["text", "content", "sentence", "body"]:
        if col in dataset.column_names:
            text_col = col
            break
    if text_col is None:
        raise ValueError(f"No suitable text column found in {input_path}")

    texts = dataset[text_col]

    # Load classifier
    print(f"🤖 Loading model: {model_name} ...")
    classifier = TextClassificationModel.from_pretrained(model_name)

    # Predict quality
    print(f"🔍 Filtering cluster with {len(texts)} items ...")
    predictions = classifier.predict(texts)

    # Determine which items to keep
    keep_indices = []
    for i, pred in enumerate(predictions):
        if isinstance(pred, dict):
            score = pred.get('logits', [0, 0])[1]  # fallback if logits key exists
        else:
            score = pred[1]  # fallback if list
        if score > 0.5:
            keep_indices.append(i)

    filtered = dataset.select(keep_indices)

    # Save filtered cluster
    filtered.to_json(output_path)
    print(f"💾 Saved filtered cluster: {os.path.basename(output_path)} | "
          f"original: {len(dataset)}, filtered: {len(filtered)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Folder with cluster JSON files")
    parser.add_argument("--output_folder", required=True, help="Folder to save filtered clusters")
    parser.add_argument("--model", default="nvidia/quality-classifier-deberta", help="Filtering model")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    files_found = 0
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".json"):
            files_found += 1
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, filename)

            print(f"\n=== Processing cluster {files_found}: {filename} ===")
            filter_cluster_file(input_path, output_path, args.model)
            print(f"✅ Finished cluster {filename}")

    if files_found == 0:
        print("⚠️ No JSON files found in input folder.")

    print("\n🎉 All clusters processed.")

if __name__ == "__main__":
    main()
