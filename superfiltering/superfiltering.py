import argparse
import os
from datasets import load_dataset
from nemo.collections.nlp.models import TextClassificationModel

def filter_cluster_file(input_path, output_path, model_name="nvidia/quality-classifier-deberta"):
    print(f"📥 Loading {input_path} ...")
    dataset = load_dataset("json", data_files=input_path)["train"]

    # Load classifier
    classifier = TextClassificationModel.from_pretrained(model_name)
    
    # Determine text column
    if "text" in dataset.column_names:
        texts = dataset["text"]
    elif "content" in dataset.column_names:
        texts = dataset["content"]
    else:
        raise ValueError("No text column found in dataset (expected 'text' or 'content')")

    # Predict quality scores
    scores = classifier.predict(texts)

    # Keep high-quality items (score > 0.5)
    keep_indices = [i for i, s in enumerate(scores) if s[1] > 0.5]
    filtered = dataset.select(keep_indices)

    # Save filtered cluster
    filtered.to_json(output_path)
    print(f"💾 Saved filtered cluster to {output_path} ({len(filtered)} samples)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Folder with cluster JSON files")
    parser.add_argument("--output_folder", required=True, help="Folder to save filtered clusters")
    parser.add_argument("--model", default="nvidia/quality-classifier-deberta", help="Filtering model")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Loop over input clusters
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, filename)
            filter_cluster_file(input_path, output_path, args.model)

if __name__ == "__main__":
    main()
