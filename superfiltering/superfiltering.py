# superfiltering.py (updated for modern NeMo)
import argparse
import os
from datasets import load_dataset, concatenate_datasets
from nemo.collections.nlp.models import TextClassificationModel

def load_clusters(folder):
    """Load all cluster JSON files into a single dataset"""
    datasets_list = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            path = os.path.join(folder, filename)
            print(f"📥 Loading {path} ...")
            ds = load_dataset("json", data_files=path)["train"]
            datasets_list.append(ds)

    if not datasets_list:
        raise ValueError(f"No JSON files found in {folder}")

    combined = concatenate_datasets(datasets_list)
    print(f"✅ Combined dataset size: {len(combined)}")
    return combined

def filter_dataset(dataset, model_name="nvidia/quality-classifier-deberta"):
    classifier = TextClassificationModel.from_pretrained(model_name)
    texts = dataset["text"] if "text" in dataset.column_names else dataset["content"]
    scores = classifier.predict(texts)
    keep_indices = [i for i, s in enumerate(scores) if s[1] > 0.5]  # adjust threshold
    filtered = dataset.select(keep_indices)
    return filtered

def save_dataset(dataset, output_path):
    dataset.to_json(output_path)
    print(f"💾 Saved filtered dataset to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Path to folder containing cluster JSON files")
    parser.add_argument("--output", required=True, help="Path to save filtered dataset JSON")
    parser.add_argument("--model", default="nvidia/quality-classifier-deberta", help="Filtering model")
    args = parser.parse_args()

    # Load data
    dataset = load_clusters(args.input_folder)

    # Filter data
    print("🔍 Filtering...")
    filtered = filter_dataset(dataset, args.model)
    print(f"✅ Filtered dataset size: {len(filtered)}")

    # Save
    save_dataset(filtered, args.output)

if __name__ == "__main__":
    main()
