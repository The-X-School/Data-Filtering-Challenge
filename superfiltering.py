# superfiltering.py
import argparse
from datasets import load_dataset
from nemo.collections.nlp.data import QualityClassifier

def load_data(path):
    """Load dataset from JSON file"""
    return load_dataset("json", data_files=path)["train"]

def filter_dataset(dataset, model_name="nvidia/quality-classifier-deberta"):
    """Apply superfiltering using a small model"""
    classifier = QualityClassifier.from_pretrained(model_name)
    return classifier.filter(dataset)

def save_dataset(dataset, output_path):
    """Save filtered dataset"""
    dataset.to_json(output_path)
    print(f"✅ Saved filtered dataset to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input dataset JSON")
    parser.add_argument("--output", required=True, help="Path to save filtered dataset JSON")
    parser.add_argument("--model", default="nvidia/quality-classifier-deberta", help="Filtering model")
    args = parser.parse_args()

    print("📥 Loading dataset...")
    dataset = load_data(args.input)
    print(f"Dataset size: {len(dataset)}")

    print("🔍 Filtering dataset...")
    filtered = filter_dataset(dataset, args.model)
    print(f"Filtered size: {len(filtered)}")

    print("💾 Saving...")
    save_dataset(filtered, args.output)

if __name__ == "__main__":
    main()
