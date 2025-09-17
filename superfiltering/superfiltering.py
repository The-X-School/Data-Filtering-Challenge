import os
import json
import boto3
import argparse
from datasets import load_dataset
from transformers import pipeline

s3 = boto3.client("s3")

def filter_cluster_file(input_bucket, input_key, output_bucket, output_prefix, model_name, threshold=0.5):
    local_input = f"/tmp/{os.path.basename(input_key)}"
    local_output = f"/tmp/{os.path.basename(input_key)}"

    print(f"📥 Downloading {input_key} from s3://{input_bucket} ...")
    s3.download_file(input_bucket, input_key, local_input)

    print(f"📥 Loading dataset from {local_input} ...")
    dataset = load_dataset(
        "json",
        data_files={"train": local_input},
        split="train",
        cache_dir="/tmp/hf_cache"
    )

    print(f"🤖 Loading model: {model_name}")
    classifier = pipeline("text-classification", model=model_name, device=-1)

    texts = dataset["text"] if "text" in dataset.column_names else dataset["content"]

    print(f"⚡ Running classification on {len(texts)} samples...")
    preds = classifier(texts, truncation=True, batch_size=16)

    keep_indices = [i for i, p in enumerate(preds) if (p["label"] == "POSITIVE" and p["score"] >= threshold)]
    filtered = dataset.select(keep_indices)

    # Save to JSONL
    print(f"💾 Saving filtered results to {local_output} ({len(filtered)} samples)")
    with open(local_output, "w") as f:
        for record in filtered.to_dict():
            f.write(json.dumps(record) + "\n")

    # Upload back to S3
    output_key = os.path.join(output_prefix, os.path.basename(input_key))
    print(f"☁️ Uploading {local_output} to s3://{output_bucket}/{output_key}")
    s3.upload_file(local_output, output_bucket, output_key)


def lambda_handler(event, context):
    """
    event example:
    {
      "input_bucket": "my-input-bucket",
      "input_keys": ["clusters/cluster1.jsonl", "clusters/cluster2.jsonl"],
      "output_bucket": "my-output-bucket",
      "output_prefix": "filtered",
      "model": "distilbert-base-uncased-finetuned-sst-2-english",
      "threshold": 0.5
    }
    """
    input_bucket = event["input_bucket"]
    input_keys = event["input_keys"]
    output_bucket = event["output_bucket"]
    output_prefix = event.get("output_prefix", "filtered")
    model_name = event.get("model", "distilbert-base-uncased-finetuned-sst-2-english")
    threshold = float(event.get("threshold", 0.5))

    for i, key in enumerate(input_keys, start=1):
        print(f"\n=== Processing cluster {i}: {key} ===")
        filter_cluster_file(input_bucket, key, output_bucket, output_prefix, model_name, threshold)

    return {"status": "done", "processed_files": input_keys}
