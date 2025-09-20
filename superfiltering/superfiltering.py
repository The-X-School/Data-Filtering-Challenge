import os
import json
import argparse
from datasets import Dataset
from transformers import GPT2Model, GPT2TokenizerFast
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import tqdm

os.environ["USE_TF"] = "0"

MAX_LEN = 1024  # GPT-2 maximum context length


def safe_load_jsonl(path):
    """Load JSONL manually and skip malformed/empty lines"""
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line {i} in {path}: {e}")
    return Dataset.from_list(records)


def extract_tokens_column(dataset, tokenizer):
    """Extract tokens or text as GPT-2 token IDs"""
    if "tokens" in dataset.column_names:
        all_tokens = []
        for x in dataset["tokens"]:
            if x is None:
                all_tokens.append([])
            elif isinstance(x, list):
                all_tokens.append(x)
            else:
                all_tokens.append(list(map(int, x)))
        return all_tokens
    elif "text" in dataset.column_names:
        return [tokenizer.encode(str(x), add_special_tokens=False) for x in dataset["text"]]
    else:
        raise ValueError(f"No 'tokens' or 'text' column found. Columns: {dataset.column_names}")


def compute_gpt2_embeddings(token_lists, model, tokenizer, device="cpu", batch_size=16):
    """Compute mean GPT-2 embeddings for token ID sequences"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []

    for i in tqdm.tqdm(range(0, len(token_lists), batch_size), desc="Computing embeddings"):
        batch_tokens = token_lists[i:i+batch_size]
        # truncate sequences longer than GPT-2 context size
        batch_tokens = [tokens[:MAX_LEN] for tokens in batch_tokens]

        batch_enc = tokenizer.pad(
            {"input_ids": batch_tokens},
            padding=True,
            return_tensors="pt"
        )
        input_ids = batch_enc["input_ids"].to(device)
        attention_mask = batch_enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeds)

    return np.vstack(embeddings)


def filter_cluster_file(input_path, output_folder, tokenizer, model, device, threshold=0.5):
    filename = os.path.basename(input_path)
    output_file = os.path.join(output_folder, f"filtered_{filename}")

    print(f"⚡ Processing {input_path} ...")
    dataset = safe_load_jsonl(input_path)
    if len(dataset) == 0:
        print(f"❌ Skipping {input_path}, no valid JSON found.")
        return []

    token_lists = extract_tokens_column(dataset, tokenizer)

    embeddings = compute_gpt2_embeddings(token_lists, model, tokenizer, device=device)

    # Use mean + threshold to filter without labels
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    mean_scores = embeddings_scaled.mean(axis=1)
    keep_indices = [i for i, s in enumerate(mean_scores) if s >= threshold]

    filtered = dataset.select(keep_indices)

    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, "w") as f:
        for record in filtered:
            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved filtered cluster to {output_file}")
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load tokenizer and model once
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()

    all_records = []
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(args.input_folder, filename)
            filtered = filter_cluster_file(file_path, args.output_folder, tokenizer, model, device, args.threshold)
            all_records.extend(filtered)

    final_output_file = os.path.join(args.output_folder, "filtered_dataset.jsonl")
    with open(final_output_file, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved complete filtered dataset to {final_output_file}")


if __name__ == "__main__":
    main()
