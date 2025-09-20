import os
import json
import argparse
from datasets import load_dataset
from transformers import GPT2Model, GPT2TokenizerFast
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import tqdm

os.environ["USE_TF"] = "0"

def extract_tokens_column(dataset):
    """Extract tokens or text as GPT-2 token IDs"""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
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
    embeddings = []
    for i in tqdm.tqdm(range(0, len(token_lists), batch_size), desc="Computing embeddings"):
        batch_tokens = token_lists[i:i+batch_size]
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

def filter_cluster_file(input_path, output_folder, threshold=0.5):
    filename = os.path.basename(input_path)
    output_file = os.path.join(output_folder, f"filtered_{filename}")

    print(f"⚡ Processing {input_path} ...")
    dataset = load_dataset("json", data_files={"train": input_path}, split="train")
    token_lists = extract_tokens_column(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()

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

    all_records = []
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(args.input_folder, filename)
            filtered = filter_cluster_file(file_path, args.output_folder, args.threshold)
            all_records.extend(filtered)

    final_output_file = os.path.join(args.output_folder, "filtered_dataset.jsonl")
    with open(final_output_file, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved complete filtered dataset to {final_output_file}")

if __name__ == "__main__":
    main()
