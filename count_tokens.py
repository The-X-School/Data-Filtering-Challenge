# takes input in lmflow format with "type" and "instance" labels
# also takes in input as a normal jsonl file
# counts the number of tokens using the Llama tokenizer

import json
import argparse
from transformers import LlamaTokenizer
from tqdm import tqdm
import os

def count_tokens(input_file):
    tokenizer = LlamaTokenizer.from_pretrained("data4elm/Llama-400M-12L")
    total_tokens = 0

    def count_lines(path):
        with open(path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # file with lmflow format
        joined = "".join(lines).strip()
        is_lmflow = joined.startswith("{") and '"instances"' in joined

        if is_lmflow:
            full_data = json.loads(joined)

            if not isinstance(full_data, dict) or "instances" not in full_data:
                print("Invalid LMFlow structure: missing 'instances'")
                return

            for instance in tqdm(full_data["instances"], desc="Processing LMFlow instances", unit="text"):
                text = instance.get("text")
                if isinstance(text, str):
                    tokens = tokenizer.encode(text)
                    total_tokens += len(tokens)
                else:
                    print("Warning: missing or invalid 'text' field.")

        else:
            # file as a 
            with open(input_file, 'r', encoding='utf-8') as infile:
                total_lines = count_lines(input_file)
                for line in tqdm(infile, total=total_lines, desc="Processing JSONL lines", unit="line"):
                    try:
                        data = json.loads(line)
                        text = data.get("text")
                        if isinstance(text, str):
                            tokens = tokenizer.encode(text)
                            total_tokens += len(tokens)
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSONL line: {e}")
                        continue

        print(f"\nTotal number of tokens: {total_tokens}")

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in a JSON or JSONL file.")
    parser.add_argument("input_file", help="Path to the input file (JSON or JSONL)")
    args = parser.parse_args()

    count_tokens(args.input_file)