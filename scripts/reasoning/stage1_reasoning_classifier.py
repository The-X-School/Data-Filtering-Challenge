_#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CACHE_DIR = "/root/.cache"
INPUT_FILE = "output/function_calling/climblab_10k_shuffled.json"
OUTPUT_FILE = "output/reasoning/stage1_reasoning_candidates.jsonl"
MAX_WORKERS = 8
BATCH_SIZE = 4

# --- Prompt Template for Stage 1 Reasoning Classification ---
PROMPT_TEMPLATE = """<|im_start|>user
Your task is to determine if the following text contains a problem, a question, or a puzzle that requires logical reasoning or mathematical calculation to solve.
- Look for questions that start with "what if", "why", "how many".
- Look for mathematical word problems or logic puzzles.
- The text should present a scenario that requires thinking, not just a simple factual answer.
- Ignore marketing copy, lists, conversational chit-chat, or code snippets.

Based on these criteria, does the text below contain a reasoning task? Answer with only "Yes" or "No".

Text:
```{text_sample}```<|im_end|>
<|im_start|>assistant
"""

# --- Model Loading ---
def load_model_and_tokenizer(model_path, cache_dir):
    """Loads the model and tokenizer with specified configurations."""
    print(f"Loading model: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
        )

        # Set padding side to left for batched generation with decoder-only models
        tokenizer.padding_side = 'left'
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# --- Batch Inference ---
def classify_batch(batch_texts, model, tokenizer):
    """Classifies a batch of texts using the model."""
    prompts = [PROMPT_TEMPLATE.format(text_sample=text) for text in batch_texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        # The generated output will include the prompt, so we need to decode from the end of the prompt
        input_len = inputs['input_ids'].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

    results = []
    for answer in decoded_outputs:
        results.append("Yes" in answer)
    return results

# --- Main Logic ---
def main():
    """Main function to run the classification process."""
    parser = argparse.ArgumentParser(description="Stage 1 Funnel Classifier for Reasoning Tasks")
    parser.add_argument("--input_file", type=str, default=INPUT_FILE)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, CACHE_DIR)
    
    print(f"Reading data from {args.input_file}")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        sys.exit(1)

    # Correctly access the list of instances from the loaded JSON object
    instances = data.get("instances", [])
    if not instances:
        print("Error: Could not find 'instances' key in the input JSON file, or it's empty.")
        sys.exit(1)
        
    all_texts = [item.get("text", "") for item in instances if item.get("text")]
    print(f"Found {len(all_texts)} text samples to process.")
    
    yes_count = 0
    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Create batches
            batches = [all_texts[i:i + args.batch_size] for i in range(0, len(all_texts), args.batch_size)]
            
            future_to_batch = {executor.submit(classify_batch, batch, model, tokenizer): i for i, batch in enumerate(batches)}
            
            print("Starting classification...")
            progress_bar = tqdm(total=len(batches), desc="Classifying Batches")
            
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                # Map results back to the original data items
                original_batch_data = instances[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
                
                try:
                    results = future.result()
                    for item, is_yes in zip(original_batch_data, results):
                        if is_yes:
                            f_out.write(json.dumps(item) + '\n')
                            yes_count += 1
                except Exception as exc:
                    print(f'Batch {batch_index} generated an exception: {exc}')
                finally:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"Accepted": yes_count})
            
            progress_bar.close()

    print("\nClassification complete.")
    print(f"Total samples accepted: {yes_count}")
    print(f"Candidate data saved to {args.output_file}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main() 