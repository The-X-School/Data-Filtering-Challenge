#!/usr/bin/env python
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
CACHE_DIR = os.path.expanduser("~/.cache")
INPUT_FILE = "output/climblab_10k.jsonl"
OUTPUT_FILE = "output/stage1_result.jsonl"
MAX_WORKERS = 8
BATCH_SIZE = 64  # Increased from 4 for much faster processing

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

# --- Streaming Processing ---
def process_streaming(input_file, output_file, model, tokenizer, batch_size):
    """Process the file in streaming mode without loading everything into memory."""
    print(f"Reading data from {input_file} in streaming mode...")
    
    yes_count = 0
    total_count = 0
    batch_buffer = []
    batch_indices = []
    
    try:
        # First pass: count total lines for progress bar
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"Total lines in file: {total_lines}")
        
        # Open output file for streaming writes
        with open(output_file, 'w', encoding='utf-8') as f_out:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                progress_bar = tqdm(total=total_lines, desc="Processing lines")
                
                for line_num, line in enumerate(f_in, 1):
                    progress_bar.update(1)
                    
                    try:
                        instance = json.loads(line.strip())
                        text = instance.get("text", "")
                        
                        if text:  # Only process non-empty texts
                            batch_buffer.append(instance)
                            batch_indices.append(text)
                            
                            # Process batch when full
                            if len(batch_buffer) >= batch_size:
                                try:
                                    results = classify_batch(batch_indices, model, tokenizer)
                                    for item, is_yes in zip(batch_buffer, results):
                                        if is_yes:
                                            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                            yes_count += 1
                                except Exception as e:
                                    print(f"\nError processing batch at line {line_num}: {e}")
                                
                                total_count += len(batch_buffer)
                                progress_bar.set_postfix({"Processed": total_count, "Accepted": yes_count})
                                batch_buffer = []
                                batch_indices = []
                                
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                        continue
                
                # Process remaining items in buffer
                if batch_buffer:
                    try:
                        results = classify_batch(batch_indices, model, tokenizer)
                        for item, is_yes in zip(batch_buffer, results):
                            if is_yes:
                                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                yes_count += 1
                        total_count += len(batch_buffer)
                    except Exception as e:
                        print(f"\nError processing final batch: {e}")
                
                progress_bar.close()
                
    except Exception as e:
        print(f"Error during streaming processing: {e}")
        sys.exit(1)
    
    print("\nStreaming classification complete.")
    print(f"Total samples processed: {total_count}")
    print(f"Total samples accepted: {yes_count}")
    print(f"Candidate data saved to {output_file}")

# --- Batch Inference ---
def classify_batch(batch_texts, model, tokenizer):
    """Classifies a batch of texts using the model."""
    prompts = [PROMPT_TEMPLATE.format(text_sample=text[:300]) for text in batch_texts]  # Truncate texts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    
    with torch.no_grad():
        # The generated output will include the prompt, so we need to decode from the end of the prompt
        input_len = inputs['input_ids'].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,  # Only need Yes/No
            do_sample=False,  # Deterministic, faster
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

    results = []
    for answer in decoded_outputs:
        results.append("yes" in answer.lower())  # Case insensitive
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
    parser.add_argument("--stream", action="store_true", help="Stream processing for large files (memory efficient)")
    args = parser.parse_args()

    # Ensure we're using absolute paths
    args.input_file = os.path.abspath(args.input_file)
    args.output_file = os.path.abspath(args.output_file)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(args.model_path, CACHE_DIR)
    
    # Add streaming processing option for very large files
    if args.stream:
        print(f"Using streaming mode for memory-efficient processing...")
        process_streaming(args.input_file, args.output_file, model, tokenizer, args.batch_size)
    else:
        # Original batch processing (loads all data into memory)
        print(f"Reading data from {args.input_file}")
        instances = []
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        instance = json.loads(line.strip())
                        instances.append(instance)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            print(f"Error: Input file not found at {args.input_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)

    # For JSONL format, the data is already a list of instances
    if not instances:
        print("Error: Input JSONL file is empty or could not be read.")
        sys.exit(1)
    
    # Create list of (index, text) tuples to maintain mapping
    indexed_texts = [(i, item.get("text", "")) for i, item in enumerate(instances) if item.get("text")]
    
    if not indexed_texts:
        print("Error: No valid text samples found in the input data.")
        sys.exit(1)
        
    print(f"Found {len(indexed_texts)} text samples to process out of {len(instances)} total instances.")
    
    yes_count = 0
    results_to_write = []
    
    # Single-threaded GPU processing (more stable for GPU operations)
    print("Starting classification...")
    
    # Process in batches
    total_batches = (len(indexed_texts) + args.batch_size - 1) // args.batch_size
    progress_bar = tqdm(total=total_batches, desc="Classifying Batches")
    
    for batch_start in range(0, len(indexed_texts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(indexed_texts))
        batch_data = indexed_texts[batch_start:batch_end]
        
        # Extract indices and texts
        batch_indices = [idx for idx, _ in batch_data]
        batch_texts = [text for _, text in batch_data]
        
        try:
            # Classify the batch
            results = classify_batch(batch_texts, model, tokenizer)
            
            # Collect results
            for idx, is_yes in zip(batch_indices, results):
                if is_yes:
                    results_to_write.append(instances[idx])
                    yes_count += 1
                    
        except Exception as e:
            print(f"\nError processing batch starting at index {batch_start}: {e}")
            # Continue with next batch instead of crashing
        
        progress_bar.update(1)
        progress_bar.set_postfix({"Accepted": yes_count})
    
    progress_bar.close()
    
    # Write all results at once (safer than concurrent writes)
    print(f"\nWriting {len(results_to_write)} accepted samples to {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for item in results_to_write:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

    print("\nClassification complete.")
    print(f"Total samples processed: {len(instances)}")
    print(f"Total samples with valid text: {len(indexed_texts)}")
    print(f"Total samples accepted: {yes_count}")
    print(f"Candidate data saved to {args.output_file}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main() 