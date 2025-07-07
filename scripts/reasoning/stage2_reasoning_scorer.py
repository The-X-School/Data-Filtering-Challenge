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
# Primary "Judge" Model: Llama-3-8B is the preferred choice.
MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
# Fallback Model: A strong, open-access model like Qwen1.5-7B-Chat.
# MODEL_PATH = "Qwen/Qwen1.5-7B-Chat"

CACHE_DIR = "/root/.cache"
INPUT_FILE = "output/reasoning/stage1_reasoning_candidates.jsonl"
OUTPUT_FILE = "output/reasoning/stage2_reasoning_scored.jsonl"
BATCH_SIZE = 16  # Increased batch size for single-threaded GPU utilization

# --- Few-Shot Prompt parts ---
SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert data quality analyst. Your mission is to curate a high-quality dataset for training a language model on reasoning tasks. Evaluate a given text on a single, unified scale of "Reasoning Potential" from 1 to 5. Your entire output must be only the JSON object.

# Evaluation Criteria & Scoring
- **Score 1 (Not a reasoning task):** Purely conversational, a list, code, marketing copy, or irrelevant. No question/problem structure.
- **Score 2 (Weak reasoning signal):** A simple question answered by direct fact retrieval (e.g., "What is the capital of France?").
- **Score 3 (Moderate reasoning potential):** A simple problem requiring one or two logical steps.
- **Score 4 (Strong reasoning potential):** A clear problem requiring multiple logical steps, calculation, or structured thinking.
- **Score 5 (Excellent reasoning potential):** A high-quality, complex reasoning problem (e.g., a multi-step math word problem, a logic puzzle).<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

EXAMPLE_1 = """
# Example 1 (Excellent)
Input Text:
```
A grocery store sells apples in bags of 5 and oranges in bags of 3. If a customer buys 4 bags of apples and 2 bags of oranges, how many total pieces of fruit did she buy?
```

Your Task:
Provide your evaluation in a valid JSON format.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{
  "reasoning_score": 5,
  "justification": "This is a classic multi-step math word problem, perfect for training reasoning."
}<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

EXAMPLE_2 = """
# Example 2 (Poor)
Input Text:
```
I was thinking of going to the beach today, but the weather looks a bit cloudy. Maybe I'll just stay in and read a book. Do you have any good book recommendations?
```

Your Task:
Provide your evaluation in a valid JSON format.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{
  "reasoning_score": 1,
  "justification": "This is conversational text and does not contain a verifiable reasoning problem."
}<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

# --- Utility Functions ---
def load_model_and_tokenizer(model_path, cache_dir, token):
    """Loads the model and tokenizer, using a token for gated models."""
    print(f"Loading model: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            token=token,
        )
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def parse_json_from_response(response_text):
    """
    Extracts a JSON object from the model's response text robustly.
    It looks for the first '{' and the last '}' to define the JSON block.
    """
    try:
        # Find the first '{' and the last '}' to extract the JSON block
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end+1]
            return json.loads(json_str)
        return None
    except (json.JSONDecodeError, TypeError):
        # Return None if parsing fails or input is not a string
        return None

# --- Batch Inference ---
def score_batch(batch_items, model, tokenizer):
    """Scores a batch of text items."""
    prompts = []
    for item in batch_items:
        text_sample = item.get("text", "")
        # Manually concatenate strings to build the prompt, avoiding .format() entirely.
        user_turn = (
            "\n# Your Turn\nInput Text:\n```\n"
            + text_sample
            + "\n```\n\nYour Task:\nProvide your evaluation in a valid JSON format.\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        full_prompt = SYSTEM_PROMPT + EXAMPLE_1 + EXAMPLE_2 + user_turn
        prompts.append(full_prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    
    input_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    decoded_outputs = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    
    results = []
    for i, response in enumerate(decoded_outputs):
        parsed_json = parse_json_from_response(response)
        item = batch_items[i]
        if parsed_json and isinstance(parsed_json.get("reasoning_score"), int):
            item.update(parsed_json)
            item['stage2_status'] = 'success'
        else:
            item['stage2_status'] = 'parse_failed'
            item['model_output'] = response
        results.append(item)
            
    return results

# --- Main Logic ---
def main():
    """Main function to run the scoring process."""
    parser = argparse.ArgumentParser(description="Stage 2 Scorer for Reasoning Tasks using a powerful judge model.")
    parser.add_argument("--input_file", type=str, default=INPUT_FILE)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--token", type=str, default=os.getenv("HUGGING_FACE_TOKEN"), help="Hugging Face token for gated models.")
    args = parser.parse_args()

    if not args.token:
        print("Warning: Hugging Face token not found. Provide it via --token or HUGGING_FACE_TOKEN env var for gated models.")

    model, tokenizer = load_model_and_tokenizer(args.model_path, CACHE_DIR, args.token)
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        sys.exit(1)

    print(f"Found {len(data)} candidate samples to score.")
    
    successful_scores = 0
    failed_parses = 0
    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        print(f"Starting scoring in single-threaded mode with batch size {args.batch_size}...")
        for i in tqdm(range(0, len(data), args.batch_size), desc="Scoring Batches"):
            batch = data[i:i + args.batch_size]
            try:
                scored_results = score_batch(batch, model, tokenizer)
                for item in scored_results:
                    f_out.write(json.dumps(item) + '\n')
                    if item.get('stage2_status') == 'success':
                        successful_scores += 1
                    else:
                        failed_parses += 1
            except Exception as e:
                print(f"FATAL: An unrecoverable error occurred in batch starting at index {i}: {e}")
                for item in batch:
                    item['stage2_status'] = 'batch_processing_exception'
                    f_out.write(json.dumps(item) + '\n')
                failed_parses += len(batch)
    
    print("\nScoring complete.")
    print(f"Total samples successfully scored: {successful_scores}")
    print(f"Total samples failed to parse or process: {failed_parses}")
    print(f"Scored data saved to {args.output_file}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main() 