"""
Script to calculate the perplexity of a fine-tuned model.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
from datasets import load_dataset

def calculate_perplexity(model_path, dataset_path):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    # Load a dataset for evaluation
    eval_dataset = load_dataset("json", data_files=dataset_path, split="train[:20]")  # Use a small subset for speed

    # Calculate loss
    total_loss = 0.0
    total_tokens = 0

    for example in eval_dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    model_path = "my-llama-400m-climblab"  # Path to your saved model
    dataset_path = "data/climblab_sample/climblab_sample.jsonl"  # Path to your dataset
    calculate_perplexity(model_path, dataset_path)