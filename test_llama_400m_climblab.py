#!/usr/bin/env python3
"""
Script to test the fine-tuned Llama-400M-12L model on a sample prompt.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to the fine-tuned model directory
model_path = "data4elm/Llama-400M-12L"

# Load the tokenizer and model
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")

# Sample prompt
test_prompt = "Answer in a single sentence: What is the capital of France?"
inputs = tokenizer(test_prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Generate output
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        num_return_sequences=1,
        do_sample=False
    )
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Prompt: {test_prompt}")
print(f"Model output: {output_text}") 