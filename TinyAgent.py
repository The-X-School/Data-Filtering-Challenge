# filter_with_tinyagent.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

MODEL_DIR = "/Users/zoesun/Downloads/Data-Filtering-Challenge/tinyagent-1.1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & tokenizer from disk without connecting to Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Define filter function
def is_instruction_useful(instruction):
    prompt = f"Decide whether the following instruction is useful for AI training. Reply only with 'Yes' or 'No'.\nInstruction: {instruction}"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    return "yes" in response and "no" not in response

# Filter data from climblab_raw.jsonl
filtered = []
with open("climblab_raw.jsonl", "r") as infile:
    for i, line in enumerate(infile):
        try:
            item = json.loads(line)
            instruction = item.get("instruction") or item.get("prompt") or str(item)
            print(f"[{i+1}] Checking: {instruction}")

            if is_instruction_useful(instruction):
                print("✅ Kept")
                filtered.append({"instruction": instruction})
            else:
                print("❌ Rejected")
        except Exception as e:
            print(f"⚠️ Error on line {i+1}: {e}")

# Save filtered data
with open("climblab_filtered.jsonl", "w") as f:
    for item in filtered:
        f.write(json.dumps(item) + "\n")

print(f"\n✅ Saved {len(filtered)} filtered instructions to climblab_filtered.jsonl")
