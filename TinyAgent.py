import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set paths and device
MODEL_DIR = "./tinyagent-1.1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Try to load the local model first
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading local model: {e}")
    print("Trying to load a public model instead...")
    
    # Fallback to a public model
    try:
        model_name = "microsoft/DialoGPT-medium"  # A smaller, public model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Loaded {model_name} successfully!")
    except Exception as e2:
        print(f"Error loading public model: {e2}")
        exit(1)

# Run a test generation
prompt = "How does photosynthesis work?"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
outputs = model.generate(**inputs, max_new_tokens=100)

print("\n--- Generated Response ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("--------------------------")
