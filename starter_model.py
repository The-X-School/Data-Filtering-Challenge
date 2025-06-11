from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "data4elm/Llama-400M-12L"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

max_length = 50
prompt = "Hello";

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=max_length)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))