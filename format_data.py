import json
import sys
import os

if len(sys.argv) != 2:
    print(f"Usage: python {os.path.basename(__file__)} <input_file.json|jsonl>")
    sys.exit(1)

input_path = sys.argv[1]

# Read the input file
instances = []
if input_path.endswith('.jsonl'):
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
else:
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        # If already in LMFlow format, do nothing
        if isinstance(data, dict) and 'type' in data and 'instances' in data:
            print(f"File {input_path} is already in LMFlow format. No changes made.")
            sys.exit(0)
        # If it's a list, treat as instances
        elif isinstance(data, list):
            instances = data
        else:
            print(f"Unrecognized JSON structure in {input_path}.")
            sys.exit(1)

# Wrap in LMFlow format
data = {
    "type": "text_only",
    "instances": instances
}

# Overwrite the original file
with open(input_path, 'w', encoding='utf-8') as fout:
    json.dump(data, fout, ensure_ascii=False, indent=2)

print(f"Reformatted {input_path} to LMFlow format.") 