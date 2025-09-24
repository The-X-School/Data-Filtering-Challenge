import os
import json

folder_path = "superfiltering/output_preselect"

def convert_to_lmflow_format(file_path):
    instances = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, dict) and 'type' in data and 'instances' in data:
                print(f"{file_path} already LMFlow format")
                return
            elif isinstance(data, list):
                instances = data
            else:
                print(f"Skipping {file_path}, unrecognized format")
                return
        except json.JSONDecodeError:
            # Treat as JSONL
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    instances.append(json.loads(line))

    cleaned_instances = [{"text": i["text"]} for i in instances if "text" in i]

    lmflow_data = {
        "type": "text_only",
        "instances": cleaned_instances
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(lmflow_data, f, ensure_ascii=False, indent=2)
    print(f"Reformatted {file_path} with {len(cleaned_instances)} instances")

# Process all JSON / JSONL files
for fname in os.listdir(folder_path):
    if fname.endswith(".json") or fname.endswith(".jsonl"):
        convert_to_lmflow_format(os.path.join(folder_path, fname))
