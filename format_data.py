import json
import sys
import os

def convert_to_lmflow_format(file_path):
    """Convert a file with multiple JSON objects to LMFlow format."""
    instances = []
    
    try:
        # First, try to read as a single JSON object
        with open(file_path, 'r', encoding='utf-8') as fin:
            try:
                data = json.load(fin)
                # If it's already in LMFlow format, do nothing
                if isinstance(data, dict) and 'type' in data and 'instances' in data:
                    print(f"File {file_path} is already in LMFlow format. No changes made.")
                    return
                # If it's a list, treat as instances
                elif isinstance(data, list):
                    instances = data
                else:
                    print(f"Unrecognized JSON structure in {file_path}.")
                    return
            except json.JSONDecodeError:
                # If single JSON fails, try reading line by line (JSONL format)
                fin.seek(0)  # Reset file pointer
                for line_num, line in enumerate(fin, 1):
                    line = line.strip()
                    if line:
                        try:
                            instances.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num}: {e}")
                            return
        
        # Clean instances: remove 'id' and 'metadata' fields, keep only 'text'
        cleaned_instances = []
        for instance in instances:
            if isinstance(instance, dict) and 'text' in instance:
                # Keep only the 'text' field
                cleaned_instances.append({'text': instance['text']})
            else:
                # If no 'text' field, skip this instance
                print(f"Warning: Skipping instance without 'text' field: {instance}")
        
        # Wrap in LMFlow format
        data = {
            "type": "text_only",
            "instances": cleaned_instances
        }
        
        # Overwrite the original file
        with open(file_path, 'w', encoding='utf-8') as fout:
            json.dump(data, fout, ensure_ascii=False, indent=2)
        
        print(f"Successfully reformatted {file_path} to LMFlow format with {len(cleaned_instances)} instances.")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <input_file.json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    convert_to_lmflow_format(input_path) 