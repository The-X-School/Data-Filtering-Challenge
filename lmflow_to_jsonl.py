import json
import sys

def lmflow_to_jsonl(input_file, output_file):
    """Convert LMFlow format back to JSONL format."""
    with open(input_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    
    if not isinstance(data, dict) or 'instances' not in data:
        print(f"Error: {input_file} is not in LMFlow format")
        return
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for instance in data['instances']:
            if 'text' in instance:
                # Write each instance as a separate JSON line
                fout.write(json.dumps({'text': instance['text']}, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(data['instances'])} instances to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python lmflow_to_jsonl.py <input_lmflow.json> <output.jsonl>")
        sys.exit(1)
    
    lmflow_to_jsonl(sys.argv[1], sys.argv[2]) 