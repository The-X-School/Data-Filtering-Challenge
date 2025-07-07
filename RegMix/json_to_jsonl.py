#lindsey made it im js stealing it
import json
import argparse
import os

def jsonl_to_json(jsonl_path, json_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Converted {jsonl_path} to {json_path}")


def json_to_jsonl(json_path, jsonl_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects to convert to JSONL.")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Converted {json_path} to {jsonl_path}")

# python json_to_jsonl.py data.json data.jsonl
# python json_to_jsonl.py data.jsonl data.json

def main():
    parser = argparse.ArgumentParser(description="Convert between JSON and JSONL formats.")
    parser.add_argument('input', help="Input file path (.json or .jsonl)")
    parser.add_argument('output', help="Output file path (.json or .jsonl)")
    args = parser.parse_args()

    input_ext = os.path.splitext(args.input)[1].lower()
    output_ext = os.path.splitext(args.output)[1].lower()

    if input_ext == '.jsonl' and output_ext == '.json':
        jsonl_to_json(args.input, args.output)
    elif input_ext == '.json' and output_ext == '.jsonl':
        json_to_jsonl(args.input, args.output)
    else:
        raise ValueError("Unsupported conversion. Use .jsonl -> .json or .json -> .jsonl")

if __name__ == '__main__':
    main()