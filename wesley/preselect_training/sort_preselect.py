import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Sort a JSONL file by a specified label inside metadata.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("output_file", type=str, help="Path to save the sorted JSONL file")

    args = parser.parse_args()

    lines = []
    with open(args.input_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            lines.append(obj)

    label_key = "__label__1"

    # Sort descending by label_key
    lines.sort(key=lambda x: x["metadata"][label_key], reverse=True)

    with open(args.output_file, "w") as f_out:
        for obj in lines:
            f_out.write(json.dumps(obj) + "\n")

    print(f"Sorted {args.input_file} by {label_key} and saved to {args.output_file}")

if __name__ == "__main__":
    main()