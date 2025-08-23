import json
import argparse
from collections import Counter

"""
This script takes in input as a jsonl and extracts the "text" field,
outputting to a new file.

If the data has "__label__1" in the metadata (added after preselect training, ignore otherwise)
the code will count the frequency of values (rounding to 2 decimal places)

To run this code:
python format_preselect.py path/to/input.jsonl path/to/output.jsonl

"""

def format(input_file, output_file):
    label_counter = Counter()
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                text = data.get("text")
                if text is not None:
                    json.dump({"text": text}, outfile)
                    outfile.write("\n")
                metadata = data.get("metadata", {})
                label_1 = metadata.get("__label__1")
                if label_1 is not None:
                    rounded_label = round(label_1, 2)
                    label_counter[rounded_label] += 1

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    if label_counter:
        print("\n__label__1 frequencies:")
        for label_value in sorted(label_counter.keys(), reverse=True):
            count = label_counter[label_value]
            print(f"{label_value:.2f}: {count}")

    print(f"formatted {input_file} into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 'text' field from JSONL file.")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output text file")
    args = parser.parse_args()

    format(args.input_file, args.output_file)