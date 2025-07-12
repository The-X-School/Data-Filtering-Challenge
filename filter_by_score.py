import json
import os
import random


def filter_jsonl_by_score(input_file_path, output_file_path, num_random_samples=5000):
    """
    Filters a JSONL file to keep all entries with a reasoning_score of 5,
    and a random sample of other entries (scores 1-4).

    Args:
        input_file_path (str): The path to the input JSONL file.
        output_file_path (str): The path to the output JSONL file.
        num_random_samples (int): The number of random samples to keep from other scores.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    other_quality_samples = []
    other_quality_lines_seen = 0

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                data = json.loads(line)
                score = data.get("reasoning_score")

                if score == 5:
                    outfile.write(line)
                elif score is not None and 1 <= score <= 4:
                    other_quality_lines_seen += 1
                    if len(other_quality_samples) < num_random_samples:
                        other_quality_samples.append(line)
                    else:
                        j = random.randint(0, other_quality_lines_seen - 1)
                        if j < num_random_samples:
                            other_quality_samples[j] = line
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

        for line in other_quality_samples:
            outfile.write(line)


if __name__ == "__main__":
    input_path = "stage2_1m.jsonl"
    output_path = "stage3_distributed.jsonl"
    
    filter_jsonl_by_score(input_path, output_path, num_random_samples=5000)
    
    print(f"Filtering complete. Output saved to {output_path}") 