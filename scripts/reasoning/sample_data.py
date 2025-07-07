import json
import random
import sys
import os
import subprocess
import argparse

def ensure_packages_installed(packages):
    """Checks if packages are installed and installs them if not."""
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def sample_jsonl_data(input_file, output_file, sample_size):
    """
    Randomly samples records from a large JSONL file using reservoir sampling
    for memory efficiency, with a progress bar. Writes output to a JSONL file.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print("Required package (tqdm) not found.")
        sys.exit(1)

    print(f"Starting memory-efficient sampling from {input_file}...")

    reservoir = []
    items_processed = 0

    try:
        # Get total number of lines for a nice progress bar
        print("Counting total lines for progress bar...")
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"Found {total_lines} lines in the input file.")

        with open(input_file, 'r', encoding='utf-8') as f:
            print("Processing items from JSONL stream...")
            for line in tqdm(f, total=total_lines, desc="Sampling Records"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line.strip()}", file=sys.stderr)
                    continue
                
                items_processed += 1
                if len(reservoir) < sample_size:
                    reservoir.append(record)
                else:
                    j = random.randint(0, items_processed - 1)
                    if j < sample_size:
                        reservoir[j] = record

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during sampling: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal items processed: {items_processed}")
    if items_processed < sample_size:
        print(f"Warning: Only {items_processed} items found, less than the desired {sample_size}.")

    print(f"Saving {len(reservoir)} sampled records to {output_file}...")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in reservoir:
                f.write(json.dumps(record) + '\n')
    except IOError as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Sampling complete.")
    print(f"Output file created at: {output_file}")

if __name__ == "__main__":
    ensure_packages_installed(["tqdm"])

    parser = argparse.ArgumentParser(
        description="Randomly sample records from a large JSONL file and save as a JSONL file."
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="output/climblab_10M.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="output/climblab_1M.jsonl",
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=1000000,
        help="The number of records to sample."
    )
    
    args = parser.parse_args()

    sample_jsonl_data(args.input_file, args.output_file, args.sample_size) 