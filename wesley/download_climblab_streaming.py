import os
import json
import itertools
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
DATASET_NAME = "nvidia/ClimbLab"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "tokenized", "climblab")
OUTPUT_FILENAME = "climblab.jsonl"
NUM_ROWS_TO_DOWNLOAD = 1_000_000

def main():
    parser = argparse.ArgumentParser(description="Download data from Hugging Face and calculate token count.")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save the output file.')
    parser.add_argument('--output_filename', type=str, default=OUTPUT_FILENAME, help='Name of the output file.')
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME, help='Name of the dataset on Hugging Face.')
    parser.add_argument('--num_rows', type=int, default=NUM_ROWS_TO_DOWNLOAD, help='Number of rows to download.')
    
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)

    # Load the dataset in streaming mode to avoid downloading the whole dataset
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)

    if not dataset:
        print("Failed to load dataset. Exiting.")
        return

    print("Dataset loaded successfully. Verifying data stream...")
    try:
        first_row = next(iter(dataset))
        print(f"Successfully fetched the first row: {first_row}")
        # We need to put the first row back into an iterable for the loop
        dataset = itertools.chain([first_row], dataset)
    except StopIteration:
        print("Dataset appears to be empty. No data to download.")
        return

    total_tokens = 0
    rows_processed = 0

    print(f"Downloading the first {args.num_rows:,} rows and saving to '{output_path}'...")

    # Using a context manager to handle the file
    with open(output_path, 'w') as f:
        # Use itertools.islice to take the first N rows from the stream
        progress_bar = tqdm(itertools.islice(dataset, args.num_rows),
                            total=args.num_rows,
                            desc="Downloading rows")
        
        for row in progress_bar:
            # The 'token_count' field contains the number of tokens
            total_tokens += row['token_count']
            rows_processed += 1
            
            # Write the row as a JSON line
            f.write(json.dumps(row) + '\n')

    print("\nDownload complete.")
    print(f"Total rows processed: {rows_processed:,}")
    print(f"Total tokens downloaded: {total_tokens:,}")
    print(f"Data saved to '{output_path}'")

if __name__ == "__main__":
    main()