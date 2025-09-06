import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
# install packages before running this
# pip install datatrove datasets orjson fasteners fasttext-numpy2-wheel regex multiprocess dill

def run_command(command, check=True):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if check and result.returncode != 0:
        print(f"Command failed: {command}")
        sys.exit(1)

def find_latest_jsonl_file(directory):
    pattern = re.compile(r"(\d{5})\.jsonl$")
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {directory}")
    latest = max(files, key=lambda f: int(pattern.match(f).group(1)))
    return os.path.join(directory, latest)

def get_next_cluster_number(directory, filename):
    match = re.search(r"cluster[_]?(\d+)", os.path.basename(filename))
    if match:
        return int(match.group(1))
    else:
        pattern = re.compile(r"cluster(\d+)_?.*\.jsonl$")
        cluster_nums = [int(pattern.match(f).group(1)) for f in os.listdir(directory) if pattern.match(f)]
        return max(cluster_nums, default=0) + 1

def main():
    parser = argparse.ArgumentParser(description="Run the full PreSelect filtering pipeline.")
    parser.add_argument("--input_path", required=True, help="Path to the input data.jsonl file")
    parser.add_argument("--model_path", default="PreSelect-classifier.bin", help="Path to FastText model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classifier threshold")
    parser.add_argument("--output_dir", default="Data-Filtering-Challenge/data/default_dir", help="Directory to save final output")
    parser.add_argument("--steps", type=str, default="0,1,2,3,4", help="Comma-separated list of steps to run (default is 1,2,3,4)")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    steps = [int(s.strip()) for s in args.steps.split(",")]
    preselect_script_dir = ""

    cluster_num = get_next_cluster_number(args.output_dir, args.input_path)
    cluster_prefix = f"cluster{cluster_num}"

    detokenized_file = os.path.join(args.output_dir, f"{cluster_prefix}_detokenized.jsonl")
    cluster_output_file = os.path.join(args.output_dir, f"{cluster_prefix}.jsonl")
    sorted_file = os.path.join(args.output_dir, f"{cluster_prefix}_sorted.jsonl")
    formatted_file = os.path.join(args.output_dir, f"{cluster_prefix}_formatted.jsonl")

    # Step 0: Detokenize the data
    if 0 in steps:
        run_command(f"python {preselect_script_dir}/detokenize_climblab.py --input_path {args.input_path} --output_path {detokenized_file}")

    # Step 1: Run preselect_training.py
    if 1 in steps:
        run_command(
            f"python {preselect_script_dir}/preselect_training.py "
            f"--input_path={detokenized_file} "
            f"--output_path={args.output_dir} "
            f"--model_path={args.model_path} "
            f"--threshold={args.threshold}"
        )
        os.remove(detokenized_file)
        original_jsonl = find_latest_jsonl_file(args.output_dir)
        os.rename(original_jsonl, cluster_output_file)
        print(f"Renamed preselect output to: {cluster_output_file}")
    else:
        cluster_output_file = find_latest_jsonl_file(args.output_dir)

    # Step 2: Sort the preselect output using sort_preselect.py
    if 2 in steps:
        run_command(f"python {preselect_script_dir}/sort_preselect.py {cluster_output_file} {sorted_file}")
        os.remove(cluster_output_file)
        print(f"Deleted original files : {cluster_output_file}")

    # Step 3: Format the preselect output
    if 3 in steps:
        run_command(
            f"python {preselect_script_dir}/format_preselect.py "
            f"{sorted_file} {formatted_file}"
        )
        os.remove(sorted_file)
        print(f"Deleted original files : {sorted_file}")

    # Step 4: Convert to LMFlow format
    if 4 in steps:
        run_command(
            f"python {preselect_script_dir} format_data.py {formatted_file}"
        )

        print(f"All steps completed successfully. Final file is at: {formatted_file}")

if __name__ == "__main__":
    main()