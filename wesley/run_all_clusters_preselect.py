import subprocess
import os
import sys
import argparse
import re

'''
this is a script that runs the full preselect pipeline on all clusters
pass in the directory to all cluster files, output directory, threshold, and model path
'''

def run_preselect_filtering(input_file, output_dir, model_path, threshold, steps):
    command = [
        "python", "Data_Filtering_Challenge/wesley/run_preselect_filtering.py",
        "--input_path", input_file,
        "--output_dir", output_dir,
        "--model_path", model_path,
        "--threshold", str(threshold),
        "--steps", ",".join(map(str, steps))
    ]
    
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error processing {input_file}")
        sys.exit(1)
    else:
        print(f"Successfully processed {input_file}")

def extract_cluster_number(filename):
    match = re.search(r"cluster[_]?(\d+)", os.path.basename(filename))
    if match:
        return int(match.group(1))
    return float('inf')

def process_all_files_in_directory(input_dir, output_dir, model_path, threshold, steps):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir), key=extract_cluster_number)
    
    for file_name in files:
        input_file = os.path.join(input_dir, file_name)

        if input_file.endswith(".json"):
            input_file = input_file.replace(".json", ".jsonl")
            os.rename(os.path.join(input_dir, file_name), input_file)
            print(f"Renamed {file_name} to {os.path.basename(input_file)}")

        if input_file.endswith(".jsonl"):
            print(f"Processing file: {input_file}")
            run_preselect_filtering(input_file, output_dir, model_path, threshold, steps)

def main():
    parser = argparse.ArgumentParser(description="Run PreSelect filtering on all clusters in a directory.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input .jsonl files")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output files")
    parser.add_argument("--model_path", default="PreSelect-classifier.bin", help="Path to FastText model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classifier threshold")
    parser.add_argument("--steps", type=str, default="1,2,3", help="Comma-separated list of steps to run")

    args = parser.parse_args()

    steps = [int(s.strip()) for s in args.steps.split(",")]

    # Run the process for all files in the input directory\
    process_all_files_in_directory(args.input_dir, args.output_dir, args.model_path, args.threshold, steps)

if __name__ == "__main__":
    main()