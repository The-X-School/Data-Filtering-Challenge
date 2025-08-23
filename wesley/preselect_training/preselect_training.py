import os
import argparse
from pathlib import Path
import re
import shutil

parser = argparse.ArgumentParser("Filter")
parser.add_argument("--input_path", type=str, help="input path name", required=True)
parser.add_argument("--output_path", type=str, help="output path name", required=True)
parser.add_argument("--model_path", type=str, default="PreSelect-classifier.bin", help="path to FastText model")
parser.add_argument("--threshold", type=float, default=0.5, help="threshold for classifier filter")

args = parser.parse_args()

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import FastTextClassifierFilter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

# Ensure output directory exists
Path(args.output_path).mkdir(parents=True, exist_ok=True)

# Find the next available file number
pattern = re.compile(r"(\d{5})\.jsonl$")
existing_files = [f for f in os.listdir(args.output_path) if pattern.match(f)]
if existing_files:
    max_num = max(int(pattern.match(f).group(1)) for f in existing_files)
else:
    max_num = 0
next_num = max_num + 1
output_file = os.path.join(args.output_path, f"{next_num:05d}.jsonl")

# Use a temporary directory for JsonlWriter
_tmp_dir = os.path.join(args.output_path, "_tmp")
Path(_tmp_dir).mkdir(parents=True, exist_ok=True)

print(f"Using FastText model: {args.model_path}")
print(f"Using threshold: {args.threshold}")

dist_executor = LocalPipelineExecutor(
    skip_completed=True,
    pipeline=[
        JsonlReader(args.input_path, text_key="text", default_metadata={}),
        FastTextClassifierFilter(args.model_path, keep_labels=[("1", args.threshold)]),
        JsonlWriter(_tmp_dir, compression=None)
    ],
    tasks=1,
)

if __name__ == "__main__":
    dist_executor.run()
    tmp_jsonl = os.path.join(_tmp_dir, "00000.jsonl")
    if os.path.exists(tmp_jsonl):
        shutil.move(tmp_jsonl, output_file)
    shutil.rmtree(_tmp_dir)
