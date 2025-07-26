#!/bin/bash
# Complete GSM8K preprocessing workflow
# Usage: bash prepare_gsm8k_for_training.sh <raw_gsm8k_file.jsonl>

if [ $# -ne 1 ]; then
    echo "Usage: bash prepare_gsm8k_for_training.sh <raw_gsm8k_file.jsonl>"
    echo "Example: bash prepare_gsm8k_for_training.sh raw_gsm8k_data.jsonl"
    exit 1
fi

RAW_FILE="data/gsm8k.jsonl"
INTERMEDIATE_FILE="data/gsm8k_text_format.jsonl"
FINAL_FILE="data/gsm8k_lmflow_format.jsonl"

if [ ! -f "$RAW_FILE" ]; then
    echo "Error: Input file $RAW_FILE does not exist"
    exit 1
fi

echo "=== GSM8K Data Preprocessing Workflow ==="
echo "Step 1: Converting GSM8K format (question/answer) to text format..."

# Step 1: Convert GSM8K format to text format
python preprocess_gsm8k.py "$RAW_FILE" "$INTERMEDIATE_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Failed to convert GSM8K format to text format"
    exit 1
fi

echo "Step 2: Converting text format to LMFlow format..."

# Step 2: Convert text format to LMFlow format
python format_data.py "$INTERMEDIATE_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Failed to convert to LMFlow format"
    exit 1
fi

# Step 3: Move the final file to the expected location
mv "$INTERMEDIATE_FILE" "$FINAL_FILE"

echo "=== Preprocessing Complete ==="
echo "Original file: $RAW_FILE"
echo "Final LMFlow format file: $FINAL_FILE"
echo ""
echo "You can now use this file for training:"
echo "bash train.sh --dataset_path $FINAL_FILE" 