#!/bin/bash

# Training script for reasoning quality classifier

# Set default values
DATA_FILE="stage2_10k.jsonl"
OUTPUT_DIR="output/reasoning_classifier"
MODEL_NAME="microsoft/deberta-v3-small"
CLASSIFICATION_TYPE="multiclass"  # or "binary"
BATCH_SIZE=16
EPOCHS=3
LEARNING_RATE=2e-5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --classification_type)
            CLASSIFICATION_TYPE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --data_file PATH          Path to stage2_10k.jsonl file"
            echo "  --output_dir PATH         Output directory for model"
            echo "  --model_name MODEL        HuggingFace model name"
            echo "  --classification_type TYPE   'multiclass' or 'binary'"
            echo "  --batch_size SIZE         Training batch size"
            echo "  --epochs NUM              Number of training epochs"
            echo "  --learning_rate RATE      Learning rate"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=== Reasoning Classifier Training Configuration ==="
echo "Data file: $DATA_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Classification type: $CLASSIFICATION_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "=================================================="

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file $DATA_FILE not found!"
    exit 1
fi

# Run training
python scripts/reasoning/train_reasoning_classifier.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$MODEL_NAME" \
    --classification_type "$CLASSIFICATION_TYPE" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size $((BATCH_SIZE * 2)) \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --max_length 512 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 50

echo "Training completed. Model saved to: $OUTPUT_DIR/final_model"
echo "Test predictions saved to: $OUTPUT_DIR/test_predictions.json" 