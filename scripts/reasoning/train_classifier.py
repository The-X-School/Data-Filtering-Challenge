import argparse
import logging
import os
import json

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(pred):
    """Computes accuracy, F1, precision, and recall for binary classification."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_classifier(args):
    """Loads data, tokenizes it, and trains a text classification model."""
    
    # --- 1. Load and prepare datasets ---
    logging.info("Loading datasets...")
    
    def load_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl(os.path.join(args.data_dir, 'train.jsonl'))
    validation_data = load_jsonl(os.path.join(args.data_dir, 'validation.jsonl'))
    test_data = load_jsonl(os.path.join(args.data_dir, 'test.jsonl'))

    raw_datasets = {
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(validation_data),
        'test': Dataset.from_list(test_data)
    }

    logging.info(f"Datasets loaded. Train: {len(raw_datasets['train'])}, Validation: {len(raw_datasets['validation'])}, Test: {len(raw_datasets['test'])}")

    # --- 2. Load tokenizer and tokenize data ---
    logging.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=args.max_length)

    logging.info("Tokenizing datasets...")
    tokenized_datasets = {
        'train': raw_datasets['train'].map(tokenize_function, batched=True),
        'validation': raw_datasets['validation'].map(tokenize_function, batched=True),
        'test': raw_datasets['test'].map(tokenize_function, batched=True)
    }

    # --- 3. Load model ---
    logging.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=2, use_safetensors=True
    )

    # --- 4. Define Training Arguments ---
    # Optimized for 4090-class GPUs
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        
        # Evaluation and saving strategy
        eval_steps=170,
        save_steps=170,
        logging_steps=170,
        
        # Performance optimizations
        bf16=True, # Key for speed on Ampere/Ada GPUs
        tf32=True, # Use TF32 for matmuls
    )
    logging.info(f"Training arguments: {training_args.to_json_string()}")

    # --- 5. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # --- 6. Train and evaluate ---
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")
    
    logging.info("Evaluating on the test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    
    logging.info("--- Test Set Evaluation Results ---")
    for key, value in test_results.items():
        logging.info(f"{key}: {value:.4f}")
        
    # --- 7. Save final model and tokenizer ---
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logging.info(f"Best model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classifier for reasoning tasks.")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/deberta-v3-small", help="Model to fine-tune.")
    parser.add_argument("--data_dir", type=str, default="output/prepare_1m", help="Directory containing train.jsonl, validation.jsonl, and test.jsonl.")
    parser.add_argument("--output_dir", type=str, default="output/reasoning_classifier_model", help="Directory to save the trained model and logs.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenizer.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")

    args = parser.parse_args()
    train_classifier(args) 