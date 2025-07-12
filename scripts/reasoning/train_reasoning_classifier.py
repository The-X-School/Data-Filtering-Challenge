import argparse
import logging
import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics_multiclass(pred):
    """Computes metrics for multi-class classification (5 classes)."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    # Calculate per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall
    }

def compute_metrics_binary(pred):
    """Computes metrics for binary classification."""
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

def load_and_prepare_data(file_path: str, classification_type: str = 'multiclass', 
                         binary_threshold: int = 3) -> Tuple[List[Dict], Dict]:
    """
    Load and prepare data from stage2_10k.jsonl
    
    Args:
        file_path: Path to the jsonl file
        classification_type: 'multiclass' (1-5) or 'binary' (high/low quality)
        binary_threshold: For binary classification, scores > threshold are high quality
    
    Returns:
        Tuple of (processed_data, label_mapping)
    """
    logging.info(f"Loading data from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    logging.info(f"Loaded {len(data)} items from {file_path}")
    
    # Check data structure
    if not data:
        raise ValueError("No valid data found in the file")
    
    sample_item = data[0]
    logging.info(f"Sample data structure: {list(sample_item.keys())}")
    
    # Prepare the processed data
    processed_data = []
    
    if classification_type == 'multiclass':
        # Use reasoning_score directly (1-5)
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}  # Map 1-5 to 0-4 for model
        reverse_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        
        for item in data:
            if 'reasoning_score' in item and 'text' in item:
                score = item['reasoning_score']
                if score in [1, 2, 3, 4, 5]:
                    processed_data.append({
                        'text': item['text'],
                        'label': label_mapping[score],
                        'original_score': score
                    })
            else:
                logging.warning(f"Item missing required fields: {item}")
                
    else:  # binary classification
        # Convert to binary: high quality (> threshold) vs low quality (<= threshold)
        label_mapping = {'low': 0, 'high': 1}
        reverse_mapping = {0: 'low', 1: 'high'}
        
        for item in data:
            if 'reasoning_score' in item and 'text' in item:
                score = item['reasoning_score']
                if score in [1, 2, 3, 4, 5]:
                    binary_label = 'high' if score > binary_threshold else 'low'
                    processed_data.append({
                        'text': item['text'],
                        'label': label_mapping[binary_label],
                        'original_score': score,
                        'binary_label': binary_label
                    })
    
    logging.info(f"Processed {len(processed_data)} valid items")
    
    # Print label distribution
    label_counts = {}
    for item in processed_data:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logging.info(f"Label distribution: {label_counts}")
    
    return processed_data, {'label_mapping': label_mapping, 'reverse_mapping': reverse_mapping}

def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, 
               test_ratio: float = 0.1, random_seed: int = 42) -> Dict[str, List[Dict]]:
    """Split data into train/validation/test sets."""
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    random.seed(random_seed)
    random.shuffle(data)
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logging.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

def train_classifier(args):
    """Train a classifier on reasoning score data."""
    
    # Load and prepare data
    processed_data, label_info = load_and_prepare_data(
        args.data_file, 
        args.classification_type, 
        args.binary_threshold
    )
    
    # Split data
    data_splits = split_data(processed_data, args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Convert to datasets
    datasets = {}
    for split_name, split_data_list in data_splits.items():
        datasets[split_name] = Dataset.from_list(split_data_list)
    
    # Load tokenizer
    logging.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=args.max_length
        )
    
    # Tokenize datasets
    logging.info("Tokenizing datasets...")
    tokenized_datasets = {}
    for split_name, dataset in datasets.items():
        tokenized_datasets[split_name] = dataset.map(tokenize_function, batched=True)
    
    # Determine number of labels
    num_labels = 5 if args.classification_type == 'multiclass' else 2
    
    # Load model
    logging.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        
        # Performance optimizations
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        
        # Reporting
        report_to="none",  # Change to "wandb" if you want to use wandb
    )
    
    # Select compute metrics function
    compute_metrics = compute_metrics_multiclass if args.classification_type == 'multiclass' else compute_metrics_binary
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    
    logging.info("--- Test Set Results ---")
    for key, value in test_results.items():
        logging.info(f"{key}: {value:.4f}")
    
    # Get detailed predictions for analysis
    test_predictions = trainer.predict(tokenized_datasets['test'])
    predicted_labels = np.argmax(test_predictions.predictions, axis=1)
    true_labels = test_predictions.label_ids
    
    # Print classification report
    if args.classification_type == 'multiclass':
        target_names = ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']
    else:
        target_names = ['Low Quality', 'High Quality']
    
    logging.info("\n--- Classification Report ---")
    logging.info(f"\n{classification_report(true_labels, predicted_labels, target_names=target_names)}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save label mapping
    with open(os.path.join(final_model_path, "label_mapping.json"), 'w') as f:
        json.dump(label_info, f, indent=2)
    
    logging.info(f"Model saved to {final_model_path}")
    
    # Save test predictions for analysis
    test_analysis = []
    for i, (text, true_label, pred_label, original_score) in enumerate(zip(
        tokenized_datasets['test']['text'], 
        true_labels, 
        predicted_labels,
        tokenized_datasets['test']['original_score']
    )):
        test_analysis.append({
            'text': text,
            'true_label': int(true_label),
            'predicted_label': int(pred_label),
            'original_score': original_score,
            'correct': true_label == pred_label
        })
    
    with open(os.path.join(args.output_dir, "test_predictions.json"), 'w') as f:
        json.dump(test_analysis, f, indent=2)
    
    logging.info(f"Test predictions saved to {os.path.join(args.output_dir, 'test_predictions.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reasoning quality classifier")
    
    # Data arguments
    parser.add_argument("--data_file", type=str, required=True, help="Path to stage2_10k.jsonl file")
    parser.add_argument("--classification_type", type=str, choices=['multiclass', 'binary'], 
                       default='multiclass', help="Classification type: multiclass (1-5) or binary (high/low)")
    parser.add_argument("--binary_threshold", type=int, default=3, 
                       help="For binary classification, scores > threshold are high quality")
    
    # Data splitting
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test data ratio")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/deberta-v3-small", 
                       help="Model to fine-tune")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="output/reasoning_classifier", 
                       help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=200, help="Save frequency")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging frequency")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_classifier(args) 