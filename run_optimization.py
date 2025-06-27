#!/usr/bin/env python
# coding=utf-8
"""
Simple runner script for Bayesian hyperparameter optimization
"""

import os
import sys
import yaml
import argparse
from bayesian_hyperparameter_tuning import BayesianHyperparameterOptimizer, OptimizationConfig


def load_config(config_path: str) -> OptimizationConfig:
    """Load configuration from YAML file"""
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Extract configuration values
    study_config = config_data.get('study', {})
    model_config = config_data.get('model', {})
    output_config = config_data.get('output', {})
    eval_config = config_data.get('evaluation', {})
    
    # Create OptimizationConfig object
    config = OptimizationConfig(
        study_name=study_config.get('name', 'dora_optimization'),
        n_trials=study_config.get('n_trials', 30),
        timeout=study_config.get('timeout', 43200),
        base_model_path=model_config.get('base_model_path', 'data4elm/Llama-400M-12L'),
        dataset_path=model_config.get('dataset_path', 'data/filtered_output'),
        output_base_dir=output_config.get('base_dir', 'optimization_results'),
        models_dir=output_config.get('models_dir', 'optimization_models'),
        evaluation_limit=eval_config.get('limit', 50),
        device=eval_config.get('device', 'cuda:0'),
        enable_pruning=study_config.get('enable_pruning', True),
        pruning_warmup_steps=study_config.get('pruning_warmup_steps', 3)
    )
    
    return config


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Run Bayesian hyperparameter optimization for DoRA fine-tuning')
    parser.add_argument('--config', type=str, default='optimization_config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing study if available')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device setting (cuda:0, cpu, etc.)')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found!")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    print(f"Starting Bayesian optimization with configuration:")
    print(f"  Study name: {config.study_name}")
    print(f"  Number of trials: {config.n_trials}")
    print(f"  Timeout: {config.timeout} seconds")
    print(f"  Base model: {config.base_model_path}")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Device: {config.device}")
    print(f"  Evaluation limit: {config.evaluation_limit}")
    print(f"  Resume: {args.resume}")
    
    # Check if lm-evaluation-harness exists
    if not os.path.exists("lm-evaluation-harness"):
        print("Error: lm-evaluation-harness directory not found!")
        print("Please ensure the evaluation harness is properly set up.")
        sys.exit(1)
    
    # Check if training script exists
    if not os.path.exists("examples/finetune.py"):
        print("Error: examples/finetune.py not found!")
        print("Please ensure you're running from the correct directory.")
        sys.exit(1)
    
    # Create optimizer and run
    optimizer = BayesianHyperparameterOptimizer(config)
    
    try:
        study = optimizer.run_optimization()
        optimizer.save_results(study)
        
        print(f"\n" + "="*60)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best improvement score: {study.best_value:.4f}")
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Total trials completed: {len(study.trials)}")
        
        # Show best hyperparameters
        print(f"\nBest hyperparameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        # Show detailed results if available
        if hasattr(study.best_trial, 'user_attrs') and 'scores' in study.best_trial.user_attrs:
            best_scores = study.best_trial.user_attrs['scores']
            baseline_scores = optimizer.get_baseline_scores()
            
            print(f"\nDetailed Results:")
            print(f"{'Task':<20} {'Baseline':<10} {'Best Model':<12} {'Improvement':<12}")
            print("-" * 55)
            
            for task in ["elmb_roleplay", "elmb_reasoning", "elmb_functioncalling", "elmb_chatrag"]:
                baseline = baseline_scores.get(task, 0.25)
                best = best_scores.get(task, 0.25)
                improvement = best - baseline
                print(f"{task:<20} {baseline:<10.4f} {best:<12.4f} {improvement:<12.4f}")
            
            print(f"\nBest model path: {study.best_trial.user_attrs.get('model_path', 'N/A')}")
        
        print(f"\nResults saved to: {config.output_base_dir}/final_results/")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print("Partial results may be available in the output directory.")
    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 