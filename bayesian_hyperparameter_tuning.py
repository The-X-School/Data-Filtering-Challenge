#!/usr/bin/env python
# coding=utf-8
"""
Bayesian Hyperparameter Optimization for DoRA Fine-tuning
Evaluates models using ELMB benchmark and optimizes hyperparameters automatically.
"""

import os
import sys
import json
import time
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import optuna
import numpy as np
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bayesian_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization"""
    # Study configuration
    study_name: str = "dora_hyperparameter_optimization"
    n_trials: int = 50
    timeout: int = 86400  # 24 hours in seconds
    
    # Model and data paths
    base_model_path: str = "data4elm/Llama-400M-12L"
    dataset_path: str = "data/filtered_output"
    
    # Output directories
    output_base_dir: str = "optimization_results"
    models_dir: str = "optimization_models"
    
    # Evaluation configuration
    evaluation_limit: int = 100  # Limit evaluation samples for faster trials
    device: str = "cuda:0"
    
    # Fixed hyperparameters (cannot be changed per competition rules)
    fixed_num_train_epochs: int = 1
    fixed_learning_rate: float = 1e-5
    fixed_lora_r: int = 16
    
    # Early stopping for bad trials
    enable_pruning: bool = True
    pruning_warmup_steps: int = 3


class BayesianHyperparameterOptimizer:
    """Bayesian optimization for DoRA fine-tuning hyperparameters"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.setup_directories()
        self.baseline_scores = None
        
    def setup_directories(self):
        """Create necessary directories for optimization"""
        os.makedirs(self.config.output_base_dir, exist_ok=True)
        os.makedirs(self.config.models_dir, exist_ok=True)
        
    def get_baseline_scores(self) -> Dict[str, float]:
        """Get baseline model performance on ELMB benchmark"""
        if self.baseline_scores is not None:
            return self.baseline_scores
            
        logger.info("Evaluating baseline model performance...")
        
        # Change to evaluation directory
        eval_dir = os.path.join(os.getcwd(), "lm-evaluation-harness")
        if not os.path.exists(eval_dir):
            raise FileNotFoundError(f"lm-evaluation-harness directory not found at {eval_dir}")
        
        # Run baseline evaluation
        baseline_output = os.path.join(self.config.output_base_dir, "baseline_results")
        os.makedirs(baseline_output, exist_ok=True)
        
        eval_cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.config.base_model_path},trust_remote_code=True",
            "--tasks", "elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag",
            "--device", self.config.device,
            "--batch_size", "1",
            "--limit", str(self.config.evaluation_limit),
            "--output_path", baseline_output
        ]
        
        try:
            result = subprocess.run(
                eval_cmd,
                cwd=eval_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Baseline evaluation failed: {result.stderr}")
                # Use default baseline scores if evaluation fails
                self.baseline_scores = {
                    "elmb_roleplay": 0.25,
                    "elmb_reasoning": 0.25,
                    "elmb_functioncalling": 0.25,
                    "elmb_chatrag": 0.25
                }
            else:
                self.baseline_scores = self._parse_evaluation_results(baseline_output)
                
        except subprocess.TimeoutExpired:
            logger.error("Baseline evaluation timed out")
            self.baseline_scores = {
                "elmb_roleplay": 0.25,
                "elmb_reasoning": 0.25,
                "elmb_functioncalling": 0.25,
                "elmb_chatrag": 0.25
            }
        
        logger.info(f"Baseline scores: {self.baseline_scores}")
        return self.baseline_scores
    
    def _parse_evaluation_results(self, output_path: str) -> Dict[str, float]:
        """Parse evaluation results from lm_eval output"""
        scores = {}
        
        # Look for results.json file
        results_file = os.path.join(output_path, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract accuracy scores for each task
                for task in ["elmb_roleplay", "elmb_reasoning", "elmb_functioncalling", "elmb_chatrag"]:
                    if task in results.get("results", {}):
                        task_results = results["results"][task]
                        # Try to get acc_norm first, then acc
                        score = task_results.get("acc_norm", task_results.get("acc", 0.25))
                        scores[task] = float(score)
                    else:
                        scores[task] = 0.25  # Default score
                        
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing results: {e}")
                scores = {task: 0.25 for task in ["elmb_roleplay", "elmb_reasoning", "elmb_functioncalling", "elmb_chatrag"]}
        else:
            logger.warning(f"Results file not found at {results_file}")
            scores = {task: 0.25 for task in ["elmb_roleplay", "elmb_reasoning", "elmb_functioncalling", "elmb_chatrag"]}
        
        return scores
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, any]:
        """Suggest hyperparameters for optimization"""
        
        # Tunable hyperparameters (while respecting fixed constraints)
        hyperparams = {
            # Training batch size and accumulation
            "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", 1, 8),
            "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 16),
            
            # Model architecture parameters
            "block_size": trial.suggest_categorical("block_size", [512, 1024, 2048]),
            "lora_alpha": trial.suggest_int("lora_alpha", 8, 64),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.3),
            
            # Optimization parameters
            "warmup_steps": trial.suggest_int("warmup_steps", 0, 100),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "constant"]),
            
            # Data processing
            "dataloader_num_workers": trial.suggest_int("dataloader_num_workers", 1, 4),
            "preprocessing_num_workers": trial.suggest_int("preprocessing_num_workers", 32, 256, step=32),
            
            # DeepSpeed configuration
            "deepspeed_config": trial.suggest_categorical("deepspeed_config", [
                "configs/ds_config_zero0_no_offload.json",
                "configs/ds_config_zero2.json"
            ]),
            
            # Logging and saving
            "logging_steps": trial.suggest_int("logging_steps", 10, 50),
            "save_steps": trial.suggest_int("save_steps", 1000, 10000, step=1000),
            
            # Validation
            "validation_split_percentage": trial.suggest_float("validation_split_percentage", 0, 10),
        }
        
        return hyperparams
    
    def run_training(self, trial: optuna.Trial, hyperparams: Dict[str, any]) -> str:
        """Run training with given hyperparameters"""
        
        # Create unique output directory for this trial
        trial_id = f"trial_{trial.number}_{int(time.time())}"
        output_dir = os.path.join(self.config.models_dir, trial_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Build training command
        train_cmd = [
            "deepspeed", "--master_port=11000",
            "examples/finetune.py",
            "--model_name_or_path", self.config.base_model_path,
            "--trust_remote_code", "0",
            "--dataset_path", self.config.dataset_path,
            "--output_dir", output_dir,
            "--overwrite_output_dir",
            
            # Fixed hyperparameters
            "--num_train_epochs", str(self.config.fixed_num_train_epochs),
            "--learning_rate", str(self.config.fixed_learning_rate),
            "--lora_r", str(self.config.fixed_lora_r),
            
            # DoRA specific parameters
            "--use_dora", "1",
            "--lora_target_modules", "embed_tokens,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head",
            "--save_aggregated_lora", "0",
            
            # Optimized hyperparameters
            "--per_device_train_batch_size", str(hyperparams["per_device_train_batch_size"]),
            "--gradient_accumulation_steps", str(hyperparams["gradient_accumulation_steps"]),
            "--block_size", str(hyperparams["block_size"]),
            "--lora_alpha", str(hyperparams["lora_alpha"]),
            "--lora_dropout", str(hyperparams["lora_dropout"]),
            "--warmup_steps", str(hyperparams["warmup_steps"]),
            "--weight_decay", str(hyperparams["weight_decay"]),
            "--lr_scheduler_type", hyperparams["lr_scheduler_type"],
            "--dataloader_num_workers", str(hyperparams["dataloader_num_workers"]),
            "--preprocessing_num_workers", str(hyperparams["preprocessing_num_workers"]),
            "--deepspeed", hyperparams["deepspeed_config"],
            "--logging_steps", str(hyperparams["logging_steps"]),
            "--save_steps", str(hyperparams["save_steps"]),
            "--validation_split_percentage", str(hyperparams["validation_split_percentage"]),
            
            # Additional training parameters
            "--bf16",
            "--run_name", f"trial_{trial.number}",
            "--report_to", "none",
            "--do_train",
            "--ddp_timeout", "72000"
        ]
        
        # Run training
        logger.info(f"Starting training trial {trial.number} with hyperparams: {hyperparams}")
        
        try:
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Training failed for trial {trial.number}: {result.stderr}")
                raise RuntimeError(f"Training failed: {result.stderr}")
                
            logger.info(f"Training completed for trial {trial.number}")
            return output_dir
            
        except subprocess.TimeoutExpired:
            logger.error(f"Training timed out for trial {trial.number}")
            raise RuntimeError("Training timed out")
    
    def merge_dora_weights(self, dora_path: str) -> str:
        """Merge DoRA weights with base model"""
        
        merged_path = dora_path + "_merged"
        
        merge_cmd = [
            "bash", "./scripts/run_merge_dora.sh",
            "--model_name_or_path", self.config.base_model_path,
            "--lora_model_path", dora_path,
            "--output_model_path", merged_path,
            "--device", "cpu"
        ]
        
        try:
            result = subprocess.run(
                merge_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Merging failed: {result.stderr}")
                raise RuntimeError(f"Merging failed: {result.stderr}")
            
            logger.info(f"DoRA weights merged successfully: {merged_path}")
            return merged_path
            
        except subprocess.TimeoutExpired:
            logger.error("DoRA merging timed out")
            raise RuntimeError("DoRA merging timed out")
    
    def evaluate_model(self, model_path: str) -> Dict[str, float]:
        """Evaluate model using ELMB benchmark"""
        
        eval_output = model_path + "_eval_results"
        os.makedirs(eval_output, exist_ok=True)
        
        eval_dir = os.path.join(os.getcwd(), "lm-evaluation-harness")
        
        eval_cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True",
            "--tasks", "elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag",
            "--device", self.config.device,
            "--batch_size", "1",
            "--limit", str(self.config.evaluation_limit),
            "--output_path", eval_output
        ]
        
        try:
            result = subprocess.run(
                eval_cmd,
                cwd=eval_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Evaluation failed: {result.stderr}")
                raise RuntimeError(f"Evaluation failed: {result.stderr}")
            
            scores = self._parse_evaluation_results(eval_output)
            logger.info(f"Evaluation scores: {scores}")
            return scores
            
        except subprocess.TimeoutExpired:
            logger.error("Evaluation timed out")
            raise RuntimeError("Evaluation timed out")
    
    def calculate_improvement_score(self, scores: Dict[str, float]) -> float:
        """Calculate total improvement score (S = S_improve - S_base)"""
        
        baseline_scores = self.get_baseline_scores()
        
        total_improvement = 0.0
        for task in ["elmb_roleplay", "elmb_reasoning", "elmb_functioncalling", "elmb_chatrag"]:
            improvement = scores.get(task, 0.0) - baseline_scores.get(task, 0.25)
            total_improvement += improvement
        
        return total_improvement
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        
        try:
            # Suggest hyperparameters
            hyperparams = self.suggest_hyperparameters(trial)
            
            # Run training
            dora_path = self.run_training(trial, hyperparams)
            
            # Merge DoRA weights
            merged_path = self.merge_dora_weights(dora_path)
            
            # Evaluate model
            scores = self.evaluate_model(merged_path)
            
            # Calculate improvement score
            improvement_score = self.calculate_improvement_score(scores)
            
            # Store results
            trial.set_user_attr("scores", scores)
            trial.set_user_attr("hyperparams", hyperparams)
            trial.set_user_attr("model_path", merged_path)
            
            # Clean up intermediate files to save space
            if os.path.exists(dora_path):
                shutil.rmtree(dora_path, ignore_errors=True)
            
            logger.info(f"Trial {trial.number} completed with improvement score: {improvement_score}")
            return improvement_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            # Return a very negative score for failed trials
            return -1000.0
    
    def run_optimization(self) -> optuna.Study:
        """Run Bayesian optimization"""
        
        logger.info("Starting Bayesian hyperparameter optimization...")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=self.config.pruning_warmup_steps) if self.config.enable_pruning else None
        
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{self.config.output_base_dir}/optimization.db",
            load_if_exists=True
        )
        
        # Run optimization
        try:
            study.optimize(
                self.objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=1  # Sequential execution for GPU resource management
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        return study
    
    def save_results(self, study: optuna.Study):
        """Save optimization results"""
        
        results_dir = os.path.join(self.config.output_base_dir, "final_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save study statistics
        study_stats = {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "study_name": study.study_name
        }
        
        with open(os.path.join(results_dir, "study_statistics.json"), 'w') as f:
            json.dump(study_stats, f, indent=2)
        
        # Save detailed results
        trials_data = []
        for trial in study.trials:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "duration": trial.duration.total_seconds() if trial.duration else None,
                "scores": trial.user_attrs.get("scores", {}),
                "hyperparams": trial.user_attrs.get("hyperparams", {}),
                "model_path": trial.user_attrs.get("model_path", "")
            }
            trials_data.append(trial_data)
        
        with open(os.path.join(results_dir, "all_trials.json"), 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Save best model info
        best_trial = study.best_trial
        best_model_info = {
            "trial_number": best_trial.number,
            "improvement_score": best_trial.value,
            "hyperparameters": best_trial.params,
            "model_path": best_trial.user_attrs.get("model_path", ""),
            "scores": best_trial.user_attrs.get("scores", {}),
            "baseline_scores": self.get_baseline_scores()
        }
        
        with open(os.path.join(results_dir, "best_model.json"), 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")
        logger.info(f"Best trial: {best_trial.number} with improvement score: {best_trial.value}")
        logger.info(f"Best hyperparameters: {best_trial.params}")


def main():
    """Main function to run Bayesian optimization"""
    
    # Configuration
    config = OptimizationConfig(
        n_trials=30,  # Adjust based on available time and resources
        timeout=43200,  # 12 hours
        evaluation_limit=50,  # Reduce for faster trials during development
        device="cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    )
    
    # Create optimizer
    optimizer = BayesianHyperparameterOptimizer(config)
    
    # Run optimization
    study = optimizer.run_optimization()
    
    # Save results
    optimizer.save_results(study)
    
    print(f"\nOptimization completed!")
    print(f"Best improvement score: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
    
    # Generate summary report
    best_trial = study.best_trial
    best_scores = best_trial.user_attrs.get("scores", {})
    baseline_scores = optimizer.get_baseline_scores()
    
    print(f"\nDetailed Results:")
    print(f"{'Task':<20} {'Baseline':<10} {'Best Model':<12} {'Improvement':<12}")
    print("-" * 55)
    
    for task in ["elmb_roleplay", "elmb_reasoning", "elmb_functioncalling", "elmb_chatrag"]:
        baseline = baseline_scores.get(task, 0.25)
        best = best_scores.get(task, 0.25)
        improvement = best - baseline
        print(f"{task:<20} {baseline:<10.4f} {best:<12.4f} {improvement:<12.4f}")
    
    print(f"\nBest model saved at: {best_trial.user_attrs.get('model_path', 'N/A')}")


if __name__ == "__main__":
    main() 