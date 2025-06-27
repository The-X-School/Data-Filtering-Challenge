#!/usr/bin/env python
# coding=utf-8
"""
Subset-Based Bayesian Optimization for DoRA Fine-tuning
Optimizes DoRA target modules and focused hyperparameter subsets for maximum efficiency.
"""

import os
import sys
import json
import time
import logging
import itertools
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

import optuna
import numpy as np
from bayesian_hyperparameter_tuning import BayesianHyperparameterOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class SubsetOptimizationConfig(OptimizationConfig):
    """Extended configuration for subset optimization"""
    
    # DoRA module optimization
    optimize_target_modules: bool = True
    min_target_modules: int = 3  # Minimum number of modules to target
    max_target_modules: int = 8  # Maximum number of modules to target
    
    # Search space optimization
    search_strategy: str = "progressive"  # "progressive", "focused", "minimal", "comprehensive"
    focus_on_task: Optional[str] = None  # Focus on specific ELMB task
    
    # Progressive optimization
    enable_progressive_refinement: bool = True
    initial_trials_per_stage: int = 10
    refinement_stages: int = 3


class DoRAModuleOptimizer:
    """Optimizer for DoRA target module selection"""
    
    # All possible DoRA target modules for the model
    ALL_MODULES = [
        "embed_tokens",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head"
    ]
    
    # Predefined module groups for efficient search
    MODULE_GROUPS = {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "embeddings": ["embed_tokens", "lm_head"],
        "core": ["q_proj", "v_proj", "gate_proj", "down_proj"],
        "extended": ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "minimal": ["q_proj", "v_proj", "down_proj"]
    }
    
    @classmethod
    def suggest_target_modules(cls, trial: optuna.Trial, min_modules: int = 3, max_modules: int = 8) -> List[str]:
        """Suggest DoRA target modules using intelligent sampling"""
        
        # Strategy 1: Use predefined groups (more efficient)
        use_groups = trial.suggest_categorical("use_module_groups", [True, False])
        
        if use_groups:
            # Select from predefined efficient combinations
            group_name = trial.suggest_categorical("module_group", list(cls.MODULE_GROUPS.keys()))
            selected_modules = cls.MODULE_GROUPS[group_name].copy()
            
            # Optionally add/remove modules from the group
            for module in cls.ALL_MODULES:
                if module not in selected_modules:
                    add_module = trial.suggest_categorical(f"add_{module}", [True, False])
                    if add_module:
                        selected_modules.append(module)
                elif len(selected_modules) > min_modules:
                    keep_module = trial.suggest_categorical(f"keep_{module}", [True, False])
                    if not keep_module:
                        selected_modules.remove(module)
        else:
            # Strategy 2: Individual module selection
            selected_modules = []
            for module in cls.ALL_MODULES:
                include = trial.suggest_categorical(f"include_{module}", [True, False])
                if include:
                    selected_modules.append(module)
            
            # Ensure minimum number of modules
            if len(selected_modules) < min_modules:
                remaining = [m for m in cls.ALL_MODULES if m not in selected_modules]
                for i in range(min_modules - len(selected_modules)):
                    if remaining:
                        selected_modules.append(remaining[i])
        
        # Ensure we don't exceed maximum
        if len(selected_modules) > max_modules:
            selected_modules = selected_modules[:max_modules]
        
        return selected_modules
    
    @classmethod
    def get_module_efficiency_score(cls, modules: List[str]) -> float:
        """Calculate efficiency score for module combination"""
        
        # Efficiency heuristics based on research
        efficiency_weights = {
            "q_proj": 0.25,      # High impact
            "v_proj": 0.25,      # High impact
            "gate_proj": 0.20,   # Good impact
            "down_proj": 0.15,   # Good impact
            "k_proj": 0.10,      # Medium impact
            "o_proj": 0.10,      # Medium impact
            "up_proj": 0.08,     # Lower impact
            "embed_tokens": 0.05, # Lower impact
            "lm_head": 0.05      # Lower impact
        }
        
        total_weight = sum(efficiency_weights.get(module, 0.01) for module in modules)
        
        # Bonus for common efficient combinations
        if set(modules) & set(cls.MODULE_GROUPS["core"]) == set(cls.MODULE_GROUPS["core"]):
            total_weight *= 1.2
        
        if set(modules) & set(cls.MODULE_GROUPS["attention"]) == set(cls.MODULE_GROUPS["attention"]):
            total_weight *= 1.1
        
        return total_weight


class SearchSpaceManager:
    """Manages different search space strategies"""
    
    @staticmethod
    def get_search_space(strategy: str, focus_task: Optional[str] = None) -> Dict[str, Dict]:
        """Get hyperparameter search space based on strategy"""
        
        if strategy == "minimal":
            return SearchSpaceManager._get_minimal_space()
        elif strategy == "focused":
            return SearchSpaceManager._get_focused_space(focus_task)
        elif strategy == "progressive":
            return SearchSpaceManager._get_progressive_space()
        elif strategy == "comprehensive":
            return SearchSpaceManager._get_comprehensive_space()
        else:
            return SearchSpaceManager._get_focused_space()
    
    @staticmethod
    def _get_minimal_space() -> Dict[str, Dict]:
        """Minimal search space - most impactful parameters only"""
        return {
            "per_device_train_batch_size": {"type": "int", "low": 1, "high": 4},
            "gradient_accumulation_steps": {"type": "int", "low": 2, "high": 8},
            "lora_alpha": {"type": "int", "low": 16, "high": 48},
            "block_size": {"type": "categorical", "choices": [1024, 2048]},
            "warmup_steps": {"type": "int", "low": 10, "high": 50}
        }
    
    @staticmethod
    def _get_focused_space(focus_task: Optional[str] = None) -> Dict[str, Dict]:
        """Focused search space - task-specific optimization"""
        
        base_space = {
            "per_device_train_batch_size": {"type": "int", "low": 1, "high": 6},
            "gradient_accumulation_steps": {"type": "int", "low": 1, "high": 12},
            "lora_alpha": {"type": "int", "low": 8, "high": 64},
            "lora_dropout": {"type": "float", "low": 0.0, "high": 0.2},
            "block_size": {"type": "categorical", "choices": [512, 1024, 2048]},
            "warmup_steps": {"type": "int", "low": 0, "high": 80},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.05}
        }
        
        # Task-specific adjustments
        if focus_task == "elmb_reasoning":
            # Reasoning benefits from larger context and more careful training
            base_space["block_size"]["choices"] = [1024, 2048]
            base_space["warmup_steps"]["high"] = 100
            base_space["weight_decay"]["high"] = 0.1
        elif focus_task == "elmb_functioncalling":
            # Function calling benefits from precise training
            base_space["lora_dropout"]["high"] = 0.1
            base_space["per_device_train_batch_size"]["high"] = 4
        elif focus_task == "elmb_roleplay":
            # Roleplay benefits from creative freedom
            base_space["lora_dropout"]["high"] = 0.3
            base_space["lora_alpha"]["high"] = 96
        elif focus_task == "elmb_chatrag":
            # RAG benefits from context processing
            base_space["block_size"]["choices"] = [1024, 2048]
            base_space["per_device_train_batch_size"]["low"] = 1
            base_space["per_device_train_batch_size"]["high"] = 3
        
        return base_space
    
    @staticmethod
    def _get_progressive_space() -> Dict[str, Dict]:
        """Progressive search space - starts broad, narrows down"""
        return {
            "per_device_train_batch_size": {"type": "int", "low": 1, "high": 8},
            "gradient_accumulation_steps": {"type": "int", "low": 1, "high": 16},
            "lora_alpha": {"type": "int", "low": 8, "high": 64},
            "lora_dropout": {"type": "float", "low": 0.0, "high": 0.3},
            "block_size": {"type": "categorical", "choices": [512, 1024, 2048]},
            "warmup_steps": {"type": "int", "low": 0, "high": 100},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
            "lr_scheduler_type": {"type": "categorical", "choices": ["linear", "cosine", "constant"]},
            "dataloader_num_workers": {"type": "int", "low": 1, "high": 4}
        }
    
    @staticmethod
    def _get_comprehensive_space() -> Dict[str, Dict]:
        """Comprehensive search space - all parameters"""
        return {
            "per_device_train_batch_size": {"type": "int", "low": 1, "high": 8},
            "gradient_accumulation_steps": {"type": "int", "low": 1, "high": 16},
            "lora_alpha": {"type": "int", "low": 8, "high": 64},
            "lora_dropout": {"type": "float", "low": 0.0, "high": 0.3},
            "block_size": {"type": "categorical", "choices": [512, 1024, 2048]},
            "warmup_steps": {"type": "int", "low": 0, "high": 100},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
            "lr_scheduler_type": {"type": "categorical", "choices": ["linear", "cosine", "constant"]},
            "dataloader_num_workers": {"type": "int", "low": 1, "high": 4},
            "preprocessing_num_workers": {"type": "int", "low": 32, "high": 256, "step": 32},
            "deepspeed_config": {"type": "categorical", "choices": [
                "configs/ds_config_zero0_no_offload.json",
                "configs/ds_config_zero2.json"
            ]},
            "logging_steps": {"type": "int", "low": 10, "high": 50},
            "save_steps": {"type": "int", "low": 1000, "high": 10000, "step": 1000},
            "validation_split_percentage": {"type": "float", "low": 0.0, "high": 10.0}
        }


class SubsetBayesianOptimizer(BayesianHyperparameterOptimizer):
    """Enhanced optimizer with subset optimization capabilities"""
    
    def __init__(self, config: SubsetOptimizationConfig):
        super().__init__(config)
        self.config = config
        self.module_optimizer = DoRAModuleOptimizer()
        self.search_space = SearchSpaceManager.get_search_space(
            config.search_strategy, 
            config.focus_on_task
        )
        self.best_modules_so_far = None
        self.stage_results = []
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, any]:
        """Enhanced hyperparameter suggestion with subset optimization"""
        
        hyperparams = {}
        
        # 1. Optimize DoRA target modules if enabled
        if self.config.optimize_target_modules:
            target_modules = self.module_optimizer.suggest_target_modules(
                trial, 
                self.config.min_target_modules, 
                self.config.max_target_modules
            )
            hyperparams["lora_target_modules"] = ",".join(target_modules)
            
            # Store module efficiency score for analysis
            efficiency_score = self.module_optimizer.get_module_efficiency_score(target_modules)
            trial.set_user_attr("module_efficiency_score", efficiency_score)
        else:
            # Use default modules
            hyperparams["lora_target_modules"] = "embed_tokens,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head"
        
        # 2. Suggest hyperparameters from defined search space
        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "int":
                value = trial.suggest_int(
                    param_name, 
                    param_config["low"], 
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
            elif param_config["type"] == "float":
                value = trial.suggest_float(
                    param_name, 
                    param_config["low"], 
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_config["type"] == "categorical":
                value = trial.suggest_categorical(param_name, param_config["choices"])
            else:
                continue
            
            hyperparams[param_name] = value
        
        # 3. Set default values for parameters not in search space
        default_params = {
            "deepspeed_config": "configs/ds_config_zero0_no_offload.json",
            "logging_steps": 20,
            "save_steps": 5000,
            "validation_split_percentage": 0,
            "dataloader_num_workers": 1,
            "preprocessing_num_workers": 128,
            "lr_scheduler_type": "linear"
        }
        
        for param, value in default_params.items():
            if param not in hyperparams:
                hyperparams[param] = value
        
        return hyperparams
    
    def calculate_task_specific_score(self, scores: Dict[str, float]) -> float:
        """Calculate improvement score with optional task focus"""
        
        baseline_scores = self.get_baseline_scores()
        
        if self.config.focus_on_task and self.config.focus_on_task in scores:
            # Focus primarily on one task
            main_improvement = scores[self.config.focus_on_task] - baseline_scores.get(self.config.focus_on_task, 0.25)
            other_improvements = sum(
                scores.get(task, 0.0) - baseline_scores.get(task, 0.25)
                for task in ["elmb_roleplay", "elmb_reasoning", "elmb_functioncalling", "elmb_chatrag"]
                if task != self.config.focus_on_task
            )
            # Weight focused task 3x more than others
            return main_improvement * 3.0 + other_improvements * 0.25
        else:
            # Standard total improvement
            return super().calculate_improvement_score(scores)
    
    def objective(self, trial: optuna.Trial) -> float:
        """Enhanced objective function with subset optimization"""
        
        try:
            # Suggest hyperparameters (including DoRA modules)
            hyperparams = self.suggest_hyperparameters(trial)
            
            # Run training
            dora_path = self.run_training(trial, hyperparams)
            
            # Merge DoRA weights
            merged_path = self.merge_dora_weights(dora_path)
            
            # Evaluate model
            scores = self.evaluate_model(merged_path)
            
            # Calculate improvement score (with optional task focus)
            improvement_score = self.calculate_task_specific_score(scores)
            
            # Store additional metrics
            trial.set_user_attr("scores", scores)
            trial.set_user_attr("hyperparams", hyperparams)
            trial.set_user_attr("model_path", merged_path)
            trial.set_user_attr("target_modules", hyperparams.get("lora_target_modules", ""))
            
            # Clean up to save space
            if os.path.exists(dora_path):
                import shutil
                shutil.rmtree(dora_path, ignore_errors=True)
            
            logger.info(f"Trial {trial.number} completed with improvement score: {improvement_score}")
            return improvement_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            return -1000.0


def main():
    """Main function for subset optimization"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run subset-based Bayesian optimization for DoRA fine-tuning')
    parser.add_argument('--strategy', type=str, default='progressive',
                        choices=['minimal', 'focused', 'progressive', 'comprehensive'],
                        help='Search strategy')
    parser.add_argument('--focus-task', type=str, default=None,
                        choices=['elmb_roleplay', 'elmb_reasoning', 'elmb_functioncalling', 'elmb_chatrag'],
                        help='Focus optimization on specific task')
    parser.add_argument('--optimize-modules', action='store_true', default=True,
                        help='Optimize DoRA target modules')
    parser.add_argument('--n-trials', type=int, default=15,
                        help='Number of trials')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for training and evaluation')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SubsetOptimizationConfig(
        study_name=f"subset_dora_optimization_{args.strategy}",
        n_trials=args.n_trials,
        device=args.device,
        search_strategy=args.strategy,
        focus_on_task=args.focus_task,
        optimize_target_modules=args.optimize_modules,
        evaluation_limit=25  # Smaller for faster iteration
    )
    
    # Create optimizer
    optimizer = SubsetBayesianOptimizer(config)
    
    print(f"Starting subset optimization:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Focus task: {args.focus_task or 'All tasks'}")
    print(f"  Optimize modules: {args.optimize_modules}")
    print(f"  Trials: {args.n_trials}")
    
    # Run optimization
    study = optimizer.run_optimization()
    
    # Save results
    optimizer.save_results(study)
    
    print(f"\nSubset optimization completed!")
    print(f"Best improvement score: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")


if __name__ == "__main__":
    main() 