#!/usr/bin/env python
# coding=utf-8
"""
Test script to validate the Bayesian optimization setup
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import optuna
        print(f"✓ optuna {optuna.__version__}")
    except ImportError as e:
        print(f"✗ optuna import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ yaml")
    except ImportError as e:
        print(f"✗ yaml import failed: {e}")
        return False
    
    try:
        from bayesian_hyperparameter_tuning import BayesianHyperparameterOptimizer, OptimizationConfig
        print("✓ bayesian_hyperparameter_tuning")
    except ImportError as e:
        print(f"✗ bayesian_hyperparameter_tuning import failed: {e}")
        return False
    
    return True


def test_file_structure():
    """Test if required files and directories exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "bayesian_hyperparameter_tuning.py",
        "run_optimization.py", 
        "optimization_config.yaml",
        "optimization_requirements.txt",
        "examples/finetune.py",
        "lm-evaluation-harness"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            all_exist = False
    
    return all_exist


def test_config_loading():
    """Test if configuration loading works"""
    print("\nTesting configuration loading...")
    
    try:
        from run_optimization import load_config
        config = load_config("optimization_config.yaml")
        
        print(f"✓ Config loaded successfully")
        print(f"  Study name: {config.study_name}")
        print(f"  Base model: {config.base_model_path}")
        print(f"  Device: {config.device}")
        print(f"  Fixed learning rate: {config.fixed_learning_rate}")
        print(f"  Fixed lora_r: {config.fixed_lora_r}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_optuna_setup():
    """Test basic Optuna functionality"""
    print("\nTesting Optuna setup...")
    
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        
        # Create a simple test study
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            
            study = optuna.create_study(
                direction="minimize",
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(),
                storage=f"sqlite:///{db_path}"
            )
            
            study.optimize(objective, n_trials=5)
            
        print(f"✓ Optuna test study completed")
        print(f"  Best value: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")
        
        return True
    except Exception as e:
        print(f"✗ Optuna test failed: {e}")
        return False


def test_hyperparameter_suggestion():
    """Test hyperparameter suggestion functionality"""
    print("\nTesting hyperparameter suggestion...")
    
    try:
        import optuna
        from bayesian_hyperparameter_tuning import BayesianHyperparameterOptimizer, OptimizationConfig
        
        config = OptimizationConfig()
        optimizer = BayesianHyperparameterOptimizer(config)
        
        # Create a mock trial
        study = optuna.create_study()
        trial = study.ask()
        
        hyperparams = optimizer.suggest_hyperparameters(trial)
        
        print("✓ Hyperparameter suggestion works")
        print("  Sample suggested hyperparameters:")
        for key, value in list(hyperparams.items())[:5]:  # Show first 5
            print(f"    {key}: {value}")
        
        return True
    except Exception as e:
        print(f"✗ Hyperparameter suggestion failed: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    
    # Test nvidia-smi command
    gpu_available = os.system("nvidia-smi > /dev/null 2>&1") == 0
    
    if gpu_available:
        print("✓ GPU detected (nvidia-smi works)")
        
        # Try to import torch and check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")
                return True
            else:
                print("⚠ CUDA not available in PyTorch")
                return False
        except ImportError:
            print("⚠ PyTorch not installed, cannot test CUDA")
            return False
    else:
        print("⚠ No GPU detected (will use CPU)")
        return False


def create_minimal_test_config():
    """Create a minimal configuration for testing"""
    print("\nCreating minimal test configuration...")
    
    minimal_config = {
        "study": {
            "name": "test_optimization",
            "n_trials": 2,
            "timeout": 1800,  # 30 minutes
            "enable_pruning": False
        },
        "model": {
            "base_model_path": "data4elm/Llama-400M-12L",
            "dataset_path": "data/filtered_output",
            "fixed_hyperparameters": {
                "num_train_epochs": 1,
                "learning_rate": 1e-5,
                "lora_r": 16
            }
        },
        "output": {
            "base_dir": "test_optimization_results",
            "models_dir": "test_optimization_models"
        },
        "evaluation": {
            "limit": 5,  # Very small for quick testing
            "device": "cpu",  # Safe default
            "tasks": ["elmb_roleplay"]  # Test with just one task
        },
        "hyperparameter_ranges": {
            "per_device_train_batch_size": {
                "type": "int",
                "low": 1,
                "high": 2
            },
            "block_size": {
                "type": "categorical",
                "choices": [512, 1024]
            },
            "lora_alpha": {
                "type": "int",
                "low": 16,
                "high": 32
            }
        }
    }
    
    try:
        import yaml
        with open("test_optimization_config.yaml", 'w') as f:
            yaml.dump(minimal_config, f, default_flow_style=False)
        
        print("✓ Minimal test config created: test_optimization_config.yaml")
        return True
    except Exception as e:
        print(f"✗ Failed to create test config: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("BAYESIAN OPTIMIZATION SETUP VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Configuration Loading", test_config_loading),
        ("Optuna Setup", test_optuna_setup),
        ("Hyperparameter Suggestion", test_hyperparameter_suggestion),
        ("GPU Availability", test_gpu_availability),
        ("Minimal Test Config", create_minimal_test_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The optimization setup is ready to use.")
        print("\nNext steps:")
        print("1. Review and adjust optimization_config.yaml")
        print("2. Run: python run_optimization.py --config test_optimization_config.yaml")
        print("3. For full optimization: python run_optimization.py")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please fix the issues before running optimization.")
        
        if not any(name == "Package Imports" and result for name, result in results):
            print("\nTo install required packages:")
            print("pip install -r optimization_requirements.txt")
        
        if not any(name == "File Structure" and result for name, result in results):
            print("\nMake sure you're running from the correct directory and all files are present.")


if __name__ == "__main__":
    main() 