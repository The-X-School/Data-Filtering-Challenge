#!/usr/bin/env python
"""
Convenient runner for subset-based DoRA optimization.
Provides preset configurations for different optimization strategies.
"""

import os
import sys
import argparse
import subprocess
from typing import List, Dict


def get_strategy_configs() -> Dict[str, Dict]:
    """Get predefined strategy configurations"""
    return {
        "quick": {
            "strategy": "minimal",
            "n_trials": 8,
            "description": "Quick test with minimal search space (2-3 hours)",
            "modules": True,
            "focus_task": None
        },
        "focused": {
            "strategy": "focused", 
            "n_trials": 15,
            "description": "Focused optimization for balanced performance (4-6 hours)",
            "modules": True,
            "focus_task": None
        },
        "roleplay": {
            "strategy": "focused",
            "n_trials": 15,
            "description": "Optimized specifically for roleplay task (4-6 hours)",
            "modules": True,
            "focus_task": "elmb_roleplay"
        },
        "reasoning": {
            "strategy": "focused",
            "n_trials": 15,
            "description": "Optimized specifically for reasoning task (4-6 hours)", 
            "modules": True,
            "focus_task": "elmb_reasoning"
        },
        "function_calling": {
            "strategy": "focused",
            "n_trials": 15,
            "description": "Optimized specifically for function calling task (4-6 hours)",
            "modules": True,
            "focus_task": "elmb_functioncalling"
        },
        "rag": {
            "strategy": "focused", 
            "n_trials": 15,
            "description": "Optimized specifically for RAG task (4-6 hours)",
            "modules": True,
            "focus_task": "elmb_chatrag"
        },
        "progressive": {
            "strategy": "progressive",
            "n_trials": 25,
            "description": "Progressive search with refinement (6-8 hours)",
            "modules": True,
            "focus_task": None
        },
        "comprehensive": {
            "strategy": "comprehensive",
            "n_trials": 40,
            "description": "Comprehensive search of all parameters (10-16 hours)",
            "modules": True,
            "focus_task": None
        },
        "modules_only": {
            "strategy": "minimal",
            "n_trials": 12,
            "description": "Focus only on DoRA module optimization (3-4 hours)",
            "modules": True,
            "focus_task": None
        }
    }


def build_command(preset: str, config: Dict, device: str = "cuda:0", custom_trials: int = None) -> List[str]:
    """Build the optimization command"""
    
    cmd = ["python", "subset_optimization.py"]
    
    # Add strategy
    cmd.extend(["--strategy", config["strategy"]])
    
    # Add number of trials
    n_trials = custom_trials or config["n_trials"]
    cmd.extend(["--n-trials", str(n_trials)])
    
    # Add device
    cmd.extend(["--device", device])
    
    # Add focus task if specified
    if config["focus_task"]:
        cmd.extend(["--focus-task", config["focus_task"]])
    
    # Add module optimization
    if config["modules"]:
        cmd.append("--optimize-modules")
    
    return cmd


def validate_environment():
    """Validate that required files exist"""
    required_files = [
        "subset_optimization.py",
        "bayesian_hyperparameter_tuning.py",
        "subset_optimization_config.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    return True


def show_available_presets():
    """Show all available optimization presets"""
    configs = get_strategy_configs()
    
    print("\n🎯 Available Optimization Presets:\n")
    
    for preset_name, config in configs.items():
        print(f"  {preset_name:15} - {config['description']}")
        print(f"                    Strategy: {config['strategy']}, Trials: {config['n_trials']}")
        if config['focus_task']:
            print(f"                    Focus: {config['focus_task']}")
        print()


def estimate_resources(preset: str, n_trials: int = None) -> Dict:
    """Estimate resource requirements"""
    configs = get_strategy_configs()
    
    if preset not in configs:
        return {}
    
    config = configs[preset]
    trials = n_trials or config['n_trials']
    
    # Rough estimates based on strategy
    time_per_trial = {
        "minimal": 15,      # 15 minutes per trial
        "focused": 20,      # 20 minutes per trial  
        "progressive": 25,  # 25 minutes per trial
        "comprehensive": 30 # 30 minutes per trial
    }
    
    base_time = time_per_trial.get(config['strategy'], 20)
    total_minutes = base_time * trials
    total_hours = total_minutes / 60
    
    return {
        "trials": trials,
        "estimated_minutes": total_minutes,
        "estimated_hours": round(total_hours, 1),
        "strategy": config['strategy'],
        "gpu_required": True,
        "disk_space_gb": 2 + (trials * 0.1)  # Base + per trial
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run subset-based DoRA optimization with preset configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_subset_optimization.py quick                    # Quick test (8 trials, 2-3 hours)
  python run_subset_optimization.py focused                  # Balanced optimization (15 trials)
  python run_subset_optimization.py roleplay                 # Focus on roleplay task
  python run_subset_optimization.py progressive --trials 30  # Progressive with 30 trials
  python run_subset_optimization.py --list                   # Show all available presets
        """
    )
    
    parser.add_argument("preset", nargs="?", help="Optimization preset to run")
    parser.add_argument("--list", action="store_true", help="Show available presets")
    parser.add_argument("--device", default="cuda:0", help="Device for training (default: cuda:0)")
    parser.add_argument("--trials", type=int, help="Override number of trials")
    parser.add_argument("--dry-run", action="store_true", help="Show command without running")
    parser.add_argument("--estimate", action="store_true", help="Show resource estimates")
    
    args = parser.parse_args()
    
    # Show available presets
    if args.list:
        show_available_presets()
        return
    
    # Validate environment
    if not validate_environment():
        return
    
    # Check if preset specified
    if not args.preset:
        print("❌ Please specify a preset or use --list to see available options")
        parser.print_help()
        return
    
    # Get configurations
    configs = get_strategy_configs()
    
    if args.preset not in configs:
        print(f"❌ Unknown preset '{args.preset}'. Available presets:")
        for name in configs.keys():
            print(f"   {name}")
        return
    
    config = configs[args.preset]
    
    # Show resource estimates
    if args.estimate:
        resources = estimate_resources(args.preset, args.trials)
        print(f"\n📊 Resource Estimates for '{args.preset}' preset:")
        print(f"   Trials: {resources['trials']}")
        print(f"   Estimated time: {resources['estimated_hours']} hours ({resources['estimated_minutes']} minutes)")
        print(f"   Strategy: {resources['strategy']}")
        print(f"   GPU required: {'Yes' if resources['gpu_required'] else 'No'}")
        print(f"   Disk space: ~{resources['disk_space_gb']:.1f} GB")
        print()
        return
    
    # Build and show command
    cmd = build_command(args.preset, config, args.device, args.trials)
    
    print(f"🚀 Running DoRA subset optimization: '{args.preset}'")
    print(f"   Description: {config['description']}")
    print(f"   Strategy: {config['strategy']}")
    print(f"   Trials: {args.trials or config['n_trials']}")
    if config['focus_task']:
        print(f"   Focus task: {config['focus_task']}")
    print(f"   Device: {args.device}")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    if args.dry_run:
        print("🔍 Dry run mode - command not executed")
        return
    
    # Show resource estimates
    resources = estimate_resources(args.preset, args.trials)
    print(f"⏱️  Estimated time: ~{resources['estimated_hours']} hours")
    print()
    
    # Confirm before running long optimizations
    if resources['estimated_hours'] > 4:
        response = input(f"⚠️  This optimization will take approximately {resources['estimated_hours']} hours. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Optimization cancelled.")
            return
    
    # Run the optimization
    try:
        print("🔄 Starting optimization...")
        result = subprocess.run(cmd, check=True)
        print("✅ Optimization completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Optimization failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main() 