# Quick Start Guide: Bayesian Hyperparameter Optimization

This guide will get you up and running with automated hyperparameter optimization for DoRA fine-tuning in just a few steps.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r optimization_requirements.txt
```

### 2. Validate Setup
```bash
python test_optimization_setup.py
```
This will check if everything is properly configured and create a test configuration.

### 3. Run Test Optimization (Recommended)
```bash
python run_optimization.py --config test_optimization_config.yaml
```
This runs a quick test with 2 trials to ensure everything works.

### 4. Run Full Optimization
```bash
python run_optimization.py
```
This runs the full optimization with 30 trials (configurable).

## ⚡ One-Minute Start

If you just want to start optimization immediately:

```bash
# Install and test
pip install -r optimization_requirements.txt
python test_optimization_setup.py

# If all tests pass, start optimization
python run_optimization.py
```

## 📊 What Gets Optimized

The system automatically optimizes these hyperparameters while keeping the competition-required fixed values:

### ✅ Fixed (Competition Rules)
- `learning_rate`: 1e-5
- `lora_r`: 16  
- `num_train_epochs`: 1

### 🔧 Optimized Automatically
- **Batch sizes**: `per_device_train_batch_size`, `gradient_accumulation_steps`
- **Model config**: `block_size`, `lora_alpha`, `lora_dropout`
- **Optimization**: `warmup_steps`, `weight_decay`, `lr_scheduler_type`
- **Data processing**: `dataloader_num_workers`, `preprocessing_num_workers`
- **Infrastructure**: DeepSpeed configuration, logging intervals

## 📈 How It Works

1. **Baseline Evaluation**: First evaluates the base model on ELMB
2. **Bayesian Search**: Uses Optuna to suggest hyperparameters intelligently
3. **Training Loop**: For each trial:
   - Trains model with suggested hyperparameters
   - Merges DoRA weights with base model
   - Evaluates on ELMB benchmark
   - Calculates improvement score
4. **Optimization**: Uses results to guide next hyperparameter suggestions
5. **Results**: Saves best model and hyperparameters

## 🎯 Expected Results

After optimization completes, you'll get:

```
optimization_results/
├── final_results/
│   ├── best_model.json       # Best hyperparameters and scores
│   ├── study_statistics.json # Optimization summary
│   └── all_trials.json      # Detailed results
└── bayesian_optimization.log # Full execution log
```

The best model will be available at the path specified in `best_model.json`.

## ⚙️ Common Customizations

### Change Number of Trials
Edit `optimization_config.yaml`:
```yaml
study:
  n_trials: 50  # Increase for more thorough search
```

### Reduce Evaluation Time
```yaml
evaluation:
  limit: 25  # Reduce samples per task
```

### CPU-Only Mode
```bash
python run_optimization.py --device cpu
```

### Custom Dataset
Edit `optimization_config.yaml`:
```yaml
model:
  dataset_path: "path/to/your/dataset"
```

## 🔍 Monitoring Progress

### Real-time Log Monitoring
```bash
tail -f bayesian_optimization.log
```

### Check Current Best
```bash
python -c "
import optuna
study = optuna.load_study(
    study_name='dora_hyperparameter_optimization',
    storage='sqlite:///optimization_results/optimization.db'
)
print(f'Best score: {study.best_value:.4f}')
print(f'Trials completed: {len(study.trials)}')
"
```

## 🚨 Troubleshooting

### Out of Memory
```yaml
# In optimization_config.yaml, reduce batch sizes:
hyperparameter_ranges:
  per_device_train_batch_size:
    high: 2  # Reduce from 8
```

### Slow Training
```yaml
# Reduce dataset or evaluation size:
evaluation:
  limit: 10  # Very small for testing
```

### CUDA Issues
```bash
python run_optimization.py --device cpu
```

## 📋 Results Interpretation

The system optimizes the **improvement score**:
```
S = S_improve - S_baseline
```

Where:
- `S_improve` = Performance of your fine-tuned model
- `S_baseline` = Performance of the base model
- Higher values = better performance improvement

Example results:
```json
{
  "improvement_score": 0.1234,
  "scores": {
    "elmb_roleplay": 0.85,      # vs baseline 0.80
    "elmb_reasoning": 0.78,     # vs baseline 0.75
    "elmb_functioncalling": 0.82, # vs baseline 0.78
    "elmb_chatrag": 0.79        # vs baseline 0.77
  }
}
```

## 🎯 Next Steps After Optimization

1. **Use Best Hyperparameters**: Apply them to your final training
2. **Full Evaluation**: Run complete ELMB evaluation (without limit)
3. **Submit**: Use the optimized model for competition submission

```bash
# Example: Train final model with best hyperparameters
python examples/finetune.py \
  --model_name_or_path data4elm/Llama-400M-12L \
  --dataset_path data/filtered_output \
  --per_device_train_batch_size 4 \    # From optimization
  --block_size 1024 \                  # From optimization
  --lora_alpha 32 \                    # From optimization
  --learning_rate 1e-5 \               # Fixed
  --lora_r 16 \                        # Fixed
  --num_train_epochs 1                 # Fixed
```

## 🆘 Need Help?

1. **Check the test**: `python test_optimization_setup.py`
2. **Read the full docs**: `BAYESIAN_OPTIMIZATION_README.md`
3. **Check logs**: `bayesian_optimization.log`
4. **Start small**: Use `test_optimization_config.yaml` first

## 💡 Pro Tips

- **Start small**: Always run the test configuration first
- **Monitor resources**: Keep an eye on GPU memory and disk space
- **Backup studies**: The SQLite database contains all your progress
- **Iterative approach**: Start with few trials, then increase
- **Use the best**: Don't forget to apply the optimized hyperparameters to your final model! 