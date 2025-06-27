# Bayesian Hyperparameter Optimization for DoRA Fine-tuning

This system provides automated hyperparameter tuning for DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tuning using Bayesian optimization. It automatically trains models with different hyperparameter configurations, evaluates them on the ELMB benchmark, and finds the optimal hyperparameters to maximize performance improvement.

## Features

- **Bayesian Optimization**: Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for efficient hyperparameter search
- **ELMB Evaluation**: Automatically evaluates models on all 4 ELMB tasks (roleplay, reasoning, function calling, RAG)
- **Competition Compliance**: Respects fixed hyperparameters required by the competition (learning_rate=1e-5, lora_r=16, num_train_epochs=1)
- **Resource Management**: Includes timeouts and cleanup to manage GPU memory and disk space
- **Progress Tracking**: Comprehensive logging and result storage
- **Flexible Configuration**: YAML-based configuration for easy customization

## Installation

1. Install the required dependencies:
```bash
pip install -r optimization_requirements.txt
```

2. Ensure your environment has the LMFlow dependencies:
```bash
pip install -e .
```

3. Make sure the lm-evaluation-harness is set up:
```bash
cd lm-evaluation-harness
pip install -e .
cd ..
```

## Quick Start

1. **Basic usage with default configuration:**
```bash
python run_optimization.py
```

2. **Custom configuration:**
```bash
python run_optimization.py --config my_config.yaml
```

3. **CPU-only mode:**
```bash
python run_optimization.py --device cpu
```

4. **Resume from previous study:**
```bash
python run_optimization.py --resume
```

## Configuration

The system uses a YAML configuration file (`optimization_config.yaml`) to control all aspects of the optimization:

### Key Configuration Sections

#### Study Configuration
```yaml
study:
  name: "dora_hyperparameter_optimization"
  n_trials: 30                    # Number of trials to run
  timeout: 43200                  # 12 hours timeout
  enable_pruning: true            # Enable early stopping of bad trials
  pruning_warmup_steps: 3
```

#### Model Configuration
```yaml
model:
  base_model_path: "data4elm/Llama-400M-12L"
  dataset_path: "data/filtered_output"
  
  # Fixed hyperparameters (DO NOT CHANGE - competition requirement)
  fixed_hyperparameters:
    num_train_epochs: 1
    learning_rate: 1e-5
    lora_r: 16
```

#### Evaluation Configuration
```yaml
evaluation:
  limit: 50                       # Samples per task for faster evaluation
  device: "cuda:0"                # or "cpu"
  tasks:
    - "elmb_roleplay"
    - "elmb_reasoning"
    - "elmb_functioncalling"
    - "elmb_chatrag"
```

## Hyperparameter Search Space

The system optimizes the following hyperparameters:

### Training Configuration
- `per_device_train_batch_size`: 1-8
- `gradient_accumulation_steps`: 1-16
- `validation_split_percentage`: 0-10%

### Model Architecture
- `block_size`: 512, 1024, or 2048
- `lora_alpha`: 8-64
- `lora_dropout`: 0.0-0.3

### Optimization
- `warmup_steps`: 0-100
- `weight_decay`: 0.0-0.1
- `lr_scheduler_type`: linear, cosine, or constant

### Data Processing
- `dataloader_num_workers`: 1-4
- `preprocessing_num_workers`: 32-256

### Infrastructure
- `deepspeed_config`: ZeRO-0 or ZeRO-2
- `logging_steps`: 10-50
- `save_steps`: 1000-10000

## Output Structure

The system creates the following output structure:

```
optimization_results/
├── baseline_results/           # Baseline model evaluation
├── optimization.db            # Optuna study database
├── final_results/
│   ├── study_statistics.json  # Best trial summary
│   ├── all_trials.json       # Detailed results for all trials
│   └── best_model.json       # Best model information
└── bayesian_optimization.log  # Detailed execution log

optimization_models/
├── trial_0_timestamp/         # Model outputs for each trial
├── trial_1_timestamp_merged/  # Merged models ready for evaluation
└── ...
```

## Understanding Results

### Improvement Score
The system optimizes the **improvement score**: `S = S_improve - S_baseline`

Where:
- `S_improve`: Performance of the fine-tuned model on ELMB
- `S_baseline`: Performance of the base model on ELMB
- The goal is to maximize the total improvement across all 4 ELMB tasks

### Results Files

1. **`best_model.json`**: Contains the best model information:
```json
{
  "trial_number": 15,
  "improvement_score": 0.1234,
  "hyperparameters": {...},
  "model_path": "path/to/best/model",
  "scores": {
    "elmb_roleplay": 0.85,
    "elmb_reasoning": 0.78,
    ...
  },
  "baseline_scores": {...}
}
```

2. **`study_statistics.json`**: Overall optimization statistics
3. **`all_trials.json`**: Detailed results for every trial

## Advanced Usage

### Custom Hyperparameter Ranges

Edit the `hyperparameter_ranges` section in your config file:

```yaml
hyperparameter_ranges:
  per_device_train_batch_size:
    type: "int"
    low: 2      # Custom minimum
    high: 16    # Custom maximum
    
  custom_param:
    type: "float"
    low: 0.001
    high: 0.01
    log: true   # Log scale sampling
```

### Multi-GPU Setup

For multiple GPUs, adjust the DeepSpeed configuration:

```yaml
advanced:
  parallel_evaluation: true

hyperparameter_ranges:
  deepspeed_config:
    type: "categorical"
    choices: 
      - "configs/ds_config_zero2.json"
      - "configs/ds_config_zero3.json"  # For larger models
```

### Resource Constraints

Adjust timeouts based on your hardware:

```yaml
resources:
  training_timeout: 7200    # 2 hours for slower hardware
  evaluation_timeout: 3600  # 1 hour for CPU evaluation
  merge_timeout: 1200       # 20 minutes for large models
```

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch the log file
tail -f bayesian_optimization.log

# Monitor GPU usage
watch nvidia-smi
```

### Study Progress
The system creates an SQLite database that can be queried:

```python
import optuna

study = optuna.load_study(
    study_name="dora_hyperparameter_optimization",
    storage="sqlite:///optimization_results/optimization.db"
)

print(f"Best value so far: {study.best_value}")
print(f"Number of trials: {len(study.trials)}")
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `per_device_train_batch_size` range or use CPU
2. **Slow Training**: Reduce dataset size or evaluation limit
3. **Evaluation Fails**: Check lm-evaluation-harness setup
4. **Disk Space**: Enable `cleanup_intermediate_files` in config

### Debug Mode

For debugging, create a minimal configuration:

```yaml
study:
  n_trials: 3
  timeout: 3600

evaluation:
  limit: 10

hyperparameter_ranges:
  per_device_train_batch_size:
    type: "int"
    low: 1
    high: 2
```

## Best Practices

1. **Start Small**: Begin with a few trials to validate the setup
2. **Monitor Resources**: Keep an eye on GPU memory and disk space
3. **Use Pruning**: Enable pruning to stop bad trials early
4. **Backup Results**: The SQLite database contains all trial history
5. **Gradual Scaling**: Increase evaluation limit and trials progressively

## Integration with Competition Workflow

After optimization completes:

1. **Use Best Model**: The best model path is provided in results
2. **Full Evaluation**: Run complete ELMB evaluation without limit
3. **Submit**: Use the optimized hyperparameters for final training

```bash
# Example: Use best hyperparameters for final training
python examples/finetune.py \
  --model_name_or_path data4elm/Llama-400M-12L \
  --dataset_path data/filtered_output \
  --per_device_train_batch_size 4 \  # From optimization
  --block_size 1024 \                # From optimization
  --lora_alpha 32 \                  # From optimization
  # ... other optimized parameters
```

## Performance Tips

- **GPU Memory**: Use ZeRO-2 or ZeRO-3 for larger models
- **Evaluation Speed**: Reduce evaluation limit during search, use full evaluation for final model
- **Storage**: Enable cleanup to manage disk space
- **Parallelization**: Use multiple GPUs if available

## Citation

If you use this optimization system in your research, please cite:

```bibtex
@software{bayesian_dora_optimization,
  title={Bayesian Hyperparameter Optimization for DoRA Fine-tuning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
``` 