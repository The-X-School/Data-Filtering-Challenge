# Model evaluation instructions 
# I changed a few lines of code, it worked when I ran it on command line on Mac

git clone https://github.com/The-X-School/Data-Filtering-Challenge.git
cd Data-Filtering-Challenge
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .

# extra package to install if needed:
pip install py-cpuinfo

# create a wandb account and get an api key  
# https://docs.wandb.ai/quickstart/
# https://wandb.ai/authorize

# Then run this code:
export WANDB_API_KEY=<your_api_key>
pip install wandb
wandb login

# here's the training script, update it with the filtered dataset before running it:
# https://github.com/The-X-School/Data-Filtering-Challenge/blob/main/train.sh
bash train.sh

# Merge dora weights, used when fine-tuning with our dataset
bash ./scripts/run_merge_dora.sh \
 --model_name_or_path Qwen/Qwen1.5-1.8B \
 --lora_model_path output_models/dora \
 --output_model_path output_models/dora_merged \

# Evaluation (I removed this line of code: `cache_dir=~/.cache` because it wasn't working):
cd Data-Filtering-Challenge/lm-evaluation-harness
pip install -e . 

lm_eval --model hf \
    --model_args pretrained=data4elm/Llama-400M-12L,trust_remote_code=True \
    --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/test_elmb