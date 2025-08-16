# create a wandb account and get an api key  
# https://docs.wandb.ai/quickstart/)  
# https://wandb.ai/authorize

# Then run this code:
export WANDB_API_KEY=<WANDB KEY>
pip install wandb
wandb login


# Merge dora weights
bash train.sh --dataset_path data/lila
bash ./scripts/run_merge_dora.sh \
 --model_name_or_path data4elm/Llama-400M-12L \
 --lora_model_path output_models/wesley_lila2 \
 --output_model_path output_models/wesley_lila2_merged
 