# create a wandb account and get an api key  
# https://docs.wandb.ai/quickstart/)  
# https://wandb.ai/authorize

# Then run this code:
export WANDB_API_KEY=216d9ade525d2d950715cbcea4b2cca0a9b6781a
pip install wandb
wandb login


# Merge dora weights
bash train.sh
bash ./scripts/run_merge_dora.sh \
 --model_name_or_path data4elm/Llama-400M-12L \
 --lora_model_path output_models/wesley_preselect10k \
 --output_model_path output_models/wesley_preselect10k_merged
