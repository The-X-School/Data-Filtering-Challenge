
# create a wandb account and get an api key  
# https://docs.wandb.ai/quickstart/)  
# https://wandb.ai/authorize

# Then run this code:
export WANDB_API_KEY=<your_api_key>
pip install wandb
wandb login


# Merge dora weights (I haven't tested this, don't know if its necessary)
bash train.sh
bash ./scripts/run_merge_dora.sh \
 --model_name_or_path Qwen/Qwen1.5-1.8B \
 --lora_model_path output_models/dora \
 --output_model_path output_models/dora_merged \
