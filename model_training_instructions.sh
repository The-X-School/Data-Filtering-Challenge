# Merge dora weights
bash train.sh --dataset_path wesley/preselect_tokenized_wesley
bash ./scripts/run_merge_dora.sh \
 --model_name_or_path data4elm/Llama-400M-12L \
 --lora_model_path output_models/wesley_tokenized \
 --output_model_path output_models/wesley_tokenized_merged
 