bash train.sh --dataset_path wesley/preselect_tokenized_wesley2

bash ./scripts/run_merge_dora.sh \
 --model_name_or_path data4elm/Llama-400M-12L \
 --lora_model_path output_models/wesley_tokenized2 \
 --output_model_path output_models/wesley_tokenized2_merged

cd ~/Data-Filtering-Challenge/lm-evaluation-harness
pip install -e . 

lm_eval --model hf \
    --model_args pretrained=../output_models/dora_merged,trust_remote_code=True \
    --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/baseline_elmb
    