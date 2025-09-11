conda activate lmflow
conda install mpi4py

cd Data-Filtering-Challenge

bash train.sh --dataset_path wesley/preselect_detokenized2

bash ./scripts/run_merge_dora.sh \
 --model_name_or_path data4elm/Llama-400M-12L \
 --lora_model_path output_models/wesley_detokenized2 \
 --output_model_path output_models/wesley_detokenized2_merged

cd lm-evaluation-harness
pip install -e . 

lm_eval --model hf \
    --model_args pretrained=../output_models/wesley_detokenized2_merged,trust_remote_code=True \
    --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/baseline_elmb
    