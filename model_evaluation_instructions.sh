# Model evaluation instructions 
#I changed a few lines of code, it worked when I ran it on command line on Mac

# Evaluation (I removed this line of code: `cache_dir=~/.cache` because it wasn't working):
cd lm-evaluation-harness
pip install -e . 

# Note: The command below evaluates the baseline model.
# When you have your own fine-tuned model, replace 'data4elm/Llama-400M-12L' ``
# with '../output_models/dora_merged' (path to your merged model directory)
# When you have your own fine-tuned model, replace 'data4elm/Llama-400M-12L' 
# with 'output_models/dora_merged' (path to your merged model directory)
lm_eval --model hf \
    --model_args pretrained=../output_models/wesley_gsm8k_merged,trust_remote_code=True \
    --model_args pretrained=../output_models/wesley_preselect8709_merged,trust_remote_code=True \
    --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
    --device cuda:0 \
    --batch_size 1 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/wesley_gsm8k
    
    --output_path ./eval_results/wesley_preselect8709_merged