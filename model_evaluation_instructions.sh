# Model evaluation instructions 
#I changed a few lines of code, it worked when I ran it on command line on Mac

git clone https://github.com/The-X-School/Data-Filtering-Challenge
cd Data-Filtering-Challenge 
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .


# extra package to install if needed:
pip install py-cpuinfo

# Evaluation (I removed this line of code: `cache_dir=~/.cache` because it wasn't working):
cd Data-Filtering-Challenge/lm-evaluation-harness
pip install -e . 

lm_eval --model hf \
    --model_args pretrained=[YOUR_MODEL_PATH],trust_remote_code=True \
    --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/test_elmb