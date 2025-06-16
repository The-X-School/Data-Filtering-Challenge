# Model evaluation instructions 
#I changed a few lines of code, it worked when I ran it on command line on Mac

git clone https://github.com/The-X-School/Data-Filtering-Challenge
cd Data-Filtering-Challenge 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

#To Check if conda is installed:
conda --version 
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .


# extra package to install if needed:
pip install py-cpuinfo
pip install -r requirements.txt

# Evaluation (I removed this line of code: `cache_dir=~/.cache` because it wasn't working):
cd ~/Data-Filtering-Challenge/lm-evaluation-harness
pip install -e . 

# Note: The command below evaluates the baseline model.
# When you have your own fine-tuned model, replace 'data4elm/Llama-400M-12L' 
# with 'output_models/dora_merged' (path to your merged model directory)
lm_eval --model hf \
    --model_args pretrained=../output_models/dora_merged,trust_remote_code=True \
    --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path ./eval_results/baseline_elmb