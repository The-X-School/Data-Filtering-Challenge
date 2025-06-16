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
pip3 install -r requirements.txt

# create a wandb account and get an api key  
# https://docs.wandb.ai/quickstart/)  
# https://wandb.ai/authorize

# Then run this code:
export WANDB_API_KEY=<your_api_key>
pip install wandb
wandb login


# Merge dora weights
bash train.sh
bash ./scripts/run_merge_dora.sh \
 --model_name_or_path Qwen/Qwen1.5-1.8B \
 --lora_model_path output_models/dora \
 --output_model_path output_models/dora_merged \
