git clone https://github.com/The-X-School/Data-Filtering-Challenge
cd Data-Filtering-Challenge 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
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


#run data mixtures
python hf_climblab.py

# create a wandb account and get an api key  
# https://docs.wandb.ai/quickstart/)  
# https://wandb.ai/authorize


# Then run this code:
export WANDB_API_KEY=
pip install wandb
wandb login

echo "run train.sh with changed data path: RegMix\ and\ input\ stuff\ --\>\ evans\ suff/JsonL_Data/"

echo "
 bash ./scripts/run_merge_dora.sh \
 --model_name_or_path data4elm/Llama-400M-12L \
 --lora_model_path output_models/finetune \
 --output_model_path output_models/dora_merged"