#!/bin/bash
set -e

# first clone the github (type this in terminal manually): 
# git clone https://github.com/The-X-School/Data-Filtering-Challenge
# it will ask you to enter your user name and password, put github access key as password

# Enter your Wandb api key manually (wandb.ai/authorize) and login using this:
# export WANDB_API_KEY=<your_api_key>

# Then run this script using:
# bash lambda_setup.sh

# install conda
wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -f -p $HOME/miniconda

# Initialize conda for shell interaction
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
$HOME/miniconda/bin/conda init bash
source ~/.bashrc

# activate conda environment
conda create -n lmflow python=3.9 -y
conda activate lmflow

# install packages
conda install mpi4py -y
pip install -e .
pip install py-cpuinfo
pip install -r requirements.txt

# install wandb, make sure you entered your api key
pip install wandb
wandb login