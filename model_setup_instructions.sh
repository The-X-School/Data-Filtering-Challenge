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
conda create -n lmflow python=3.10.18 -y
conda activate lmflow
conda install mpi4py
pip install -e .

# extra package to install if needed:
pip install py-cpuinfo
pip install -r requirements.txt
pip install --upgrade datasets pyarrow
