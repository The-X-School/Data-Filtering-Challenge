#git clone https://github.com/The-X-School/Data-Filtering-Challenge
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
pip install -e .

# extra package to install if needed:
pip install py-cpuinfo
pip install fastparquet==2024.11.0  #check this
pip3 install -r requirements.txt

git branch -r
git checkout evans-basement

cd RegMix

python hf_climblab.py

mv format_data.py JsonL_Data/
echo "moved format_data.py to JsonL_Data/"

cd JsonL_Data/

for i in {1..20}; do

    i_path="cluster_$i.json"
    echo $i_path
    python format_data.py $i_path
done

mv format_data.py ..
cd ../..

bash train.sh

