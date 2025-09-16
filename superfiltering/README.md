# Superfiltering for RegMix (20 Clusters)

This repository contains code to apply **Superfiltering** on the [RegMix dataset](https://huggingface.co/datasets/OptimalScale/ClimbLab) clustered into 20 subsets.
The pipeline first slices clusters according to a predicted distribution, then applies **weak-to-strong data filtering** to select high-quality samples.

---

## Repo Structure
data/
├── filtered_clusters_example #filtered data from preselect filtered into 20 clusters

superfiltering-regmix/  
├── superfiltering.py        # Main filtering script  
├── README.md                # Documentation  

requirements.txt #requirements download txt
---

## Setup

Clone the repo and install dependencies:

git clone https://github.com/yourusername/superfiltering-regmix.git  
cd superfiltering-regmix  
pip install -r requirements.txt

If you’re starting from the **raw RegMix clusters** on Hugging Face, you may also need:

pip install fastparquet pandas

---

## Data Preparation

All data was prepared in the previous steps with the preselect model.

## Run Superfiltering

After preparing the cluster JSON files, run:

python superfiltering.py --input_folder new_data --output data/filtered_regmix.json

- `--input_folder` → folder containing your `cluster_*.json` files  
- `--output` → path where the combined, filtered dataset will be saved  

---

## Configuration

- The default filtering model is:  
  nvidia/quality-classifier-deberta
- You can change this by passing `--model <model_name>` to `superfiltering.py`.

---

## Notes
 
- For strict reproducibility with the official [Superfiltering paper](https://arxiv.org/abs/2402.00530).

---

## References

- [Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning](https://arxiv.org/abs/2402.00530)  
- [OptimalScale RegMix Dataset](https://huggingface.co/datasets/OptimalScale/ClimbLab)  
- [NVIDIA NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/)
