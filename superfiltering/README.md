# Superfiltering for RegMix (20 Clusters)

This repository contains code to apply **Superfiltering** on the [RegMix dataset](https://huggingface.co/datasets/OptimalScale/ClimbLab) clustered into 20 subsets.
The pipeline first slices clusters according to a predicted distribution, then applies **weak-to-strong data filtering** to select high-quality samples.

---

## Repo Structure

superfiltering-regmix/  
├── superfiltering.py        # Main filtering script  
├── regmix_preprocess.py     # (Optional) Script to slice raw RegMix clusters  
├── requirements.txt         # Dependencies  
├── README.md                # Documentation  
├── data/                    # Place your RegMix cluster files here (not tracked by git)  

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

1. Download the [RegMix dataset (20 clusters)](https://huggingface.co/datasets/OptimalScale/ClimbLab/tree/main).  
2. Run the preprocessing script to slice each cluster based on the predicted distribution:

python regmix_preprocess.py

This will generate JSON cluster files in a `new_data/` folder:

new_data/  
 ├── cluster_1.json  
 ├── cluster_2.json  
 ...  
 └── cluster_20.json  

---

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

- The dataset itself is **not included** in this repo (due to size). You’ll need to download it from Hugging Face.  
- Filtering can take time depending on dataset size and model choice. Consider batching or using a GPU environment for speed.  
- For strict reproducibility with the official [Superfiltering paper](https://arxiv.org/abs/2402.00530), you may need to adapt your dataset format to match their repo’s expected schema.  

---

## References

- [Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning](https://arxiv.org/abs/2402.00530)  
- [OptimalScale RegMix Dataset](https://huggingface.co/datasets/OptimalScale/ClimbLab)  
- [NVIDIA NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/)
