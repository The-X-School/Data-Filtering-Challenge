# Data-Filtering-Challenge

This is X-School's program for their submission to the Data-Filtering-Challenge

The main script for the filter is wesley/run_preselect_filtering.py which calls all of the other scripts used for filtering. 

Instructions:

1. Set up a virtual enviroment and download all the dependencies.
    cd Data-Filtering-Challenge
    #Enter the repository if you havent already.
    conda create -n lmflow python=3.10.18 -y
    conda activate lmflow
    conda install mpi4py
    pip install -e .
    pip install -r requirements.txt

2. Run the script to download data from climblab. Due to constraints, we were not able to handle the entire dataset, so we just drew a sample from the climblab dataset. Previously we ran tests with random samples but that would make the result inconsistent, so we settled on pulling just the first million lines, or roughly 886m tokens from the dataset. Effectively, this also means that our program will not go over the 10b token limit. 
    #Note: You might need to insert a working huggingface token into here for it to download. 
    cd wesley
    python download_climblab_streaming.py

3. Run the script to detokenize
    python detokenize_climblab.py
    #The detokenized dataset should be saved to detokenized/climblab/climblab.jsonl

4. Run the trained preselect model on the data. A script exists to run it already.
    #I'm pretty sure you have to exit out of the /wesley folder to run this, so make sure you are in /Data-Filtering-Challenge before running this.

    python wesley/run_preselect_filtering.py \
    --input_path=wesley/detokenized/climblab \
    --model_path=wesley/model_function_calling_10k_wesley.bin \
    --output_dir=wesley/preselect_tokenized_wesley \
    --threshold=0.79

    #The resulting data should be stored in wesley/preselect_tokenized_wesley, in a few jsonl files. The resulting data should already be in the right format to use to train the model. Our version of the train.sh script is located in the Data-Filtering-Challenge repository if needed to run training/evaluation with the data. 
