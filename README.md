# Data-Filtering-Challenge

This is X-School's program for their submission to the Data-Filtering-Challenge

The main script for the filter is `wesley/run_preselect_filtering.py` which calls all of the other scripts used for filtering. 

Instructions:

1. Set up a virtual enviroment and download all the dependencies. Make sure that conda is installed already.
    ```
    git clone https://github.com/The-X-School/Data-Filtering-Challenge
    cd Data-Filtering-Challenge
    git checkout wesley-final
    
    conda create -n lmflow python=3.10.18 -y
    conda activate lmflow
    conda install mpi4py
    pip install -e .
    pip install -r requirements.txt
    sudo apt-get install git-lfs
    git lfs pull
    ```
    Might need to also install these:
    `pip install datasets transformers datatrove orjson fasteners fasttext-numpy2-wheel`

2. Run the script to download data from climblab. Due to constraints, we were not able to handle the entire dataset, so we just drew a sample from the climblab dataset. Previously we ran tests with random samples but that would make the result inconsistent, so we settled on pulling just the first million lines, or 885,931,732 tokens from the dataset. Effectively, this also means that our program will not go over the 10b token limit. 
    Note: You might need to insert a working huggingface token into here for it to download.
    ```
    cd wesley
    python download_climblab_streaming.py
    ```

3. Run the script to detokenize. The detokenized dataset should be saved to `detokenized/climblab/climblab.jsonl`.
    ```
    python detokenize_climblab.py
    ```
    

4. Run the trained preselect model on the data. Make sure you are in `/Data-Filtering-Challenge` before running this.
    ```
    python run_preselect_filtering.py \
    --input_path=detokenized/climblab/climblab.jsonl \
    --model_path=model_function_calling_10k_wesley.bin \
    --output_dir=preselect_detokenized \
    --threshold=0.79
    ```

    The resulting data should be stored in `wesley/preselect_detokenized`, in a few jsonl files. It should already be in the right format to use to train the model. Our version of the train.sh script is located in the Data-Filtering-Challenge repository if needed to run training/evaluation with the data. 
