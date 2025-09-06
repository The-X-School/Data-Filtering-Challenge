# Data-Filtering-Challenge

This is X-School's program for their submission to the Data-Filtering-Challenge

The main script for the filter is wesley/run_preselect_filtering.py which calls all of the other scripts used for filtering. 

Instructions:

1. Set up a virtual enviroment and download all the dependencies.
    conda create -n lmflow python=3.10.18 -y
    conda activate lmflow
    conda install mpi4py
    pip install -e .
    pip install -r requirements.txt

2. Run the script to download data from climblab. Due to constraints, we were not able to handle the entire dataset, so we just drew a sample from the climblab dataset. Previously we ran tests with random samples but that would make the result inconsistent, so we settled on pulling just the first million lines, or roughly 886m tokens from the dataset. Effectively, this also means that our program will not go over the 10b token limit. 
    #Note: You might need to insert a working huggingface token into here for it to download. 
    python download_climblab_streaming.py

3. Run the script to detokenize
