from huggingface_hub import hf_hub_download
from huggingface_hub import login
from huggingface_hub import list_repo_files
from huggingface_hub import get_hf_file_metadata
from huggingface_hub import hf_hub_url
import random
import os
import pandas as pd
import json

#install huggingface_hub with conda install huggingface_hub also install pandas and fastparquet
wanted_clusters = [];  #input from which cluster (out of 20)
files_per_cluster  = 1;  #1 is normally fine --> max is 100
rows_per_cluster = 1000;    #1000 is also normally fine
foldername = "data" #creates a folder in current position --> can replace with path to put in other place


#make folder
try:
    os.mkdir(foldername)
    print(f"Folder {foldername} created successfully in the current directory.")
except FileExistsError:
    print(f"Folder {foldername} already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

HF_TOKEN = "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"
login(token=HF_TOKEN)

files = list_repo_files(repo_id="OptimalScale/ClimbLab", repo_type="dataset", token=HF_TOKEN)

cluster_order = [0, 11, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

for i in wanted_clusters:
    cluster_num = i-1
    cluster_pos = cluster_order[cluster_num]
    for i in range(files_per_cluster):
        filename = files[2 + cluster_pos * 100 + i]

        dataset = hf_hub_download(
            repo_id="OptimalScale/ClimbLab",
            filename=filename,
            repo_type="dataset",
            force_download = True
        )
        
        #convert to dataframe
        print(f"\nConverting parquet {filename} into dataframe...")
        dataframe_dataset = pd.read_parquet(dataset, engine = 'fastparquet')
        #print the dataset
        print(f"Parquet Conversion of {filename} complete!")
        print(dataframe_dataset)


        #slicing row
        sliced_rows = dataframe_dataset.iloc[0:rows_per_cluster]


        #convert to json
        json_sliced = sliced_rows.to_json(
        orient="records",
        lines=True
        )

        #write into the folder
        filename = f"cluster_{i+1}_i.json"
        if(rows_per_cluster !=0):
            file_path = os.path.join(foldername, filename)

            with open(file_path, "w") as file:
                file.write(json_sliced)
        else:
            print(f"file 'cluster_{i+1}.json' not created: No Data")

    


