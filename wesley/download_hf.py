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
wanted_clusters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];  #input from which cluster (out of 20)
files_per_cluster  = 1;  #1 is normally fine --> max is 100
rows_per_file = -1;    #1000 is also normally fine ---> set to -1 if you want to have full file
foldername = "data" #creates a folder in current position --> can replace with path to put in other place


#make folder
try:
    os.mkdir(foldername)
    print(f"Folder {foldername} created successfully in the current directory.")
except FileExistsError:
    print(f"Folder {foldername} already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

token = os.getenv("HF_TOKEN")
login(token=token)

files = list_repo_files(repo_id="nvidia/ClimbLab", repo_type="dataset", token=token)

cluster_order = [0, 11, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

for j in wanted_clusters:
    cluster_num = j-1
    cluster_pos = cluster_order[cluster_num]
    for i in range(files_per_cluster):
        filename = files[2 + cluster_pos * 100 + i]

        dataset = hf_hub_download(
            repo_id="nvidia/ClimbLab",
            filename=filename,
            repo_type="dataset",
            force_download = True
        )

        #convert to dataframe
        print(f"\nConverting parquet {filename} into dataframe...")
        if dataset.endswith('.parquet'):
           dataframe_dataset = pd.read_parquet(dataset, engine='pyarrow')
        else:
           print(f"Skipping non-parquet file: {dataset}")
           continue;
        #print the dataset
        print(f"Parquet Conversion of {filename} complete!")
        print(dataframe_dataset)


        #slicing row
        sliced_rows = []
        if(rows_per_file == -1):
            sliced_rows = dataframe_dataset.iloc[0:]
            print(f"all rows used within file, file size: {len(dataframe_dataset)}")
        else:
            sliced_rows = dataframe_dataset.iloc[0:rows_per_file]
            print(f"First {rows_per_file}: {sliced_rows}")

        #convert to json
        json_sliced = sliced_rows.to_json(
        orient="records",
        lines=True
        )

        #write into the folder
        filename = f"cluster_{j}_{i}.json"
        if(rows_per_file !=0):
           file_path = os.path.join(foldername, filename)

            with open(file_path, "w") as file:
                file.write(json_sliced)
            print(f"file '{filename}' created!")
        else:
            print(f"file '{filename}' not created: No Data")
