#designed to import from https://huggingface.co/datasets/OptimalScale/ClimbLab/tree/main
#regmix but im lowk working on inputting now
#need to install fastparquet and pandas beforehand jsyk
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from huggingface_hub import list_repo_files
from huggingface_hub import get_hf_file_metadata
from huggingface_hub import hf_hub_url
import random
import os
import pandas as pd
import json

#get the list of files in the dataset
files = ["cluster1.jsonl", "cluster2.jsonl", "cluster3.jsonl", "cluster4.jsonl", "cluster5.jsonl", "cluster6.jsonl", "cluster7.jsonl", "cluster8.jsonl", "cluster9.jsonl", "cluster10.jsonl", "cluster11.jsonl", "cluster12.jsonl", "cluster13.jsonl", "cluster14.jsonl", "cluster15.jsonl", "cluster16.jsonl", "cluster17.jsonl", "cluster18.jsonl", "cluster19.jsonl", "cluster20.jsonl"]
print(f"file names: {files}")

predicted_dist = [0.587189, 1.000000, 0.082324, 0.577493, 0.574668, 0.580747, 0.085329, 1.000000, 0.574213, 0.578897, 0.600457, 0.587480, 0.576953, 0.581199, 0.590750, 0.580672, 0.575514, 0.585379, 0.585687, 0.579300]
print()
print(f"cluster_distribution: {predicted_dist}")

print("creating cluster distribution files....")
outfile = "cluster_dist.txt"
with open(outfile, "w") as f:
    f.write(str(predicted_dist))
print("file created as 'cluster_dist.txt' \n ")


df_dt_totalsize = 0

#create folder for data
foldername = "new_data"
try:
    os.mkdir(foldername)
    print(f"Folder {foldername} created successfully in the current directory.")
except FileExistsError:
    print(f"Folder {foldername} already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

current_dir = os.getcwd()
print(f"current directory: {current_dir}")

for i in range(20):
    foldername = "JsonL_Data"
    filename = files[i]
    print(f"filename {filename}")

    

    #getting json file
    file_path = os.path.join(foldername, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
            print(f"File content: {data}")
    else:
        print(f"File {file_path} does not exist")
        continue


    print(f"\nConverting jsonl {filename} into dataframe...")
    dataframe_dataset = pd.read_json(file_path, lines=True)
    #print the dataset
    print(f"JsonL Conversion of {filename} complete!")
    print(dataframe_dataset)

    cluster_size = int(predicted_dist[i])
    sliced_rows = dataframe_dataset.iloc[0:cluster_size]




    json_sliced = sliced_rows.to_json(
        orient="records",
        lines=True
    )

    filename = f"cluster_{i+1}.json"

    if(cluster_size !=0):
        file_path = os.path.join(foldername, filename)

        with open(file_path, "w") as file:
            file.write(json_sliced)
    else:
        print(f"file 'cluster_{i+1}.json' not created: No Data")
        
    
    df_dt_totalsize += sliced_rows.size
    #convert to json + turn into another file

    print(f'''
          \nIteration {i}
          \nCluster {i+1}
          \nFilename {filename}
          \nFile size: {sliced_rows.size/1024:.2f}Kb
          \nTotal File size: {df_dt_totalsize/(1024*1024):.2f} Mb
          \n First {cluster_size} rows: {sliced_rows} \n
        ''')

# print(f"Total File Size: {df_dt_totalsize/(1024**3):.2f} GB")
