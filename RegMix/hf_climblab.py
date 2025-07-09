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

HF_TOKEN = "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"
login(token=HF_TOKEN)

#get the list of files in the dataset
files = list_repo_files(repo_id="OptimalScale/ClimbLab", repo_type="dataset", token=HF_TOKEN)
print(f"file name: {files}")



cluster_order = [1,10,11,12,13,14,15,16,17,18,19,2,20,3,4,5,6,7,8,9]

cluster_distribution = [0]*20

total_size = 0

print()

for cluster in range(20):
    #adjusts so that each file from a cluster is ~about the same size
#    standardize = average_file_size/ ordered_avg_cluster_file_sizes[cluster]

    cluster_distribution[cluster] = random.randint(0, 30) #*standardize    #randomly assign distribution size

    # print(f"standardize {cluster+1}: {standardize}")

    # total_size += cluster_distribution[cluster] * ordered_avg_cluster_file_sizes[cluster]

print(f"cluster_distribution: {cluster_distribution}")

print("creating cluster distribution files....")
outfile = "cluster_dist.txt"
with open(outfile, "w") as f:
    f.write(str(cluster_distribution))
print("file created as 'cluster_dist.txt' \n ")

# print(f"\n Total size: {total_size/(1024*1024*1024)} GB")

df_dt_totalsize = 0

#create folder for data
foldername = "JsonL_Data"
try:
    os.mkdir(foldername)
    print(f"Folder {foldername} created successfully in the current directory.")
except FileExistsError:
    print(f"Folder {foldername} already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

for i in range(20):
    current_pos = cluster_order[i]-1
    filename = files[2+i*100]
    print(f"filename {filename}")

    #download the dataset
    dataset = hf_hub_download(
        repo_id="OptimalScale/ClimbLab",
        filename=filename,
        repo_type="dataset",
        force_download = True
    )
    print(f"\nConverting parquet {filename} into dataframe...")
    dataframe_dataset = pd.read_parquet(dataset, engine = 'fastparquet')
    #print the dataset
    print(f"Parquet Conversion of {filename} complete!")
    print(dataframe_dataset)

    cluster_size = int(1000*cluster_distribution[current_pos]/10)

    sliced_rows = dataframe_dataset.iloc[0:cluster_size]
    # print(f"\n\n first 100 rows of df dataset from cluster {1}: {first_100_rows}")
         # Get file size in bytes
#    if(cluster_distribution[current_pos] == 0):
#        sliced_rows = dataframe_dataset.iloc[0]

    json_sliced = sliced_rows.to_json(
        orient="records",
        lines=True
    )

    cluster_size = cluster_distribution[current_pos]/10
    filename = f"cluster_{current_pos+1}.json"

    if(cluster_size !=0):
        file_path = os.path.join(foldername, filename)

        with open(file_path, "w") as file:
            file.write(json_sliced)
    else:
        print(f"file 'cluster_{current_pos+1}.json' not created: No Data")
        
    
    df_dt_totalsize += sliced_rows.size
    #convert to json + turn into another file

    file_size_bytes = os.path.getsize(dataset)
    print(f'''
          \nIteration {i}
          \nCluster {current_pos+1}
          \nFilename {filename}
          \nFile size: {sliced_rows.size/1024:.2f}Kb
          \nTotal File size: {df_dt_totalsize/(1024*1024):.2f} Mb
          \n First {cluster_size} rows: {sliced_rows} \n
        ''')

# print(f"Total File Size: {df_dt_totalsize/(1024**3):.2f} GB")