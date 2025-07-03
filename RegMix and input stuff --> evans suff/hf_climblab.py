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
'''
#find average size of cluster

#cluster size is a list of lists, each list contains the size of the cluster and the order of the cluster
cluster_sizes = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
print(f"cluster size: {cluster_sizes}")
#cluster order is the order of the clusters in the dataset
cluster_order = [1,10,11,12,13,14,15,16,17,18,19,2,20,3,4,5,6,7,8,9]

for cluster in range(20):
    cluster_sizes[cluster][1] = cluster_order[cluster]
    print(f"cluster {cluster+1}: {cluster_sizes[cluster][1]} cluster size: {cluster_sizes}")
    print(f"cluster size: {cluster_sizes}")

    for file in range(100):

        #get the file url
        file_url = hf_hub_url(
            repo_id="OptimalScale/ClimbLab",
            filename=files[cluster*100+file],
            repo_type="dataset",
        )

        #get the file metadata
        file_metadata = get_hf_file_metadata(
            url= file_url,
            token=HF_TOKEN
        )

        #add the file size to the cluster size
        cluster_sizes[cluster][0] += file_metadata.size

        #print the file size and name
        print(f"File size: {int(file_metadata.size/(1024*1024))} MB")
        print(f"File name: {files[cluster*100+file]}")
    

print(f"cluster size: {cluster_sizes}")

ordered_cluster_sizes = [0]*20

for cluster_index in range(len(cluster_sizes)):
    swap_pos = cluster_sizes[cluster_index][1]-1
    
    ordered_cluster_sizes[swap_pos] = cluster_sizes[cluster_index][0]
    #swap positions to make cluster in order
    

print(ordered_cluster_sizes)
'''

ordered_avg_cluster_file_sizes = [408642393.0, 2875032634.09, 2164786985.28, 1744217619.07, 2471566923.25, 1785179635.91, 1679482711.97, 658651974.49, 347071598.9, 3574917461.46, 913836075.79, 1999874016.68, 309743241.73, 413883453.09, 923521510.83, 1414205422.06, 2162286812.32, 1375350040.61, 1349084078.86, 2199706985.4]

average_file_size = 1538552078.7395003
#constants found in Find\ Average\ File\ size.py

cluster_order = [1,10,11,12,13,14,15,16,17,18,19,2,20,3,4,5,6,7,8,9]

cluster_distribution = [0]*20

total_size = 0

print()

for cluster in range(20):
    #adjusts so that each file from a cluster is ~about the same size
#    standardize = average_file_size/ ordered_avg_cluster_file_sizes[cluster]

    cluster_distribution[cluster] = random.randint(0, 3) #*standardize    #randomly assign distribution size

    # print(f"standardize {cluster+1}: {standardize}")

    # total_size += cluster_distribution[cluster] * ordered_avg_cluster_file_sizes[cluster]   

print(f"cluster_distribution: {cluster_distribution}")


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

    sliced_rows = dataframe_dataset[0:1000*cluster_distribution[current_pos]]
    # print(f"\n\n first 100 rows of df dataset from cluster {1}: {first_100_rows}")
         # Get file size in bytes

    json_sliced = sliced_rows.to_json(
        orient="records",
        lines=True
    )

    
    filename = f"cluster {current_pos}.JSONL"
    file_path = os.path.join(foldername, filename)

    with open(file_path, "w") as file:
        file.write(json_sliced)

    df_dt_totalsize += sliced_rows.size

    #convert to json + turn into another file

    file_size_bytes = os.path.getsize(dataset)
    print(f'''
          \nCluster {i} 
          \nFilename {filename}
          \nFile size: {sliced_rows.size/1024:.2f}Kb 
          \nTotal File size: {df_dt_totalsize/(1024*1024):.2f} Mb 
          \n First {10000*cluster_distribution[current_pos]} rows: {sliced_rows} \n 
        ''')

# print(f"Total File Size: {df_dt_totalsize/(1024**3):.2f} GB")