#made to calculate average size of cluster 
#can be quickly modified for other metadata/size + input stuff
#from optimal scale
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from huggingface_hub import list_repo_files
from huggingface_hub import get_hf_file_metadata
from huggingface_hub import hf_hub_url
import random
import os

HF_TOKEN = "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"
login(token=HF_TOKEN)

#get the list of files in the dataset
files = list_repo_files(repo_id="OptimalScale/ClimbLab", repo_type="dataset", token=HF_TOKEN)
print(files)




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

for i in range(20):
    ordered_cluster_sizes[i] = ordered_cluster_sizes[i]/100

print(f"average size: {ordered_cluster_sizes}")