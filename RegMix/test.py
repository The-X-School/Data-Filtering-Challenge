#testing for my other stuff
cluster_size = [40864239300, 357491746146, 91383607579, 199987401668, 30974324173, 41388345309, 92352151083, 141420542206, 216228681232, 137535004061, 134908407886, 287503263409, 219970698540, 216478698528, 174421761907, 247156692325, 178517963591, 167948271197, 65865197449, 34707159890]
cluster_order = [1,10,11,12,13,14,15,16,17,18,19,2,20,3,4,5,6,7,8,9]

cluster_sizes = [] 
for i in range(20):
    cluster_sizes.append([cluster_size[i],cluster_order[i]])

print(cluster_sizes)




new_cluster = [[0,0]]*20
print(f"\n new cluster setup {new_cluster}")

for cluster_index in range(20):
    swap_pos = cluster_sizes[cluster_index][1]-1

    #swap positions to make cluster in order
    new_cluster[swap_pos] = cluster_sizes[cluster_index]


print()
print()
print()
print()
print(f"cluster size ordered: {new_cluster}")



