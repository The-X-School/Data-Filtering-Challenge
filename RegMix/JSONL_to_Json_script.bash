mkdir -p /home/ubuntu/Data-Filtering-Challenge/RegMix/Json_Data/
for i in {1..20}; do

    i_path="/home/ubuntu/Data-Filtering-Challenge/RegMix/Json_Data/cluster_$i.JSONL"
    o_path="/home/ubuntu/Data-Filtering-Challenge/RegMix/Json_Data/cluster_$i.json"

    python json_to_jsonl.py $i_path $o_path

done