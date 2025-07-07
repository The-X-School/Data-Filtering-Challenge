python hf_climblab.py

mkdir -p /home/ubuntu/Data-Filtering-Challenge/RegMix/Json_Data/
mv json_to_jsonl.py JsonL_Data/

cd JsonL_Data/

for i in {1..20}; do

    i_path="cluster_$i.JSONL"
    o_path="cluster_$i.json"

    python json_to_jsonl.py $i_path $o_path

cd ..

for i in {1..20}; do

    filename="/home/ubuntu/Data-Filtering-Challenge/RegMix/JsonL_Data/cluster_$i.json" 
    mv $filename Json_Data/

rm -rf JsonL_Data/
done