python hf_climblab.py

mkdir -p /home/ubuntu/Data-Filtering-Challenge/RegMix/Json_Data/
echo "now at" 
pwd


mv format_data.py JsonL_Data/

echo "moved format_data.py to JsonL_Data/"

cd JsonL_Data/

for i in {1..20}; do

    i_path="cluster_$i.JSONL"

    python json_to_jsonl.py $o_path
done

mv format_data.py ..
cd ..

