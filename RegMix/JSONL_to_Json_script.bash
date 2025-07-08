python hf_climblab.py

mv format_data.py JsonL_Data/

echo "moved format_data.py to JsonL_Data/"

cd JsonL_Data/

for i in {1..20}; do

    i_path="cluster_$i.JSONL"
    echo $i_path
    python format_data.py $i_path
done

mv format_data.py ..
cd ../..

bash train.sh

