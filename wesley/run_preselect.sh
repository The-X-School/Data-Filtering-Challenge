pip install packaging
pip install numpy
pip install datasets==2.14.6
pip install tokenizers>=0.13.3
pip install peft==0.15.2
pip install torch>=2.0.1
pip install wandb
pip install deepspeed>=0.14.4
pip install sentencepiece
pip install transformers>=4.31.0
pip install cpm_kernels==1.0.11
pip install evaluate==0.4.0
pip install bitsandbytes>=0.40.0
pip install pydantic
pip install accelerate>=0.27.2
pip install einops>=0.6.1
pip install accelerate ndjson fasttext datatrove orjson datasets fasteners fasttext-numpy2-wheel

python run_preselect_filtering.py \
    --input_path=detokenized/climblab \
    --model_path=model_function_calling_10k_wesley.bin \
    --output_dir=preselect_tokenized_wesley \
    --threshold=0.79