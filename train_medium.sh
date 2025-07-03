#!/bin/bash
# Medium Training Script - 1-2 hours, good balance

# Parses arguments
model_name_or_path=data4elm/Llama-400M-12L
dataset_path=data/glaive_filtered 
output_dir=output_models/finetune_medium
deepspeed_args="--master_port=11002"

# Safety related arguments
trust_remote_code=0

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_dora_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Medium Finetune - 1000 steps (~1-2 hours)
exp_id=finetune_medium
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --block_size 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 24 \
    --use_dora 1 \
    --lora_r 16 \
    --lora_target_modules="embed_tokens,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head" \
    --save_aggregated_lora 0 \
    --deepspeed configs/ds_config_zero2.json \
    --bf16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --max_steps 1000 \
    --save_steps 200 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 128 \
    --report_to none \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err 