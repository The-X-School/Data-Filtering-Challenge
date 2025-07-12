#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Parses arguments
model_name_or_path="data4elm/Llama-400M-12L"
trust_remote_code=0
dataset_path="analysis/10k_Preselect"
output_dir="output_models/wesley_preselect8709"
deepspeed_args="--master_port=11000"

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

# Finetune
# Note that project dir will contain files that show you loss and other metrics during finetuning.
exp_id=finetune_with_dora
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed --master_port 11000 --include localhost:0\
    examples/finetune.py \
    --model_name_or_path "data4elm/Llama-400M-12L" \
    --trust_remote_code 0 \
    --dataset_path "analysis/10k_Preselect" \
    --output_dir "output_models/wesley_preselect8709" \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
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
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 128 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
