#!/bin/bash

# Enable CUDA launch blocking for debugging
export CUDA_LAUNCH_BLOCKING=1

# Move to the theorem proving directory
cd ./theorem_proving

# Define the training configuration parameters
generator_id=mistral
verifierID=mma_forml4_combined_v2
checkpoint_dir='mistralai/Mistral-7B-v0.1'  # Base model directory

# Set the unique run identifier and output model directory
final_id=formalalign_reproduce
save_dir=/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/checkpoints/${generator_id}/${final_id}_${verifierID}

# Set log file path
log_dir=/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/logs
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file=${log_dir}/training_${final_id}_${timestamp}.log

# Activate multi-GPU training using CUDA and Accelerate
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
CUDA_VISIBLE_DEVICES=0,3 /research/projects/trans_llm/Zeru_Shi/conda/envs/formalalign/bin/accelerate launch \
  --main_process_port=29999 \
  --config_file ../configs/zero2_2gpu.yaml \
  ../train_clip.py \
  --model_name_or_path ${checkpoint_dir} \
  --project_dim 4096 \
  --clip_temperature 0.05 \
  --data_dir ../data/mma_forml4/train.jsonl \
  --data_id mma_forml4_combined \
  --target_set train \
  --val_target_set val \
  --save_dir ${save_dir} \
  --generator_id ${generator_id} \
  --verifier_id ${verifierID} \
  --dedup True \
  --loss_level token \
  --loss_on_llm True \
  --num_train_epoches 1 \
  --eval_steps 10000 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --learning_rate 2e-6 \
  --weight_decay 0 \
  --save_steps 100 \
  --lr_scheduler_type "linear" \
  --warmup_ratio 0.03 \
  --save_epoches 1 \
  --save_best False \
  --save_total_limit 1 \
  --logging_dir None \
  --logging_steps 10 \
  --resume_from_checkpoint False \
  --seed 42 2>&1 | tee ${log_file} 