#!/bin/bash

# Enable CUDA launch blocking for debugging
export CUDA_LAUNCH_BLOCKING=1

# Move to the theorem proving directory
cd ./theorem_proving

# Define the training configuration parameters
n_solution=10
generator_id=mistral
verifierID=mma_lean
checkpoint_dir='mistralai/Mistral-7B-v0.1'  # Base model directory

# Set the unique run identifier and output model directory
final_id=formalalign_reproduce
save_dir=/opt/tiger/models/${final_id}

# Activate multi-GPU training using CUDA and Accelerate
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
CUDA_VISIBLE_DEVICES=0,1 /research/projects/trans_llm/Zeru_Shi/conda/envs/formalalign/bin/accelerate launch \
  --main_process_port=20114 \
  --config_file ../configs/zero1.yaml \
  ../train_clip.py \
  --model_name_or_path ${checkpoint_dir} \
  --project_dim 4096 \
  --data_dir ../data/mma/lean_train.jsonl \
  --data_id lean4test \
  --target_set train \
  --save_dir ${save_dir} \
  --generator_id ${generator_id} \
  --verifier_id ${verifierID} \
  --dedup True \
  --per_problem_sampling_solution ${n_solution} \
  --loss_level token \
  --loss_on_llm True \
  --num_train_epoches 1 \
  --eval_steps 10000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --learning_rate 2e-6 \
  --weight_decay 0 \
  --save_steps 1000 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_epoches 1 \
  --save_best False \
  --save_total_limit 0 \
  --logging_dir None \
  --logging_steps 1 \
  --seed 42  