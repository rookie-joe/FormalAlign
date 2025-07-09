#!/bin/bash

# Enable CUDA launch blocking for debugging
export CUDA_LAUNCH_BLOCKING=1

# Move to the theorem proving directory
cd ./theorem_proving

# Define the evaluation configuration parameters
generator_id=mistral
verifierID=mma_forml4_combined_v2
final_id=formalalign_reproduce

# Set the model checkpoint directory (this should be your trained model path)
checkpoint_dir=/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/checkpoints/${generator_id}/${final_id}_${verifierID}

data_id=minimal_test
set_id=minif2f_test
test_data_dir=../data/${data_id}/${set_id}.jsonl

# Set output directory
output_dir=/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/eval_results/${generator_id}_${verifierID}/${data_id}_${set_id}
# create output directory if it doesn't exist
mkdir -p ${output_dir}

# Set log file path
log_dir=/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/logs
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file=${log_dir}/eval_alignment_${final_id}_${timestamp}_${data_id}.log

# Evaluation method configuration
# Set to True to use autoregressive certainty calculation (recommended)
# Set to False to use teacher forcing (old method)

# Run evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 /research/projects/trans_llm/Zeru_Shi/conda/envs/formalalign/bin/accelerate launch \
  --main_process_port=29999 \
  --config_file ../configs/zero1.yaml \
  ../eval_clip+cert.py \
  --model_name_or_path ${checkpoint_dir} \
  --project_dim 4096 \
  --data_dir ${test_data_dir} \
  --data_id ${data_id} \
  --target_set test \
  --generator_id ${generator_id} \
  --verifier_id ${verifierID} \
  --verifier_output_dir ${output_dir} \
  --per_device_eval_batch_size 32 \
  --seed 42 2>&1 | tee "${log_file}" 