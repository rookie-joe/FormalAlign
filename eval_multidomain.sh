#!/bin/bash

# Enable CUDA launch blocking for debugging
export CUDA_LAUNCH_BLOCKING=1

# Move to the theorem proving directory
cd ./theorem_proving

# Define the evaluation configuration parameters
generator_id=mistral
verifierID=mma_forml4_minif2f_combined
final_id=formalalign_multidomain

# Set the model checkpoint directory (this should be your trained model path)
checkpoint_dir=/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/checkpoints/${generator_id}/${final_id}_${verifierID}

# Evaluation datasets
datasets=(
    "forml4:clean_set:clean_formatted_random_test_clip"
    "minif2f:clean_set:clean_formatted_random_test_clip"
    "mma:clean_set:clean_formatted_random_test_clip"
)

# Set log file path
log_dir=/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/logs
timestamp=$(date +"%Y%m%d_%H%M%S")

# Run evaluation on each dataset
for dataset_info in "${datasets[@]}"; do
    IFS=':' read -r data_id set_id <<< "$dataset_info"
    
    # test data dir
    test_data_dir=../data/${data_id}/${set_id}.jsonl
    
    # Set output directory
    output_dir=/research/projects/trans_llm/Zeru_Shi/alisa/FormalAlign/eval_results/${generator_id}_${verifierID}/${data_id}_${set_id}
    # create output directory if it doesn't exist
    mkdir -p ${output_dir}
    
    # Set log file path
    log_file=${log_dir}/eval_alignment_${final_id}_${timestamp}_${data_id}.log
    
    echo "Evaluating on ${data_id} dataset..."
    echo "Test data: ${test_data_dir}"
    echo "Output dir: ${output_dir}"
    echo "Log file: ${log_file}"
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=0,1,3 /research/projects/trans_llm/Zeru_Shi/conda/envs/formalalign/bin/accelerate launch \
      --main_process_port=29999 \
      --config_file ../configs/zero1_3gpu.yaml \
      ../eval_alignment.py \
      --model_name_or_path ${checkpoint_dir} \
      --project_dim 4096 \
      --data_dir ${test_data_dir} \
      --data_id ${data_id} \
      --target_set test \
      --generator_id ${generator_id} \
      --verifier_id ${verifierID} \
      --output_dir ${output_dir} \
      --per_device_eval_batch_size 64 \
      --use_autoregressive_certainty True \
      --max_new_tokens 512 \
      --seed 42 2>&1 | tee "${log_file}"
    
    echo "Evaluation completed for ${data_id}"
    echo "----------------------------------------"
done

echo "All evaluations completed!" 