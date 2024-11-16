import json

# Load the .jsonl file
file_path = 'gpt4o_scoring.jsonl'
best_threshold = 0.3
best_correctness_ratio = 0

def extract_alignment_score(model_response):
    # Find the alignment score line and extract the score after '# Alignment Score:'
    score_prefix = '# Alignment Score: '
    if score_prefix in model_response:
        try:
            score_str = model_response.split(score_prefix)[-1].strip()
            return float(score_str)
        except ValueError:
            return None
    return None

# Iterate over all possible thresholds with interval of 0.01
for threshold in range(30, 101):  # 0.30 to 1.00 with interval of 0.01
    threshold /= 100  # Convert to float
    correct_matches = 0
    total_samples = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            total_samples += 1
            model_response = sample['model_response']
            label = sample['label']
            
            # Extract alignment score from model_response
            score = extract_alignment_score(model_response)
            
            # Determine the model label based on alignment score
            if score is not None:
                model_label = score >= threshold * 5
                label = str(label).lower() == 'true'  # Convert label to boolean
                
                if model_label == label:
                    correct_matches += 1
    
    # Calculate correctness ratio for the current threshold
    if total_samples > 0:
        correctness_ratio = (correct_matches / total_samples) * 100
        if correctness_ratio > best_correctness_ratio:
            best_correctness_ratio = correctness_ratio
            best_threshold = threshold

# Output the best threshold and its correctness ratio
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Best Correctness Ratio: {best_correctness_ratio:.2f}%")