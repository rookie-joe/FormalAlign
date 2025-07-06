import json
import os

def convert_to_clip_format(input_file, output_file):
    # Read source data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Convert to CLIP format, grouped by question
    clip_data = []
    for idx, item in enumerate(data):
        clip_entry = {
            'idx': idx,  # Question index
            'input': item['input'],  # Question text
            'outputs': []  # List of outputs for this question
        }
        
        # Add each output/solution for this question
        for output in item['outputs']:
            clip_entry['outputs'].append({
                'response': output['response'],  # Solution text
                'label': 1 if output['label'] else 0,  # Validation label
            })
        
        clip_data.append(clip_entry)
    
    # Write to JSONL format
    with open(output_file, 'w') as f:
        for entry in clip_data:
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    input_file = 'minif2f/misalignment/test_misalignment.json'
    output_file = 'minif2f/misalignment/test_clip.jsonl'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert data
    convert_to_clip_format(input_file, output_file)
    print(f"Converted data saved to {output_file}") 