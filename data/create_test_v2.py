import json
import os

def convert_to_clip_format(input_file, output_file):
    # Read source data
    with open(input_file, 'r') as f:
        data = json.load(f)
    

    
    # Write to JSONL format
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    input_file = 'minif2f/misalignment/valid_misalignment.json'
    # output_file = 'minif2f/misalignment/test_clip.jsonl'
    # input_file = 'forml4/misalignment/formatted_random_test.json'
    output_file = input_file.replace('.json', '.jsonl')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert data
    convert_to_clip_format(input_file, output_file)
    print(f"Converted data saved to {output_file}") 