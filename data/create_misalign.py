'''
The script applies 6 modification strategies to the responses in the dataset, 20 modified versions of each original response. 
Each modified response is labeled as false and includes a misalign_type field indicating the strategy used for modification. 
Finally, it prints the average coverage rate for each modification strategy after processing the entire dataset.

The 6 modification strategies are:
    Constant Modification: Increments a randomly chosen constant in the expression.
    Exponent Modification: Increments a randomly chosen exponent in the expression.
    Variable Introduction: Introduces a new random variable in the declaration of variables.
    Variable Type Change: Changes the type of one variable in the declaration to a randomly chosen type.
    Equality Modification: Swaps all instances of \u2260 (not equal) with = (equal) and vice versa.
    Unpaired Modification: Replaces the current response with a randomly chosen response from all other responses in the input file.

Example command:
python create_misalign.py --input_file minif2f/alignment/test_lean4.json --output_path minif2f/misalignment --seed 42
python create_misalign.py --input_file forml4/alignment/formatted_random_test.json --output_path forml4/misalignment --seed 42
'''

import json
import re
import random
import argparse
import os
import string

def modify_constant(expression):
    """
    Modify a constant in the given mathematical expression.
    """
    constants = re.findall(r'(?<!\\u)\b\d+\b', expression)
    if constants:
        chosen_constant = random.choice(constants)
        new_constant = str(int(chosen_constant) + random.randint(1, 100))
        expression = re.sub(r'(?<!\\u)\b' + re.escape(chosen_constant) + r'\b', new_constant, expression, 1)
    return expression

def modify_exponent(expression):
    """
    Modify an exponent in the given mathematical expression.
    Handles integer and decimal exponents, both positive and negative.
    """
    # This pattern captures both the base and exponent, along with optional spaces
    exponent_pattern = r'(\s*-?\s*\w+\s*|\s*\(-?\s*\d+\s*\))\s*\^\s*(\s*-?\s*\d+\.?\d*\s*|\s*\w+\s*|\s*\(.+?\)\s*)'
    
    # Find all base-exponent pairs in the expression
    matches = re.finditer(exponent_pattern, expression)
    exponents = [match.group(0) for match in matches]  # Capture the entire matched expression
    # print(f'found exponents: {exponents}')
    
    if exponents:
        # Choose a random exponent to modify
        old_expression = random.choice(exponents)
        
        # Extract base and old exponent from the old_expression
        base, old_exponent = re.match(exponent_pattern, old_expression).groups()
        
        # Remove any surrounding spaces from base and exponent
        base = base.strip()
        old_exponent = old_exponent.strip().strip('(').strip(')')
        # print(f'old_exponent: {old_exponent}')
        
        # Validate that the chosen exponent is a number
        try:
            # Convert the exponent to a float if it's a valid number
            original_value = float(old_exponent)
            modification = random.randint(-5, 5)

            # Apply the modification, ensuring the result isn't zero
            new_value = original_value + modification
            while new_value == 0 or new_value == original_value:
                modification = random.randint(-5, 5)
                new_value = original_value + modification
            # print(f'new_value: {new_value}')

            # post-process the new exponent before converting to str: add () if negative; convert to int if applicable.
            if new_value % 1 == 0:  # Check if new_value has no decimal part
                new_value = int(new_value)
            if new_value < 0:
                new_exponent = f'({new_value})'
            else:
                new_exponent = str(new_value)

        except ValueError:
            # Handle the case where old_exponent is not a valid number
            new_exponent = '(' + old_exponent + random.choice(['+', '-']) + str(random.randint(1, 5)) + ')'
        
        # Construct the new exponent expression
        new_expression = f'{base} ^ {new_exponent}'.strip()
        # print(f'new_expression: {new_expression}')
        
        # Replace the first occurrence of the original matched expression with the new expression
        expression = expression.replace(old_expression.strip(), new_expression, 1)
    
    return expression

def introduce_variable(expression):
    """
    Introduce a new random variable in the declaration of variables.
    """
    # Find all variable declarations in the form of (variables : ℝ)
    variable_declarations = re.findall(r'([^:]+ : [^\s]+)', expression)
    if variable_declarations:
        # Choose a random variable declaration to modify
        chosen_declaration = random.choice(variable_declarations)
        # Extract the variable type, e.g., ℝ
        variable_type_match = re.search(r'[^\s]+$', chosen_declaration)
        if variable_type_match:
            variable_type = variable_type_match.group(0)
            # Extract existing variable names
            existing_variables = re.findall(r'\b[a-zA-Z]\w*\b', chosen_declaration)
            # Find a new variable name not in existing variables
            new_variable = None
            while new_variable is None or new_variable in existing_variables:
                new_variable = random.choice(string.ascii_lowercase)
            # Introduce the new variable
            new_declaration = f'({new_variable} : {variable_type})'
            expression = expression.replace(chosen_declaration, f'{chosen_declaration}\n{new_declaration}')
    return expression

def change_variable_type(expression):
    """
    Change the type of one variable in the declaration to a randomly chosen type.
    """
    available_types = ['ℕ', 'ℤ', 'ℚ', 'ℝ', '𝔹', '𝕊', '𝕋', 'α', '×', 'β', 'ℒ', '𝕎']
    
    # Find all variable declarations in the form of (variables : type)
    variable_declarations = re.findall(r'\(([^:]+ : [^\s]+)\)', expression)
    
    if variable_declarations:
        # Choose one of the variable declarations to modify
        chosen_declaration = random.choice(variable_declarations)
        current_type_match = re.search(r': ([^\s]+)', chosen_declaration)
        if current_type_match:
            current_type = current_type_match.group(1)
            new_variable_type = current_type
            # Randomly pick a new type for the variables in the declaration
            while new_variable_type == current_type:
                new_variable_type = random.choice(available_types)
            # Modify the type of all variables in this declaration (while other declarations are unchanged)
            modified_declaration = chosen_declaration.replace(current_type, new_variable_type)
            expression = expression.replace(chosen_declaration, modified_declaration)
    return expression


def modify_equality(expression):
    """
    Change all instances of ≠ to = in the given expression, and vice versa.
    """
    if '≠' in expression:
        expression = expression.replace('≠', '=')
    elif '=' in expression:
        expression = expression.replace('=', '≠')
    return expression

def modify_unpaired(responses, current_response):
    """
    Extract a random response from all the other responses in the input file.
    """
    other_responses = [resp for resp in responses if resp != current_response]
    if other_responses:
        return random.choice(other_responses)
    return current_response

def modify_response(response, modification_type, all_responses):
    if modification_type == 'constant':
        return modify_constant(response)
    elif modification_type == 'exponent':
        return modify_exponent(response)
    elif modification_type == 'variable_new':
        return introduce_variable(response)
    elif modification_type == 'variable_type':
        return change_variable_type(response)
    elif modification_type == 'equality':
        return modify_equality(response)
    elif modification_type == 'unpaired':
        return modify_unpaired(all_responses, response)
    else:
        raise ValueError("Invalid modification type.")

def modify_dataset(data, seed):
    random.seed(seed)
    modified_data = []
    modification_types = ['constant', 'exponent', 'variable_new', 'variable_type', 'equality', 'unpaired']
    strategy_counts = {strategy: 0 for strategy in modification_types}
    total_samples = len(data)

    # Collect all responses for the 'unpaired' strategy
    all_responses = [output["response"] for sample in data for output in sample["outputs"]]

    for sample in data:
        modified_responses = []
        modified_sample = sample.copy()
        original_response = modified_sample["outputs"][0]["response"]

        # Append the original response with label true
        modified_sample["outputs"][0]["label"] = True

        # Apply 20 random modifications
        for _ in range(20):
            chosen_modification_type = random.choice(modification_types)
            new_response = modify_response(original_response, chosen_modification_type, all_responses)
            
            # Ensure the modification is valid
            while new_response == original_response or new_response in modified_responses:
                chosen_modification_type = random.choice(modification_types)
                new_response = modify_response(original_response, chosen_modification_type, all_responses)

            modified_sample["outputs"].append({
                "response": new_response,
                "label": False,
                "misalign_type": chosen_modification_type
            })
            modified_responses.append(new_response)
            strategy_counts[chosen_modification_type] += 1

        modified_data.append(modified_sample)

    # Calculate average coverage rate
    avg_coverage_rate = {strategy: count / (20 * total_samples) for strategy, count in strategy_counts.items()}
    print("Average coverage rate per strategy rule:")
    for strategy, rate in avg_coverage_rate.items():
        print(f"{strategy}: {rate:.4f}")

    return modified_data

def main(input_file, output_path, seed):
    random.seed(seed)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Determine output file name
    output_file = os.path.join(output_path, os.path.basename(input_file))

    # Open input data and modify
    with open(input_file, 'r',encoding='utf-8') as f:
        data = json.load(f)

    modified_data = modify_dataset(data, seed)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)

    print(f"Dataset modified and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify constants, exponents, or variables in the dataset.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for replicability.')

    args = parser.parse_args()

    main(args.input_file, args.output_path, args.seed)