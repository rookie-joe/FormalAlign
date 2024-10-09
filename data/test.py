import re
import random

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
    print(f'found exponents: {exponents}')
    
    if exponents:
        # Choose a random exponent to modify
        old_expression = random.choice(exponents)
        
        # Extract base and old exponent from the old_expression
        base, old_exponent = re.match(exponent_pattern, old_expression).groups()
        
        # Remove any surrounding spaces from base and exponent
        base = base.strip()
        old_exponent = old_exponent.strip().strip('(').strip(')')
        print(f'old_exponent: {old_exponent}')
        
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
            print(f'new_value: {new_value}')

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
        print(f'new_expression: {new_expression}')
        
        # Replace the first occurrence of the original matched expression with the new expression
        expression = expression.replace(old_expression.strip(), new_expression, 1)
    
    return expression


# Test the function
test_expressions = [
    "x^2 + y^3",
    "2^(-3) * z^4.5",
    "a^1.5 - b^(-2.7)",
    "c^0.01 + d^100",
    "theorem map_pow : ∀ (x) (n : ℕ), v (x ^ n) = v x ^ n :=",
    "theorem map_pow : "
]

for expr in test_expressions:
    modified = modify_exponent(expr)
    print(f"Original: {expr}")
    print(f"Modified: {modified}")
    print()
