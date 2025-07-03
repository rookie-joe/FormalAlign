import re


def evaluate_expression(strings):
    correct_count = 0

    for string in strings:
        # Find the expression inside <<>>
        match = re.search(r'<<(.+?)>>', string)
        if match:
            expression = match.group(1)
            # Separate the expressions before and after the '='
            before_equal, after_equal = expression.split('=')

            try:
                # Evaluate the expression before the '='
                computed_value = eval(before_equal.strip())
                # Convert the after_equal to an integer for comparison
                actual_value = float(after_equal.strip())

                # Compare the computed value with the actual value
                if computed_value == actual_value:
                    correct_count += 1
            except Exception as e:
                print(f"Error evaluating expression: {expression}. Error: {e}")

    # Calculate accuracy
    accuracy = correct_count / len(strings) if strings else 0
    return accuracy


if '__main__' == __name__:
    # Example usage
    strings = [
        "In the fourth game, Clayton scored the average of his points from the first three games. This is 24+14+10 = <<24+14+10=48>>",
        "Another example where 2*5 = <<2*5=10>>"
    ]

    accuracy = evaluate_expression(strings)
    print(f"Accuracy: {accuracy}")
