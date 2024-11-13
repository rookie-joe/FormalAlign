'''
Pre-requisite: get FORML4 dataset from https://github.com/rookie-joe/PDA/tree/main/data/FormL4
'''

import json
import os


def format_forml4(data):
    autoform_temp = '''Statement in natural language:
{question}
Translate the statement in natural language to Lean:'''
    
    formatted_data = []
    for item in data:
        formatted_data.append(
            {
            'input': autoform_temp.format(question=item['nl_problem']),
            'outputs': [
                {
                    'response': item['formal'].split(":=")[0] + ":=",
                    'label': True,
                }
            ]
            }
        )
    return formatted_data

def save_data(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def open_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

file_list = ['basic_test.json', 'random_test.json']
for data_file in file_list:
    data = open_data(data_file)
    output_file=f'formatted_{data_file}'
    save_data(format_forml4(data), output_file)


