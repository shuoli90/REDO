import os
os.environ["ANTHROPIC_API_KEY"] = YOUR_ANTRHOPIC_API_KEY

from typing import DefaultDict
from utils import (
    find_intervals, 
    identify_code_location, 
    prompt_anthropic,
    extract_tag_list,
)
import shutil
import time
from pathlib import Path

import argparse
from tqdm import tqdm
import json
from core.data import descriptions
from core.data import tokenization
import numpy as np
import base64
import re
import subprocess, signal
from tqdm import tqdm

OPUS = "claude-3-opus-20240229"
OPUS_BR = "anthropic.claude-3-opus-20240229-v1:0"
HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
SONNET_3_5 = "claude-3-5-sonnet-20240620"

SYSTEM_TEMPLATE = """
You are an experienced program analyzer who can identify potential runtime errors without running the programs.
"""

# TEMPLATE_att1 = """
# Your task is to predict the first runtime error that might crash the python program. \
# You will firstly be provided with the implementation. \
# Also, a description of the input will be provided. This description describes how the python program will be tested. \
# You can assume that all input are valid, exlcuding corner cases like invalid string, empty string or integers not within the specified range.

# Here is the implementation:
# <Implementation>
# {implementation}
# </Implementation>

# Here is the description of the input:
# <Input description>
# {input}
# </Input description>

# There might be no error or several potential runtime errors in the implementation. \
# If there are potential runtime errors, please predict the first one that might crash the program. \
# This error probably will be triggerred first. 

# Here is the list of potential runtime errors:
# <Error list>
# 1: 'No Error', \
# 2: 'Other', \
# 3: 'Timeout', \
# 4: 'AssertionError', \
# 5: 'AttributeError', \
# 6: 'decimal', \
# 7: 'EOFError', \
# 8: 'FileNotFoundError', \
# 9: 'ImportError', \
# 10: 'IndentationError', \
# 11: 'IndexError', \
# 12: 'KeyError', \
# 13: 'MathDomainError', \
# 14: 'MemoryError', \
# 15: 'ModuleNotFoundError', \
# 16: 'NameError', \
# 17: 'OSError', \
# 18: 'OverflowError', \
# 19: 're.error', \
# 20: 'RecursionError', \
# 21: 'RuntimeError', \
# 22: 'StopIteration', \
# 23: 'SyntaxError', \
# 24: 'TabError', \
# 25: 'TypeError', \
# 26: 'UnboundLocalError', \
# 27: 'ValueError', \
# 28: 'ZeroDivisionError', \
# 29: 'numpy.AxisError' \
# </Error list>

# Please first decribe the number of input from the input description and the number of assumed input in the implementation; and check if the inputs are parsed in the correct way, e.g., \
# whether lines are split by "/n", instead of " ". \
# Then, list all runtime errors by their presence order (from early to late) in the program in the "Potential errors" section, being wrapped by <Potential errors></Potential errors>; \
# next, output your reasoning process in the "Reasoning" section, being wrapped by <Reasoning></Reasoning>; \
# finally, output the index of the first runtime error in the "Conclusion" section, being wrapped by <Conclusion></Conclusion>. \
# """

# TEMPLATE_att2 = """
# Given the description on input and a implemented script, please prediction what kinds of runtime errors the implementation would encounter:

# Here is the implementation:
# <Implementation>
# {implementation}
# </Implementation>

# Here is the description of the input:
# <Input description>
# {input}
# </Input description>

# Please predict whether given the input, the first runtime error that might crash the program. Potential runtime errors are:
# <Error list>
# 1: 'No Error', \
# 2: 'Other', \
# 3: 'Timeout', \
# 4: 'AssertionError', \
# 5: 'AttributeError', \
# 6: 'decimal', \
# 7: 'EOFError', \
# 8: 'FileNotFoundError', \
# 9: 'ImportError', \
# 10: 'IndentationError', \
# 11: 'IndexError', \
# 12: 'KeyError', \
# 13: 'MathDomainError', \
# 14: 'MemoryError', \
# 15: 'ModuleNotFoundError', \
# 16: 'NameError', \
# 17: 'OSError', \
# 18: 'OverflowError', \
# 19: 're.error', \
# 20: 'RecursionError', \
# 21: 'RuntimeError', \
# 22: 'StopIteration', \
# 23: 'SyntaxError', \
# 24: 'TabError', \
# 25: 'TypeError', \
# 26: 'UnboundLocalError', \
# 27: 'ValueError', \
# 28: 'ZeroDivisionError', \
# 29: 'numpy.AxisError' \
# </Error list>

# Please output the index of your predicted error type in the "Conclusion" section, being wrapped by <Conclusion></Conclusion>; \
# and your reasoning in the "Reasoning" section, being wrapped by <Reasoning></Reasoning>. 
# """

TEMPLATE = """
Given the description of input and the implemented script, please check if the implementation contains runtime errors. \
You can assume that the inputs are always valid, and relect the common case.

Here is the implementation:
<Implementation>
{implementation}
</Implementation>

Here is the description of the input:
<Input description>
{input}
</Input description>

Potential runtime errors are:
<Error list>
1: 'No Error', \
2: 'Other', \
3: 'Timeout', \
4: 'AssertionError', \
5: 'AttributeError', \
6: 'decimal', \
7: 'EOFError', \
8: 'FileNotFoundError', \
9: 'ImportError', \
10: 'IndentationError', \
11: 'IndexError', \
12: 'KeyError', \
13: 'MathDomainError', \
14: 'MemoryError', \
15: 'ModuleNotFoundError', \
16: 'NameError', \
17: 'OSError', \
18: 'OverflowError', \
19: 're.error', \
20: 'RecursionError', \
21: 'RuntimeError', \
22: 'StopIteration', \
23: 'SyntaxError', \
24: 'TabError', \
25: 'TypeError', \
26: 'UnboundLocalError', \
27: 'ValueError', \
28: 'ZeroDivisionError', \
29: 'numpy.AxisError' \
</Error list>

Please explain the logic of the implementation in the "Implementation" section, especially how empty strings or lists are handled. \
if the implementation is mostly correct and should run without errors in most cases, please claim "No Error"; \
finally, the index of the identified runtime error that crashes the program in the "Conclusion" section, being wrapped by <Conclusion></Conclusion>. \
"""

error_types = {
    1: 'No Error',
    2: 'Other',
    3: 'Timeout',
    4: 'AssertionError',
    5: 'AttributeError',
    6: 'decimal',
    7: 'EOFError',
    8: 'FileNotFoundError',
    9: 'ImportError',
    10: 'IndentationError',
    11: 'IndexError',
    12: 'KeyError',
    13: 'MathDomainError',
    14: 'MemoryError',
    15: 'ModuleNotFoundError',
    16: 'NameError',
    17: 'OSError',
    18: 'OverflowError',
    19: 're.error',
    20: 'RecursionError',
    21: 'RuntimeError',
    22: 'StopIteration',
    23: 'SyntaxError',
    24: 'TabError',
    25: 'TypeError',
    26: 'UnboundLocalError',
    27: 'ValueError',
    28: 'ZeroDivisionError',
    29: 'numpy.AxisError'
}

error_to_index = {
    'No Error': 1,
    'Other': 2,
    'Timeout': 3,
    'AssertionError': 4,
    'AttributeError': 5,
    'decimal': 6,
    'EOFError': 7,
    'FileNotFoundError': 8,
    'ImportError': 9,
    'IndentationError': 10,
    'IndexError': 11,
    'KeyError': 12,
    'MathDomainError': 13,
    'MemoryError': 14,
    'ModuleNotFoundError': 15,
    'NameError': 16,
    'OSError': 17,
    'OverflowError': 18,
    're.error': 19,
    'RecursionError': 20,
    'RuntimeError': 21,
    'StopIteration': 22,
    'SyntaxError': 23,
    'TabError': 24,
    'TypeError': 25,
    'UnboundLocalError': 26,
    'ValueError': 27,
    'ZeroDivisionError': 28,
    'numpy.AxisError': 29
}

error_list = [
    'AssertionError',
    'AttributeError',
    'decimal',
    'EOFError',
    'FileNotFoundError',
    'ImportError',
    'IndentationError',
    'IndexError',
    'KeyError',
    'MathDomainError',
    'MemoryError',
    'ModuleNotFoundError',
    'NameError',
    'OSError',
    'OverflowError',
    're.error',
    'RecursionError',
    'RuntimeError',
    'StopIteration',
    'SyntaxError',
    'TabError',
    'TypeError',
    'UnboundLocalError',
    'ValueError',
    'ZeroDivisionError',
    'numpy.AxisError'
]

def check_error_type(error_list, error_message):
    if error_message == '':
        return "No Error"
    else:
        for error in error_list:
            if error in error_message:
                return error

def identify_pyright_error(error_message: str) -> dict:
    error_types = {
        'TypeError': [
            'cannot be assigned', 'incompatible type', 'type mismatch',
            'Argument of type', 'cannot be applied', 'TypeVar', 'types', 'Object of type', 'method not defined on type',
            'not iterable'
        ],
        'SyntaxError': [
            'invalid syntax', 'unexpected token', 'SyntaxError'
        ],
        'ImportError': [
            'cannot import', 'No module named', 'ImportError', 'cannot be resolved', 'unknown import symbol'
        ],
        'NameError': [
            'is not defined', 'unresolved reference', 'NameError'
        ],
        'AttributeError': [
            'Cannot access member', 'has no attribute', 'AttributeError', 'Cannot access attribute', 'not a known attribute of module',
        ],
        'IndexError': [
            'list index out of range', 'tuple index out of range', 'IndexError'
        ],
        'KeyError': [
            'key not found', 'KeyError'
        ],
        'ValueError': [
            'invalid value', 'ValueError', 'cannot be interpreted'
        ],
        'IndentationError': [
            'unexpected indent', 'IndentationError'
        ],
        'AssertionError': [
            'assertion failed', 'AssertionError'
        ],
        'TypeHintError': [
            'type hint', 'Type hint', 'type annotation', 'Type annotation'
        ],
        'UnboundLocalError':[
            "unbound",
        ],
        'ModuleNotFoundError': [
            'could not be resolved'
        ]

    }
    
    identified_type = 'Unknown Error'
    description = error_message

    for error_type, keywords in error_types.items():
        if any(keyword in error_message for keyword in keywords):
            identified_type = error_type
            break
    
    return {
        'Error Type': identified_type,
        'Description': description
    }

def extract_sample_input(description):
    # Regular expression to find content within <pre> tags under <h3>Sample Input
    pattern = r'<h3>Sample Input.*?</h3><pre>(.*?)</pre>'
    matches = re.findall(pattern, description, re.DOTALL)
    
    # Extract and clean up the inputs
    sample_inputs = [match.strip() for match in matches]
    return sample_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sample', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_error', action='store_true')
    parser.add_argument('--split', type=int, default=1)
    args = parser.parse_args()
    np.random.seed(args.seed)

    # with open('../test_records.jsonl', 'r') as f:
    #     test_records = [json.loads(line) for line in f]
    indices = []
    if args.no_error:
        with open(f'safes_{args.split}_{args.seed}.jsonl', 'r') as f:
            for line in f:
                indices.append(json.loads(line))
    else:
        with open(f'errors_{args.split}.jsonl', 'r') as f:
            for line in f:
                indices.append(json.loads(line))

    count = 0
    results = []
    save_file_name = f'google_bench_{args.seed}_{args.split}_SONNET.jsonl' if not args.no_error else f'google_bench_noerror_{args.seed}_{args.split}_SONNET.jsonl'
    for record in tqdm(indices):
        empty = False
        problem_id = record['problem_id']['bytesList']['value'][0]
        submission_id = record['submission_id']['bytesList']['value'][0]
        problem_id = base64.b64decode(problem_id).decode('utf-8')
        submission_id = base64.b64decode(submission_id).decode('utf-8')
        true_error_type = record['target']['int64List']['value']
        true_error_type = int(true_error_type[0])
        true_error_type = true_error_type - 1 if true_error_type != 1 else 1
        root_directory = '/home/ubuntu/mnt/agent/amazon-Q/NGDEBirds/NGDEBirdsScienceTransforms/src/birds_transforms/examples/Project_CodeNet'
        implementation_path = os.path.join(root_directory, 'data', problem_id, 'Python', f'{submission_id}.py')
        description_path = os.path.join(root_directory, 'problem_descriptions', f'{problem_id}.html')
        if os.path.exists(implementation_path) and os.path.exists(description_path):
            with open(implementation_path, 'r') as f:
                implementation = f.read()
            description_path = os.path.join(root_directory, 'problem_descriptions', f'{problem_id}.html')
            with open(description_path, 'r') as f:
                problem_description = f.read()
                info = descriptions.extract_input_information(problem_description)
                sample_inputs = extract_sample_input(problem_description)
            # # insert pyright
            command = ['pyright', implementation_path, '--outputjson']
            pyright_result = subprocess.run(command, capture_output=True, text=True)
            sa_output = eval(pyright_result.stdout)['generalDiagnostics']
            if len(sa_output) == 0:
                sa_type = {'Error Type': 'NoError', "Description": ""}
            else:
                error = sa_output[0]
                sa_type = identify_pyright_error(error['message'])
                
            if len(sample_inputs) == 0:
                empty = True
                running_error_types = [true_error_type]
            else:
                running_error_types = []
                for sample_input in sample_inputs:
                    sample_input = sample_input.strip()
                    process = subprocess.Popen(
                        ['python', implementation_path], 
                        stdin=subprocess.PIPE, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        )
                    try:
                        stdout, stderr = process.communicate(input=sample_input.encode(), timeout=1.0)
                        error_type_tmp = check_error_type(error_list, stderr.decode())
                        running_error_types.append(error_type_tmp)
                        # Check if the process is still alive and terminate if necessary
                        if process.poll() is None:  # poll() returns None if the process is still running
                            process.terminate()  # or process.kill() if you want to forcefully kill it
                    except subprocess.TimeoutExpired:
                        process.kill()
                        running_error_types.append('Timeout')
            if len(running_error_types) == 0:
                running_error_types = []
            else:
                running_types = []
                for error in running_error_types:
                    try:
                        running_type = error_to_index[error]
                    except KeyError:
                        running_type = 30
                    running_types.append(running_type)
                
            prompt = TEMPLATE.format(
                input=info,
                implementation=implementation
            )
            # try:
            #     ant_response = prompt_anthropic(
            #         system=SYSTEM_TEMPLATE,
            #         prompt=prompt,
            #         model_id=SONNET_3_5,
            #         temperature=0.0,
            #     )
            # except:
            #     continue
            
            max_retries = 10
            delay = 20
            try:
                for attempt in range(max_retries):
                    try:
                        ant_response = prompt_anthropic(
                            system=SYSTEM_TEMPLATE,
                            prompt=prompt,
                            model_id=SONNET_3_5,
                            temperature=0.0,
                        )
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        time.sleep(delay)
            except:
                continue

            # openai_response = prompt_openai(
            #     system=SYSTEM_TEMPLATE,
            #     prompt=prompt,
            #     model_id='gpt-4o',
            #     temperature=0.0,
            # )[0]
            # breakpoint()

            ant_error_type = extract_tag_list('Conclusion', ant_response)
            if len(ant_error_type) == 0:
                ant_error_type = 1
            else:
                ant_error_type = ant_error_type[0].strip()
            # potential_errors = extract_tag_list('Potential errors', ant_response)
            # if len(potential_errors) == 0:
            #     potential_errors = []
            # else:
            #     potential_errors = potential_errors[0].strip().split('\n')
            # openai_error_type = extract_tag_list('Conclusion', openai_response)[0].strip()
            result = {
                'problem_id': problem_id,
                'submission_id': submission_id,
                'ant_error_type': ant_error_type,
                # 'openai_error_type': openai_error_type,
                'true_error_type': true_error_type,
                'running_error_types': running_types,
                'sa_type': sa_type,
                'ant_response': ant_response,
                # 'openai_response': openai_response,
                'implementation': implementation,
                # 'potential_errors': potential_errors
                'empty': empty
            }
            results.append(result)
            count += 1
            print('Progress:', count / args.num_sample)
            
            if (count+1) % 10 == 0:
                # dump results to a jsonl file
                with open(save_file_name, 'w') as f: 
                    for result in results:
                        f.write(json.dumps(result) + '\n')            

    # dump results to a jsonl file
    with open(save_file_name, 'w') as f: 
        for result in results:
            f.write(json.dumps(result) + '\n')
            
