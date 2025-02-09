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
    save_file_name = f'google_bench_{args.seed}_{args.split}_pyright.jsonl' if not args.no_error else f'google_bench_noerror_{args.seed}_{args.split}_pyright.jsonl'
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
                
            stderr = ""
            if len(sample_inputs) == 0:
                empty = True
                running_types = [true_error_type]
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
                        stderr = stderr.decode()
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

            # ant_error_type = extract_tag_list('Conclusion', ant_response)
            # if len(ant_error_type) == 0:
            #     ant_error_type = 1
            # else:
            #     ant_error_type = ant_error_type[0].strip()
            result = {
                'problem_id': problem_id,
                'submission_id': submission_id,
                # 'ant_error_type': ant_error_type,
                'true_error_type': true_error_type,
                'running_error_types': running_types,
                'sa_type': sa_type,
                # 'ant_response': ant_response,
                'implementation': implementation,
                'empty': empty,
                'stderr': stderr
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
            
