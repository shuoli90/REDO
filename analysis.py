import json
import numpy as np
import re
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import argparse
import numpy as np
import os

def map_to_python_errors(pyflakes_message):
    """
    Maps pylint messages to Python error types.
    """
    error_map = {
        "assert": "AssertionError",
        "has no attribute": "AttributeError",
        "decimal": "decimal",
        "unexpected EOF": "EOFError",
        "No such file or directory": "FileNotFoundError",
        "No module named": "ImportError",
        "unexpected indent": "IndentationError",
        "index out of range": "IndexError",
        "key not found": "KeyError",
        "math domain error": "MathDomainError",
        "out of memory": "MemoryError",
        "Module not found": "ModuleNotFoundError",
        "is not defined": "NameError",
        "OS error": "OSError",
        "numerical result out of range": "OverflowError",
        "invalid regular expression": "re.error",
        "maximum recursion depth exceeded": "RecursionError",
        "runtime error": "RuntimeError",
        "StopIteration": "StopIteration",
        "invalid syntax": "SyntaxError",
        "inconsistent use of tabs and spaces": "TabError",
        "unsupported operand type(s)": "TypeError",
        "local variable referenced before assignment": "UnboundLocalError",
        "invalid literal for int()": "ValueError",
        "division by zero": "ZeroDivisionError",
        "axis": "numpy.AxisError"
    }

    for key, error_type in error_map.items():
        if key in pylint_message:
            return error_type
    return "UnknownError"

def extract_error_types(pylint_output):
    """
    Extracts the error types from pylint output.
    """
    error_pattern = re.compile(r'^[CRWEF]\d{4}: (.*)')
    error_types = set()

    for line in pylint_output:
        match = error_pattern.match(line)
        if match:
            pylint_message = match.group(1)
            error_type = map_to_python_errors(pylint_message)
            error_types.add(error_type)

    return list(error_types)

error_to_index = {
    'NoError': 1,
    'No Error': 1,
    'Other': 2,
    'Timeout': 3,
    'AssertionError': 4,
    'AttributeError': 5,
    'Attribute Error': 5,
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
    'Name Error': 16,
    'OSError': 17,
    'OverflowError': 18,
    're.error': 19,
    'RecursionError': 20,
    'RuntimeError': 21,
    'StopIteration': 22,
    'SyntaxError': 23,
    'TabError': 24,
    'TypeError': 25,
    'Type Error': 25,
    'UnboundLocalError': 26,
    'ValueError': 27,
    'ZeroDivisionError': 28,
    'numpy.AxisError': 29,
    'Unknown Error':30,
    'TypeHintError':30,
    'Type Hint Error': 30,
}

def check_all_ones(lst):
    for element in lst:
        if element != 1:
            return element
    return 1

def multi_index_to_latex_with_subscript_and_benchmark_order(df: pd.DataFrame,  method_order: list, metric_order: list)-> str:
    """
    Converts a multi-index pandas DataFrame to a LaTeX table with Score values and their corresponding
    std values as subscripts, and orders the Benchmark index by a given sequence.

    Parameters:
    df (pd.DataFrame): The input DataFrame should have 'Method', 'Benchmark', 'Metric', 'Score', and 'std' columns.
    benchmark_order (list): A list of Benchmark names to specify the order of the Benchmark index.

    Returns:
    str: A LaTeX table as a string.
    """
    # Pivot the dataframe to organize Benchmark and Metric as rows, Method as columns
    pivot_df = df.pivot_table(index=['Metric'], columns='Method', values=['Score', 'std'], aggfunc='first')
    
    # Reorder the index based on the given benchmark order
    # Reorder the index and columns based on the given orders
    ordered_pivot_df = pivot_df.reindex(metric_order, level='Metric')
    ordered_pivot_df = ordered_pivot_df.reindex(method_order, axis=1, level='Method')
    
    # Create a new DataFrame where Score and std are combined with LaTeX subscript formatting
    formatted_df = ordered_pivot_df['Score'].copy()
    for method in formatted_df.columns:
        formatted_df[method] = formatted_df[method].round(1).astype(str) + \
                               " $_{" + ordered_pivot_df['std'][method].round(1).astype(str) + "}$"
    
    # Convert to LaTeX
    latex_table = formatted_df.to_latex(
        multirow=True,
        column_format="ll" + "c" * len(df['Method'].unique()),
        escape=False
    )

    return latex_table

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--error_only', action='store_true', help='Only consider error cases')
    parser.add_argument('-n', '--no_context', action='store_true')
    args = parser.parse_args()

    if args.error_only and args.no_context:
        file_list = ['google_bench_nocontext']
        pyright_file_list = ['google_bench']
    elif args.error_only:
        file_list = ['google_bench']
        pyright_file_list = ['google_bench']
    elif args.no_context:
        file_list = ['google_bench_nocontext', 'google_bench_nocontext_noerror']
        pyright_file_list = ['google_bench', 'google_bench_noerror']
    else:
        file_list = [f'google_bench', f'google_bench_noerror']
        pyright_file_list = [f'google_bench', f'google_bench_noerror']

    seeds = [21, 42, 84]
    # seeds = [42]
    seed_list = []
    method_list = []
    metric_list = []
    score_list = []
    for seed in seeds:
        tmp = {}
        final_results = []
        # dump results to a jsonl file
        if file_list == ['google_bench_nocontext_noerror', 'google_bench_nocontext']:
            if os.path.exists(f'./runtime/google_bench_all_nocontext_{seed}_SONNET.jsonl'):
                with open(f'./runtime/google_bench_all_nocontext_{seed}_SONNET.jsonl', 'r') as f: 
                    for line in f:
                        final_results.append(json.loads(line))
        elif file_list == ['google_bench_nocontext']:
            if os.path.exists(f'./runtime/google_bench_error_nocontext_{seed}_SONNET.json'):
                with open(f'./runtime/google_bench_error_nocontext_{seed}_SONNET.json', 'r'):
                    for line in f:
                        final_results.append(json.loads(line))
        elif file_list == ['google_bench_noerror', 'google_bench']:
            if os.path.exists(f'./runtime/google_bench_all_{seed}_SONNET.jsonl'):
                with open(f'./runtime/google_bench_all_{seed}_SONNET.jsonl', 'r') as f:
                    for line in f:
                        final_results.append(json.loads(line))
        elif file_list == ['google_bench']:
            if os.path.exists(f'./runtime/google_bench_error_{seed}_SONNET.json'):
                with open(f'./runtime/google_bench_error_{seed}_SONNET.json', 'r') as f:
                    for line in f:
                        final_results.append(json.loads(line))
        if len(final_results) == 0:

            results_pyright = []
            for file in pyright_file_list:
                for idx in range(1, 5):
                    if 'noerror' in file:
                        file_name = file + f'_{seed}_{idx}_pyright.jsonl'
                    else:
                        file_name = file + f'_42_{idx}_pyright.jsonl'
                    with open(file_name, 'r') as f:
                        for line in f:
                            results_pyright.append(json.loads(line))
            pyright_table = pd.DataFrame.from_records(results_pyright)

            sonnet_results = []
            for file in file_list:
                for idx in range(1, 5):
                    file_name = file + f'_42_{idx}_SONNET.jsonl'
                    with open(file_name, 'r') as f:
                        for idx, line in enumerate(f):
                            tmp = json.loads(line.strip())
                            try:
                                tmp['ant_error_type'] = int(tmp['ant_error_type'])
                            except:
                                continue
                            sonnet_results.append(tmp)
            SONNET_table = pd.DataFrame.from_records(sonnet_results)

            for tmp in results_pyright:
                problem_id = tmp['problem_id']
                submission_id = tmp['submission_id']
                sa_type = tmp['sa_type']['Error Type']
                sa_message = tmp['sa_type']['Description']
                sa_type = error_to_index[sa_type]
                tmp['sa_type'] = sa_type
                # tmp['pylint_error_type'] = project_pylint_to_error(tmp['pylint_output'][1]) if len(tmp['pylint_output']) > 0 else 1

                SONNET_record = SONNET_table[SONNET_table['problem_id'] == tmp['problem_id']]
                SONNET_record = SONNET_record[SONNET_record['submission_id'] == tmp['submission_id']]
                if len(SONNET_record) == 0:
                    tmp['ant_error_type_sa'] = int(sa_type)
                else:
                    tmp['ant_error_type_sa'] = int(SONNET_record['ant_error_type'].iloc[0].item())

                tmp['running_error_types'] = check_all_ones(tmp['running_error_types'])
                tmp['prediction_sa'] = sa_type if sa_type != 1 else tmp['ant_error_type_sa']
                final_results.append(tmp)
            
            # dump results to a jsonl file
            if file_list == ['google_bench_nocontext_noerror', 'google_bench_nocontext']:
                with open(f'./runtime/google_bench_all_nocontext_{seed}_SONNET.jsonl', 'w') as f: 
                    for result in final_results:
                        f.write(json.dumps(result) + '\n')
            elif file_list == ['google_bench_nocontext']:
                with open(f'./runtime/google_bench_error_nocontext_{seed}_SONNET.jsonl', 'w') as f:
                    for result in final_results:
                        f.write(json.dumps(result) + '\n')
            elif file_list == ['google_bench_noerror', 'google_bench']:
                with open(f'./runtime/google_bench_all_{seed}_SONNET.jsonl', 'w') as f:
                    for result in  final_results:
                        f.write(json.dumps(result) + '\n')
            elif file_list == ['google_bench']:
                 with open(f'./runtime/google_bench_error_{seed}_SONNET.jsonl', 'w') as f:
                    for result in final_results:
                        f.write(json.dumps(result) + '\n')
        
        sa_types = []
        pylint_types = []
        error_types = []
        predictions_sa = []
        predictions_pylint = []
        gts = []
        running_gts = []
        cnt = 0
        for result in final_results:
            error_types.append(result['ant_error_type_sa'])
            gts.append(result['true_error_type'])
            sa_types.append(result['sa_type'])
            # pylint_types.append(result['pylint_error_type'])
            predictions_sa.append(result['prediction_sa'])
            # predictions_pylint.append(result['prediction_pylint'])
            running_gts.append(result['running_error_types'])
        predictions_sa = np.array(predictions_sa)
        # predictions_pylint = np.array(predictions_pylint)
        error_types = np.array(error_types)
        gts = np.array(gts)
        sa_types = np.array(sa_types)
        running_gts = np.array(running_gts)

        for errors, method in zip([error_types, sa_types, predictions_sa], ['LLM', 'PyRight', 'REDO-PyRight']):
        
            weighted_f1 = f1_score(
                y_true=gts, 
                y_pred=errors, 
                average='weighted')
            seed_list.append(seed)
            method_list.append(method)
            metric_list.append('W.F1')
            score_list.append(weighted_f1)

            accuracy = accuracy_score(
                y_true=gts,
                y_pred=errors
            )
            seed_list.append(seed)
            method_list.append(method)
            metric_list.append('Accuracy')
            score_list.append(accuracy)

            weighted_f1 = f1_score(
                y_true=running_gts, 
                y_pred=errors, 
                average='weighted')
            seed_list.append(seed)
            method_list.append(method)
            metric_list.append('Running W.F1')
            score_list.append(weighted_f1)

            accuracy = accuracy_score(
                y_true=running_gts,
                y_pred=errors
            )
            seed_list.append(seed)
            method_list.append(method)
            metric_list.append('Running Accuracy')
            score_list.append(accuracy)
    
    arrays = [
        method_list,
        metric_list
        ]
    index = pd.MultiIndex.from_arrays(arrays, names=('Method', 'Metric'))
    score_list = [item * 100 for item in score_list]
    df = pd.DataFrame({'Score': score_list, 'Seed': seed_list}, index=index)
    df_mean = df.groupby(level=['Method', 'Metric']).mean().reset_index()
    df_std = df.groupby(level=['Method', 'Metric']).std().reset_index()
    df_mean['std'] = df_std['Score']
    df_mean = df_mean.drop('Seed', axis=1)
    df_latex = multi_index_to_latex_with_subscript_and_benchmark_order(
        df_mean,
        metric_order=['Accuracy', 'Running Accuracy', 'W.F1', 'Running W.F1'],
        method_order=['PyRight', 'LLM', 'REDO-PyRight'])
    print(df_latex)
    # Dump DataFrame to JSON file
    df.to_json('output.json', orient='records', lines=True)




            