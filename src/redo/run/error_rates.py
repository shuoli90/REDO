# read in *.log files

import os
import argparse
import re
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filter_match(file_content, pattern, instance_id, special_list, match_string, instance_list, error_count):
    matches = [match.group() for match in re.finditer(pattern, file_content)]
    repo_name = instance_id.split("__")[0]
    if repo_name in special_list:
        match_string = match_string + ':'
    filtered_matches =  [match for match in matches if match.lower().startswith(match_string) or match.endswith("...")]
    if 'test_nameerror' in file_content or ': NameError' in file_content:
        return instance_list, error_count, False
    if len(filtered_matches) > 0:
        error_count += 1
        instance_list.append(instance_id)
    return instance_list, error_count, len(filtered_matches)>0

def generate_prefixes(s):
    """Generate all possible prefix substrings of a given string."""
    return [s[:i] for i in range(1, len(s) + 1)]

def generate_patterns(error_name):
    prefixes = generate_prefixes(error_name)[1:]
    prefix_pattern = r'\b(' + '|'.join(re.escape(prefix) for prefix in prefixes) + r').\.\.\.'
    # Create regex pattern for the exact match "keyword:"
    exact_pattern1 = re.escape(error_name) + r':'
    exact_pattern2 = re.escape(error_name)
    pattern = r'(' + prefix_pattern + r')|(' + exact_pattern1 + r')|(' + exact_pattern2 + r')'
    return pattern


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    file_count = 0
    indentation_error_count = 0
    syntax_error_count = 0
    type_error_count = 0
    name_error_count = 0
    attribution_error_count = 0
    assertion_error_count = 0
    value_error_count = 0
    passed_count = 0
    index_error_count = 0
    depre_warning_count = 0

    syntax_instances = []
    indentation_instances = []
    type_instances = []
    name_instances = []
    attribute_instances = []
    assertion_instances = []
    value_instances = []
    passed_instances = []
    unknown_instances = []
    index_instances = []
    depre_instances = []

    type_pattern = generate_patterns('TypeError')
    syntax_pattern = generate_patterns('SyntaxError')
    indentation_pattern = generate_patterns('IndentationError')
    attribute_pattern = generate_patterns('AttributeError')
    name_pattern = generate_patterns('NameError')
    assertion_pattern = generate_patterns('AssertionError')
    value_pattern = generate_patterns('ValueError')
    index_pattern = generate_patterns('IndexError')
    depreation_pattern = generate_patterns('DeprecationWarning')

    log_dirs = [
        # "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240702_codestory_aide_mixed/logs/",
        # "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240627_abanteai_mentatbot_gpt4o/logs/",
        # '/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240622_Lingma_Agent/logs/',
        # "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240617_factory_code_droid/logs/",
        # "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240621_autocoderover-v20240620/logs/",
        # "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240523_aider/logs/",
        "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240723_marscode-agent-dev/logs"
    ]

    results_table = {
        "Method": ['Codestory', 'Mentatbot', 'Lingma', 'Droid', 'Auto', 'Marscode'],
        # "Method": ['Codestory', 'Mentatbot'],
        "Syntax": [],
        # "Inde": [],
        "Type": [],
        "Name": [],
        "Attribute": [],
        # "Asse": [],
    }

    for idx, log_dir in enumerate(log_dirs):
        # if idx > 1:
        #     break
        # predicted_file = ""
        for instance_id in tqdm(os.listdir(log_dir), total=len(os.listdir(log_dir))):
            if os.path.isdir(os.path.join(log_dir, instance_id)):
                if os.path.exists(os.path.join(log_dir, instance_id, "report.json")):
                    file_count += 1
                    with open(os.path.join(log_dir, instance_id, 'report.json'), "r") as f:
                        file_content = json.load(f)
                        if file_content[instance_id]['resolved']:
                            passed_count += 1
                            passed_instances.append(instance_id)
                        else:
                            with open(os.path.join(log_dir, instance_id, 'test_output.txt')) as f:
                                file_content = f.read()
                                # instance_id = file.split(".")[0]
                                # repo_name = instance_id.split("__")[0]
                                special_list = ['matplotlib', 'pytest-dev']
                                # if predicted_file == "":
                                #     predicted_file = file.split("_")[3:]
                                #     predicted_file = "_".join(predicted_file).split(".")[0]

                                syntax_instances, syntax_error_count, syntax_found = filter_match(
                                    pattern=syntax_pattern,
                                    file_content=file_content,
                                    instance_id=instance_id,
                                    special_list=special_list,
                                    match_string='syntaxerror',
                                    instance_list=syntax_instances,
                                    error_count=syntax_error_count)
                                
                                indentation_instances, indentation_error_count, indentation_found = filter_match(
                                    pattern=indentation_pattern,
                                    file_content=file_content,
                                    instance_id=instance_id,
                                    special_list=special_list,
                                    match_string='indentationerror',
                                    instance_list=indentation_instances,
                                    error_count=indentation_error_count)
                                
                                type_instances, type_error_count, type_found = filter_match(
                                    pattern=type_pattern, 
                                    file_content=file_content, 
                                    instance_id=instance_id, 
                                    special_list=special_list, 
                                    match_string='typeerror', 
                                    instance_list=type_instances, 
                                    error_count=type_error_count)
                                
                                name_instances, name_error_count, name_found = filter_match(
                                    pattern=name_pattern,
                                    file_content=file_content,
                                    instance_id=instance_id,
                                    special_list=special_list,
                                    match_string='nameerror',
                                    instance_list=name_instances,
                                    error_count=name_error_count)
                        
                                attribute_instances, attribution_error_count, attribute_found = filter_match(
                                    pattern=attribute_pattern,
                                    file_content=file_content,
                                    instance_id=instance_id,
                                    special_list=special_list,
                                    match_string='attributeerror',
                                    instance_list=attribute_instances,
                                    error_count=attribution_error_count)
                                
                                value_instances, value_error_count, value_found = filter_match(
                                    pattern=value_pattern,
                                    file_content=file_content,
                                    instance_id=instance_id,
                                    special_list=special_list,
                                    match_string='valueerror',
                                    instance_list=value_instances,
                                    error_count=value_error_count)

                                assertion_instances, assertion_error_count, assertion_found = filter_match(
                                    pattern=assertion_pattern,
                                    file_content=file_content,
                                    instance_id=instance_id,
                                    special_list=special_list,
                                    match_string='assertionerror',
                                    instance_list=assertion_instances,
                                    error_count=assertion_error_count)
                                
                                index_instances, index_error_count, index_found = filter_match(
                                    pattern=index_pattern,
                                    file_content=file_content,
                                    instance_id=instance_id,
                                    special_list=special_list,
                                    match_string='indexerror',
                                    instance_list=index_instances,
                                    error_count=index_error_count)
                                
                                depre_instances, depre_warning_count, depre_found = filter_match(
                                    pattern=depreation_pattern,
                                    file_content=file_content,
                                    instance_id=instance_id,
                                    special_list=special_list,
                                    match_string='deprecationwarning',
                                    instance_list=depre_instances,
                                    error_count=depre_warning_count)
                                
                                if not any([syntax_found, indentation_found, type_found, name_found, attribute_found, assertion_found, index_found, depre_found, value_found]):
                                    unknown_instances.append(instance_id)

        if args.verbose:
            # print('log dir', predicted_file)
            print('Passed count', passed_count)
            print('Syntax error count', len(list(set(syntax_instances))))
            # print('Syntax rate', syntax_error_count / file_count)
            print('Indentation error count', len(list(set(indentation_instances))))
            # print('Indentation rate', indentation_error_count / file_count)
            print('Type error count', len(list(set(type_instances))))
            # print('Type rate', type_error_count / file_count)
            print('Name error count', len(list(set(name_instances))))
            # print('Name rate', name_error_count / file_count)
            print('Attribution error count', len(list(set(attribute_instances))))
            # print('Attribution rate', attribution_error_count / file_count)
            print('Assertion error count', len(list(set(assertion_instances))))
            # print('Assertion rate', assertion_error_count / file_count)
            print('Unknown count', len(unknown_instances))
        
        results_table["Syntax"].append(len(list(set(syntax_instances))))
        # results_table["Inde"].append(len(list(set(indentation_instances))))
        results_table["Type"].append(len(list(set(type_instances))))
        results_table["Name"].append(len(list(set(name_instances))))
        results_table["Attribute"].append(len(list(set(attribute_instances))))
        # results_table["Asse"].append(len(list(set(assertion_instances))))

        attribute_instances.sort()
        type_instances.sort()
        name_instances.sort()
        value_instances.sort()
        assertion_instances.sort()
        syntax_instances.sort()
        indentation_instances.sort()
        index_instances.sort()
        if args.verbose:
            print('attribute error instances', attribute_instances)
            print('type error instances', type_instances)
            print('name error instances', name_instances)
            print('assertion error instances', assertion_instances)
            print('syntax error instances', syntax_instances)
            print('index error instances', index_instances)
            print('indentation error instances', indentation_instances)
            print('Deprecation warning instance', depre_instances)
            print('Unkown instance', unknown_instances)

        results = {
            'attribute_instances': attribute_instances,
            'type_instances': type_instances,
            'name_instances': name_instances,
            'assertion_instances': assertion_instances,
            'syntax_instances': syntax_instances,
            'indentation_instances': indentation_instances,
            'index_instances': index_instances,
            'depre_instances': depre_instances,
            'value_instances': value_instances, 
            'passed_instances': passed_instances,
            'unknown_instances': unknown_instances
        }
        # write results to a json file
        with open(f'./eval_marscode_results.json', 'w') as f:
            json.dump(results, f)
        breakpoint()
    
    df = pd.DataFrame(results_table)

    # Set 'Method' column as the index
    df.set_index('Method', inplace=True)
    # Set the positions and width for the bars

    bar_width = 0.1
    error_types = df.columns
    positions = np.arange(len(error_types))

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = ['mediumseagreen', 'steelblue', 'tomato', 'gold', 'orchid', 'lightcoral']
    
    # Plot each method
    for i, method in enumerate(df.index):
        ax.bar(positions + i * bar_width, df.loc[method], width=bar_width, label=method, color=colors[i])
    
    # Set the x-ticks to be in the center of the groups
    ax.set_xticks(positions + bar_width * (len(df.index) - 1) / 2)
    ax.set_xticklabels(error_types, fontsize=18)
    ax.set_yticklabels(ax.get_yticks(), fontsize=18)

    # Add labels and legend
    # plt.title('Distribution of Different Errors')
    plt.xlabel('Error Types', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.legend(
        fontsize=18, 
        bbox_to_anchor=(1.04, 0.5), 
        loc="center left", 
        borderaxespad=0)
    plt.tight_layout()
    plt.grid()
    plt.savefig('Error_distribution.pdf')