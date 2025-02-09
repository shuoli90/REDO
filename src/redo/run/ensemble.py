import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools

def compute_cm(all_instances, eval_results, predicted_all, sa_predicted=None):
    predicted_binary_error = []
    true_binary_error = []
    predicted_list = []
    true_predicted = []
    if sa_predicted != None:
        predicted_all = predicted_all + sa_predicted
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    for instance in all_instances:
        if instance not in eval_results['passed_instances'] and instance in predicted_all:
            true_positives.append(instance)
        elif instance in eval_results['passed_instances'] and instance in predicted_all:
            false_positives.append(instance)
        elif instance in eval_results['passed_instances'] and instance not in predicted_all:
            true_negatives.append(instance)
        else:
            false_negatives.append(instance)

    return true_positives, false_positives, true_negatives, false_negatives

def df_to_latex_table(df, row_order, column_order):
    # Pivot the DataFrame to get the desired format
    pivot_df = df.pivot(index='base', columns='aux', values='improve_cnt').fillna(0)
    
    # Reindex the DataFrame to the desired order
    pivot_df = pivot_df.reindex(index=row_order, columns=column_order, fill_value=0)
    
    # Convert the pivoted DataFrame to LaTeX format with integers
    latex_table = pivot_df.to_latex(
        buf=None,
        column_format='|'.join(['l'] + ['r'] * len(pivot_df.columns)),
        header=True,
        index=True,
        bold_rows=True,
        float_format=lambda x: '%d' % x  # Format numbers as integers
    )
    
    return latex_table

def create_heatmap(df, row_order, column_order, fontsize=30, semantic=False):
    # Pivot the DataFrame to get the desired format
    pivot_df = df.pivot(index='base', columns='aux', values='improve_cnt').fillna(0)
    
    # Reindex the DataFrame to the desired order
    pivot_df = pivot_df.reindex(index=row_order, columns=column_order, fill_value=0)
    
    # Create the heatmap
    plt.figure(figsize=(14, 14))
    sns.heatmap(
        pivot_df, 
        annot=True, 
        cmap='YlGnBu',
        linewidths=0.5, 
        linecolor='gray',
        annot_kws={'size': fontsize}  # Set the font size for annotations
    )
    
    # Set labels and title
    # plt.title('Heatmap of Improve Count', fontsize=fontsize)
    plt.xlabel('Aux', fontsize=30)
    plt.ylabel('Base', fontsize=30)
    
    # Adjust tick label size
    plt.xticks(fontsize=fontsize, rotation=45)
    plt.yticks(fontsize=fontsize, rotation=45)
    plt.tight_layout()
    if semantic:
        plt.savefig('a_heatmap_semantic.pdf')
    else:
        plt.savefig('a_heatmap.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic', action='store_true')
    args = parser.parse_args()

    methods = [
        'codestory-mixed', 'Demo', 'marscode', 'lingma', 'droid', 'autocoderover', 
    ]

    # Generate all two-element permutations
    permutations = list(itertools.permutations(methods, 2))

    results = {}
    results_table = []
    for perm in permutations:
        for model, method in zip(['base', 'aux'], perm):
            with open(f'eval_{method}_results.json', 'r') as f:
                eval_results = json.load(f)
            all_instances = []
            for key in eval_results:
                if args.semantic:
                    if key == 'assertion_instances':
                        continue
                all_instances.extend(eval_results[key])
            all_instances = list(set(all_instances))
            with open(f'predicted_{method}_results.json', 'r') as f:
                static_predicted_results = json.load(f)
            static_predicted_all = []
            for key in static_predicted_results:
                if 'predicted' in key:
                    static_predicted_all.extend(static_predicted_results[key])
            static_predicted_results_pyrightm = list(set(static_predicted_all))
            if method == 'autocoderover':
                prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240621_autocoderover-v20240620/all_preds.jsonl"
            elif method == 'aider':
                prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240523_aider/all_preds.jsonl"
            elif method == 'codestory-mixed':
                prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240702_codestory_aide_mixed/all_preds.jsonl"
            elif method == 'Demo':
                prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240627_abanteai_mentatbot_gpt4o/all_preds.jsonl"
            elif method == 'droid':
                prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240617_factory_code_droid/all_preds.jsonl"
            elif method == 'opus_func_margin':
                prediction_path = "~/predictions/opus_func_margin.jsonl"
            elif method == 'lingma':
                prediction_path = '/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240622_Lingma_Agent/all_preds.jsonl'
            elif method == 'marscode':
                prediction_patch = '/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240723_marscode-agent-dev/all_preds.jsonl'
            else:
                raise ValueError("Invalid method")

            preds = pd.read_json(prediction_path, lines=True)
            instances = preds['instance_id'].to_list()

            with open(f'predicted_{method}_results_pylint.json', 'r') as f:
                static_predicted_pylint = json.load(f)
            static_predicted_all_pylint = []
            for key in static_predicted_pylint:
                if 'predicted' in key:
                    static_predicted_all_pylint.extend(static_predicted_pylint[key])
            static_predicted_results_pylint = list(set(static_predicted_all_pylint))
            static_predicted_pylint_aggressive = []
            for instance_id, modified_issues in zip(instances, static_predicted_pylint['original_issues_lsit']):
                if len(modified_issues) > 0:
                    static_predicted_pylint_aggressive.append(instance_id)

            with open(f'predicted_{method}_results_new.json', 'r') as f:
                static_predicted_pyright = json.load(f)
            static_predicted_all_sa = []
            for key in static_predicted_pyright:
                if 'predicted' in key:
                    static_predicted_all_sa.extend(static_predicted_pyright[key])
            static_predicted_results_pyright = list(set(static_predicted_all_sa))
            static_predicted_pyright_aggressive = []
            for instance_id, modified_issues in zip(instances, static_predicted_pyright['original_issues']):
                if len(modified_issues) > 0:
                    static_predicted_pyright_aggressive.append(instance_id)

            if method == 'autocoderover':
                method_tmp = 'autocoderover-v20240620-gpt-4o-2024-05-13'
                file_path = f'runtime_detection_predicted_{method_tmp}_results_newpatch_singleshot.jsonl'
            else:
                file_path = f'runtime_detection_predicted_{method}_results_newpatch_singleshot.jsonl'
            predicted_results = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    predicted_results.append(json.loads(line.strip()))
            all_predicted_results = [result for result in predicted_results if result['instance_id'] in all_instances]
            filtered_predicted_results_pyrightm = [result for result in all_predicted_results if result['instance_id'] not in static_predicted_results_pyrightm]
            filtered_predicted_results_pyright = [result for result in all_predicted_results if result['instance_id'] not in static_predicted_results_pyright]
            filtered_predicted_results_pylint = [result for result in all_predicted_results if result['instance_id'] not in static_predicted_results_pylint]

            if method == 'codestory-mixed':
                method = 'Codestory'
            elif method == 'droid':
                method = 'Droid'
            elif method == 'autocoderover':
                method = 'Auto'
            elif method == 'aider':
                method = 'Aider'
            elif method == 'Demo':
                method = 'Mentatbot'
            elif method == 'lingma':
                method = 'Lingma'
            elif method == 'marscode':
                method = 'Marscode'

            unsafe_instances = []
            for pred in filtered_predicted_results_pylint:
                if 'Unsafe' in pred['conclusion']:
                    unsafe_instances.append(pred['instance_id'])
            safe_instances = []
            for pred in filtered_predicted_results_pylint:
                if 'Safe' in pred['conclusion']:
                    safe_instances.append(pred['instance_id'])

            true_positives, false_positives, true_negatives, false_negatives = compute_cm(all_instances, eval_results, unsafe_instances, static_predicted_results_pylint)
            results[model] = {
                'true_positives': true_positives, 
                'false_positives': false_positives, 
                'true_negatives': true_negatives, 
                'false_negatives': false_negatives,
                'method': method,}
        
        base = results['base']['method']
        aux = results['aux']['method']
        false_positive_base = results['base']['false_positives']
        true_positive_base = results['base']['true_positives']
        true_negative_aux = results['aux']['true_negatives']
        false_negative_aux = results['aux']['false_negatives']

        improve_cnt = 0
        degrade_cnt = 0
        for instance in false_positive_base:
            if instance in false_negative_aux:
                degrade_cnt += 1
                # print('degrade instance', instance)

        for instance in true_positive_base:
            if instance in true_negative_aux:
                improve_cnt += 1
                # print('improve instance', instance)
        # print('*'*30)
        print('improve cnt', improve_cnt, 'degrade cnt', degrade_cnt)
        perm_tmp = {
            'base': base,
            'aux': aux,
            'improve_cnt': improve_cnt - degrade_cnt,
            # 'degrade_cnt': degrade_cnt,
        }
        results_table.append(perm_tmp)
    # generate dataframe from results_table
    results_table = pd.DataFrame.from_records(results_table)
    order = ['Codestory', 'Mentatbot', 'Marscode', 'Lingma', 'Droid', 'Auto']
    create_heatmap(results_table, row_order=order, column_order=order, semantic=args.semantic)
    results_print = df_to_latex_table(results_table, row_order=order, column_order=order)
    print(results_print)
