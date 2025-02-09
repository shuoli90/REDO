import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

def multi_index_to_latex_with_subscript_and_benchmark_order(df: pd.DataFrame, benchmark_order: list, method_order: list, metric_order: list)-> str:
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
    pivot_df = df.pivot_table(index=['Benchmark', 'Metric'], columns='Method', values=['Score', 'std'], aggfunc='first')
    
    # Reorder the index based on the given benchmark order
    # Reorder the index and columns based on the given orders
    ordered_pivot_df = pivot_df.reindex(benchmark_order, level='Benchmark')
    ordered_pivot_df = ordered_pivot_df.reindex(metric_order, level='Metric')
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


def compute_cm(all_instances, eval_results, predicted_all, sa_predicted=None):
    predicted_binary_error = []
    true_binary_error = []
    predicted_list = []
    true_predicted = []
    if sa_predicted != None:
        predicted_all = predicted_all + sa_predicted
    for instance in all_instances:
        # if instance not in eval_results['passed_instances']:
        if (instance not in eval_results['passed_instances']) and (instance not in eval_results['assertion_instances']): 
            true_binary_error.append(1)
        else:
            true_binary_error.append(0)

        if instance in predicted_all:
            predicted_binary_error.append(1)
            predicted_list.append(instance)
        else:
            predicted_binary_error.append(0)

    zero_zero = []
    one_one = []
    one_zero = []
    zero_one = []
    for predicted, gt in zip(predicted_binary_error, true_binary_error):
        if predicted == 1 and gt == 1:
            one_one.append(1)
        elif predicted == 1 and gt == 0:
            one_zero.append(1)
        elif predicted == 0 and gt == 1:
            zero_one.append(1)
        elif predicted == 0 and gt == 0:
            zero_zero.append(1)
        else:
            pass
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = len(zero_zero)
    cm[1, 0] = len(one_zero)
    cm[0, 1] = len(zero_one)
    cm[1, 1] = len(one_one)
    precision = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    recall = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    f1 = 2 * (precision * recall) / (precision + recall)
    wf1 = f1_score(true_binary_error, predicted_binary_error, average='weighted')
    return precision, recall, f1, wf1, cm, predicted_list

 # Plotting the clustered bar plot
def plot_clustered_multi_index_bar(df, first_level_order):
    # drop all rows whose metric is 'F1'
    new_index = pd.MultiIndex.from_product(
        [first_level_order, df.index.get_level_values('Benchmark').unique(), df.index.get_level_values('Metric').unique()],
        names=['Method', 'Benchmark', 'Metric']
    )
    df_new = df.reindex(new_index)
    # Unstack the second and third levels of the index to get a DataFrame suitable for bar plotting
    unstacked_df = df_new.unstack(level=[1, 2])
    
    # Set the positions and width for the bars
    bar_width = 0.05
    positions = np.arange(len(unstacked_df))

    # Define colors for the third level
    colors = {
        'Precision': 'skyblue',
        'Recall': 'coral'
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot each group and store handles for legend
    handles = {}
    labels = []
    for i, col in enumerate(unstacked_df.columns):
        third_level = col[2]
        if col not in ['Precision', 'Recall']:
            continue
        bar = ax.bar(positions + i * bar_width, unstacked_df[col], width=bar_width, label=third_level, color=colors[third_level])
        if third_level not in handles:
            handles[third_level] = bar
            labels.append(third_level)
    
    # Set the x-ticks to be in the center of the groups
    ax.set_xticks(positions + bar_width * (len(unstacked_df.columns) - 1) / 2)
    ax.set_xticklabels(unstacked_df.index.get_level_values(0))

    # Show only the third level in the legend
    plt.legend(
        handles=[handles[key] for key in labels], 
        labels=labels, 
        fontsize=30, 
        bbox_to_anchor=(0.5, 1.2), 
        loc="upper center", 
        borderaxespad=0,
        ncol=2)
    # put the legend to the mid-upper

    plt.xlabel("Method", fontsize=30)
    plt.ylabel("Score", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig("ablation.pdf")


if __name__ == "__main__":

    # methods = [
    #     'marscode', 'droid', 'lingma'
    # ]
    methods = [
        'codestory-mixed', 'Demo', 'marscode', 'lingma', 'droid', 'autocoderover',
    ]
    # methods = [
    #      'codestory-mixed', 'autocoderover'
    # ]

    method_list = []
    bench_list = []
    score_name = []
    score_list = []
    pyrights = []
    seeds = []

    for method in methods:
        with open(f'eval_{method}_results.json', 'r') as f:
            eval_results = json.load(f)
        all_instances = []
        for key in eval_results:
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
            prediction_path = '/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240723_marscode-agent-dev/all_preds.jsonl'
        else:
            raise ValueError("Invalid method")
        preds = pd.read_json(prediction_path, lines=True)
        instances = preds['instance_id'].to_list()

        # with open(f'predicted_{method}_results_pylint.json', 'r') as f:
        #     static_predicted_pylint = json.load(f)
        # static_predicted_all_pylint = []
        # for key in static_predicted_pylint:
        #     if 'predicted' in key:
        #         static_predicted_all_pylint.extend(static_predicted_pylint[key])
        # static_predicted_results_pylint = list(set(static_predicted_all_pylint))
        # static_predicted_pylint_aggressive = []
        # for instance_id, modified_issues in zip(instances, static_predicted_pylint['original_issues_lsit']):
        #     if len(modified_issues) > 0:
        #         static_predicted_pylint_aggressive.append(instance_id)

        if method == 'autocoderover':
            method_name = 'ACR'
        elif method == 'aider':
            method_name = 'Aider'
        elif method == 'codestory-mixed':
            method_name = 'CodeStory'
        elif method == 'Demo':
            method_name = 'Demo'
        elif method == 'droid':
            method_name = 'Droid'
        elif method == 'lingma':
            method_name ='Lingma'
        elif method == 'marscode':
            method_name = 'Marscode'
        else:
            raise ValueError("Invalid method")
        
        with open(f'predicted_{method}_results_pyflakes.json', 'r') as f:
            static_predicted_pyflakes = json.load(f)
        static_predicted_all_pyflakes = []
        for key in static_predicted_pyflakes:
            if 'predicted' in key:
                static_predicted_all_pyflakes.extend(static_predicted_pyflakes[key])
        static_predicted_results_pyflakes = list(set(static_predicted_all_pyflakes))
        # static_predicted_pyflakes_aggressive = []
        # for instance_id, modified_issues in zip(instances, static_predicted_pyflakes['original_issues_lsit']):
        #     if len(modified_issues) > 0:
        #         static_predicted_pyflakes_aggressive.append(instance_id)

        with open(f'predicted_{method}_results_new.json', 'r') as f:
            static_predicted_pyright = json.load(f)
        static_predicted_all_sa = []
        for key in static_predicted_pyright:
            if 'predicted' in key:
                static_predicted_all_sa.extend(static_predicted_pyright[key])
        static_predicted_results_pyright = list(set(static_predicted_all_sa))
        # static_predicted_pyright_aggressive = []
        # for instance_id, modified_issues in zip(instances, static_predicted_pyright['original_issues']):
        #     if len(modified_issues) > 0:
        #         static_predicted_pyright_aggressive.append(instance_id)

        if method == 'autocoderover':
            method_tmp = 'autocoderover-v20240620-gpt-4o-2024-05-13'
            file_path = f'runtime_detection_predicted_{method_tmp}_results_newpatch_singleshot.jsonl'
        else:
            file_path = f'runtime_detection_predicted_{method}_results_newpatch_singleshot.jsonl'
        predicted_results_1 = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                predicted_results_1.append(json.loads(line.strip()))
        
        if method == 'autocoderover':
            method_tmp = 'autocoderover-v20240620-gpt-4o-2024-05-13'
            file_path = f'runtime_detection_predicted_{method_tmp}_results_ant_singleshot_SONNET_05.jsonl'
        else:
            file_path = f'runtime_detection_predicted_{method}_results_ant_singleshot_SONNET_05.jsonl'
        predicted_results_OPUS = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                predicted_results_OPUS.append(json.loads(line.strip()))
        
        if method == 'autocoderover':
            method_tmp = 'autocoderover-v20240620-gpt-4o-2024-05-13'
            file_path = f'runtime_detection_predicted_{method_tmp}_results_newpatch_singleshot_new2.jsonl'
        else:
            file_path = f'runtime_detection_predicted_{method}_results_newpatch_singleshot_new2.jsonl'
        predicted_results_2 = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                predicted_results_2.append(json.loads(line.strip()))

        if method == 'autocoderover':
            method_tmp = 'autocoderover-v20240620-gpt-4o-2024-05-13'
            file_path = f'runtime_detection_predicted_{method_tmp}_results_newpatch_singleshot_new3.jsonl'
        else:
            file_path = f'runtime_detection_predicted_{method}_results_newpatch_singleshot_new3.jsonl'
        predicted_results_3 = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                predicted_results_3.append(json.loads(line.strip()))

        all_predicted_results_1 = [result for result in predicted_results_1 if result['instance_id'] in all_instances]
        filtered_predicted_results_pyright_1 = [result for result in all_predicted_results_1 if result['instance_id'] not in static_predicted_results_pyright]
        filtered_predicted_results_pyflakes_1 = [result for result in all_predicted_results_1 if result['instance_id'] not in static_predicted_results_pyflakes]

        all_predicted_results_2 = [result for result in predicted_results_2 if result['instance_id'] in all_instances]
        filtered_predicted_results_pyright_2 = [result for result in all_predicted_results_2 if result['instance_id'] not in static_predicted_results_pyright]
        filtered_predicted_results_pyflakes_2 = [result for result in all_predicted_results_2 if result['instance_id'] not in static_predicted_results_pyflakes]

        all_predicted_results_3 = [result for result in predicted_results_3 if result['instance_id'] in all_instances]
        filtered_predicted_results_pyright_3 = [result for result in all_predicted_results_3 if result['instance_id'] not in static_predicted_results_pyright]
        filtered_predicted_results_pyflakes_3 = [result for result in all_predicted_results_3 if result['instance_id'] not in static_predicted_results_pyflakes]

        filtered_predicted_results_pyright_list = [filtered_predicted_results_pyright_1, filtered_predicted_results_pyright_2, filtered_predicted_results_pyright_3]
        filtered_predicted_results_pyflakes_list = [filtered_predicted_results_pyflakes_1, filtered_predicted_results_pyflakes_2, filtered_predicted_results_pyflakes_3]

        # all_predicted_results_OPUS = [result for result in predicted_results_OPUS if result['instance_id'] in all_instances]
        # filtered_predicted_results_pyright_OPUS = [result for result in all_predicted_results_OPUS if result['instance_id'] not in static_predicted_results_pyright]
        # filtered_predicted_results_pyflakes_OPUS = [result for result in all_predicted_results_OPUS if result['instance_id'] not in static_predicted_results_pyflakes]

        # all_predicted_results_2 = [result for result in predicted_results_2 if result['instance_id'] in all_instances]
        # filtered_predicted_results_pyright_2 = [result for result in all_predicted_results_2 if result['instance_id'] not in static_predicted_results_pyright]
        # filtered_predicted_results_pyflakes_2 = [result for result in all_predicted_results_2 if result['instance_id'] not in static_predicted_results_pyflakes]

        # all_predicted_results_3 = [result for result in predicted_results_3 if result['instance_id'] in all_instances]
        # filtered_predicted_results_pyright_3 = [result for result in all_predicted_results_3 if result['instance_id'] not in static_predicted_results_pyright]
        # filtered_predicted_results_pyflakes_3 = [result for result in all_predicted_results_3 if result['instance_id'] not in static_predicted_results_pyflakes]

        # filtered_predicted_results_pyright_list = [filtered_predicted_results_pyright_OPUS, filtered_predicted_results_pyright_OPUS, filtered_predicted_results_pyright_OPUS]
        # filtered_predicted_results_pyflakes_list = [filtered_predicted_results_pyflakes_OPUS, filtered_predicted_results_pyflakes_OPUS, filtered_predicted_results_pyflakes_OPUS]

        # filtered_predicted_results_pyright_list = [filtered_predicted_results_pyright_1, filtered_predicted_results_pyright_3]
        # filtered_predicted_results_pyflakes_list = [filtered_predicted_results_pyflakes_1, filtered_predicted_results_pyflakes_3]

        three_results = []
        for index, all_predicted_results in enumerate([all_predicted_results_1, all_predicted_results_2, all_predicted_results_3]):
        # for index, all_predicted_results in enumerate([all_predicted_results_OPUS, all_predicted_results_OPUS, all_predicted_results_OPUS]):
        # for index, all_predicted_results in enumerate([all_predicted_results_1, all_predicted_results_3]):
            # if index != 0:
            #     continue
            # pyrightM
            all_instances = []
            for key in eval_results:
                all_instances.extend(eval_results[key])
            all_instances = list(set(all_instances))
            # pyrightm_precision, pyrightm_recall, pyrightm_f1, pyrightm_wf1, pyrightm_cm, pyrightm_predicted = compute_cm(all_instances, eval_results, static_predicted_results_pyrightm)
            # seeds.extend([index]*4)
            # method_list.extend(['PyRightM']*4)
            # bench_list.extend([method_name]*4)
            # score_name.extend(['Precision', 'Recall', 'F1', 'WF1'])
            # score_list.extend([pyrightm_precision, pyrightm_recall, pyrightm_f1, pyrightm_wf1])

            # # pyright
            pyright_precision, pyright_recall, pyright_f1, pyright_wf1, pyright_cm, pyright_predicted = compute_cm(all_instances, eval_results, static_predicted_results_pyright)
            seeds.extend([index]*4)
            method_list.extend(['PyRight']*4)
            bench_list.extend([method_name]*4)
            score_name.extend(['Precision', 'Recall', 'F1', 'WF1'])
            score_list.extend([pyright_precision, pyright_recall, pyright_f1, pyright_wf1])

            ylabels = ['Safe', 'Unsafe']
            xlabels = ['Passed', 'Failed']
            plt.figure(figsize=(10, 7))
            sns.heatmap(pyright_cm, annot=True, fmt='d', xticklabels=xlabels, yticklabels=ylabels, annot_kws={"size": 30})
            plt.xlabel('Ground Truth', size=30)
            plt.ylabel('Predicted', size=30)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.tight_layout()
            plt.savefig(f'cm_pyright.pdf')

            # # pyflakes only
            lint_precision, lint_recall, lint_f1, lint_wf1, lint_cm, lint_predicted = compute_cm(all_instances, eval_results, static_predicted_results_pyflakes)
            seeds.extend([index]*4)
            method_list.extend(['Pyflakes']*4)
            bench_list.extend([method_name]*4)
            score_name.extend(['Precision', 'Recall', 'F1', 'WF1'])
            score_list.extend([lint_precision, lint_recall, lint_f1, lint_wf1])

            # # pyright aggressive 
            # pyright_a_precision, pyright_a_recall, pyright_a_f1, pyright_a_wf1, pyright_a_cm, pyright_a_predicted = compute_cm(all_instances, eval_results, static_predicted_pyright_aggressive)
            # method_list.extend(['PyRight A'] * 4)
            # bench_list.extend([method]*4)
            # score_name.extend(['Precision', 'Recall', 'F1', 'WF1'])
            # score_list.extend([pyright_a_precision, pyright_a_recall, pyright_a_f1, pyright_a_wf1])
            # pyrights.append(len(static_predicted_pyright_aggressive))
            
            # LLM only
            unsafe_instances_all = []
            for pred in all_predicted_results:
                if 'Unsafe' in pred['conclusion']:
                    unsafe_instances_all.append(pred['instance_id'])
            safe_instances_all = []
            for pred in all_predicted_results:
                if 'Safe' in pred['conclusion']:
                    safe_instances_all.append(pred['instance_id'])
            all_predicted_instances = safe_instances_all + unsafe_instances_all
            llm_precision, llm_recall, llm_f1, llm_wf1, llm_cm, llm_predicted = compute_cm(all_instances, eval_results, unsafe_instances_all)
            seeds.extend([index]*4)
            method_list.extend(['LLM']*4)
            bench_list.extend([method_name]*4)
            score_name.extend(['Precision', 'Recall', 'F1', 'WF1'])
            score_list.extend([llm_precision, llm_recall, llm_f1, llm_wf1])
            ylabels = ['Safe', 'Unsafe']
            xlabels = ['Passed', 'Failed']
            plt.figure(figsize=(10, 7))
            sns.heatmap(llm_cm, annot=True, fmt='d', xticklabels=xlabels, yticklabels=ylabels, annot_kws={"size": 30})
            plt.xlabel('Ground Truth', size=30)
            plt.ylabel('Predicted', size=30)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.tight_layout()
            

            # REDO
            for tmp in ['REDO-PyRight', 'REDO-Pyflakes']:
            # for tmp in ['PyRight-REDO', 'pyflakes-REDO']:
                if tmp == 'REDO-Pyflakes':
                    filtered_predicted_results_tmp = filtered_predicted_results_pyflakes_list[index]
                    sa_predicted = static_predicted_results_pyflakes
                else:
                    filtered_predicted_results_tmp = filtered_predicted_results_pyright_list[index]
                    sa_predicted = static_predicted_results_pyright

                unsafe_instances = []
                for pred in filtered_predicted_results_tmp:
                    if 'Unsafe' in pred['conclusion']:
                        unsafe_instances.append(pred['instance_id'])
                safe_instances = []
                for pred in filtered_predicted_results_tmp:
                    if 'Safe' in pred['conclusion']:
                        safe_instances.append(pred['instance_id'])
                redo_precision, redo_recall, redo_f1, redo_wf1, redo_cm, redo_predicted = compute_cm(all_instances, eval_results, unsafe_instances, sa_predicted)
                seeds.extend([index]*4)
                method_list.extend([tmp]*4)
                bench_list.extend([method_name]*4)
                score_name.extend(['Precision', 'Recall', 'F1', 'WF1'])
                score_list.extend([redo_precision, redo_recall, redo_f1, redo_wf1])

                ylabels = ['Safe', 'Unsafe']
                xlabels = ['Passed', 'Failed']
                plt.figure(figsize=(10, 7))
                sns.heatmap(redo_cm, annot=True, fmt='d', xticklabels=xlabels, yticklabels=ylabels, annot_kws={"size": 30})
                plt.xlabel('Ground Truth', size=30)
                plt.ylabel('Predicted', size=30)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                plt.tight_layout()
                plt.savefig(f'cm_{method}.pdf')
    
    # print('Average false positive:', np.mean(pyrights))
    # breakpoint()
    arrays = [
        method_list,
        bench_list,
        score_name,
    ]
    # Plotting multiple groups
    index = pd.MultiIndex.from_arrays(arrays, names=('Method', 'Benchmark', 'Metric'))
    score_list = [item * 100 for item in score_list]
    df = pd.DataFrame({'Score': score_list, 'Seed': seeds}, index=index)
    # group by Seed and compute mean and std for each group
    df_mean = df.groupby(level=['Method', 'Benchmark', 'Metric']).mean().reset_index()
    df_std = df.groupby(level=['Method', 'Benchmark', 'Metric']).std().reset_index()
    df_mean['std'] = df_std['Score']
    # drop colum Seed of df_mean
    df_mean = df_mean.drop('Seed', axis=1)
    # df_std = df.groupby(level=['Seed', 'Method', 'Benchmark', 'Metric']).std().reset_index()
    # Define the desired order for the first level index
    # desired_order = ["Pyflakes", "PyRight", "LLM", 'pyflakes-REDO', 'PyRight-REDO',]
    df_latex = multi_index_to_latex_with_subscript_and_benchmark_order(
        df_mean,
        benchmark_order=['CodeStory', 'Demo', 'Marscode', 'Lingma', 'Droid', 'ACR',],
        metric_order=['Precision', 'Recall', 'F1'],
        method_order=['Pyflakes', 'PyRight', 'LLM', 'REDO-Pyflakes', 'REDO-PyRight'])
    # df_latex = df_to_latex(df, column_order=['codestory-mixed', 'Demo','lingma', 'droid',])
    # df_latex = df_to_latex(df, column_order=['marscode', 'lingma', 'droid',])
    # df_latex = df_latex.reindex(desired_order, level='first')
    # df_latex = df_to_latex(df, column_order=['codestory-mixed', 'autocoderover'])
    print(df_latex)
    # list 'Method' and 'Metric' as row index and 'Benchmark' as column

    # new_arrays = [] 
    # for item in arrays:
    #     item = [element for index, element in enumerate(item) if (index + 1) % 3 != 0]
    #     new_arrays.append(item)
    # score_list = [score for index, score in enumerate(score_list) if (index + 1) % 3 != 0]

    # index = pd.MultiIndex.from_arrays(new_arrays, names=('Method', 'Benchmark', 'Metric'))
    # df = pd.DataFrame({'Score': score_list}, index=index)

    # plot_clustered_multi_index_bar(df, ['pyflakes', 'PyRight-REDO'])
