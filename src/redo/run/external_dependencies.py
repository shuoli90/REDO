import os
os.environ["ANTHROPIC_API_KEY"] = YOUR_ANTRHOPIC_API_KEY
import datasets
import pandas as pd
from pathlib import Path
import subprocess

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

from git import Repo
import argparse
from tqdm import tqdm
import json
import argparse
from collections import defaultdict
import numpy as np

def extract_modified_lines(patch):
    file_lines = DefaultDict(list)
    # file_lines = {}
    for diff in patch.split("--- a/"):
        if not diff.strip():
            continue
        diff_file = diff.splitlines()[0]
        for hunk in diff.split("@@ -")[1:]:
            if not hunk.strip():
                continue
            try:
                start_line, num_lines = hunk.split(" ")[0].split(",")
            except Exception as e:
                print("Failed to extract from hunk in patch")
                continue
            start_line, num_lines = int(start_line), int(num_lines)
            offset = 0
            previous_line_type = None
            for line in hunk.splitlines()[1:]:
                if line.startswith("-"):
                    file_lines[diff_file].append(start_line + offset)
                    # file_lines.add(f"{diff_file} | line number {start_line + offset}")
                    offset += 1
                    previous_line_type = "remove"
                elif line.startswith(" "):
                    offset += 1
                    previous_line_type = None
                elif line.startswith("+") and previous_line_type is None:
                    # file_lines.add(f"{diff_file} | line number {start_line + offset - 1}")
                    file_lines[diff_file].append(start_line + offset - 1)
                    previous_line_type = "add"

    for diff_file in file_lines:
        file_lines[diff_file] = find_intervals(file_lines[diff_file])

    return file_lines

# setup repo
def setup_repo(x, repo_root='~'):
    """Checks out the specified commit of a repository for the given input.

    Clones the repository if not already present and checks out the specified commit. A
    temporary copy of the repository at the specified commit is prepared for further use.

    Args:
        x: Input data containing 'repo' (GitHub repo path) and 'base_commit' (commit hash).

    Returns:
        The input data augmented with 'repo_dir' (temporary directory of the repo) and
            'repo_url'.
    """
    repo_url = f"https://github.com/{x['repo']}.git"

    # Create base repo directory if it doesn't exist
    repo_root = (
        Path(repo_root).expanduser().resolve() if repo_root else Path.home() / "repos"
    )
    base_repo_dir = repo_root / x["repo"].split("/")[-1]
    if not base_repo_dir.is_dir() or not (base_repo_dir / ".git").exists():
        if base_repo_dir.exists():
            shutil.rmtree(base_repo_dir)
        base_repo_dir.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(repo_url, base_repo_dir).git.checkout(x["base_commit"])

    # Copy base repo to temporary directory
    repo_dir = Path("/tmp") / str(time.time()) / x["repo"].split("/")[-1]
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_repo_dir, repo_dir, dirs_exist_ok=True)

    # Checkout required commit
    repo = Repo(repo_dir)
    git = repo.git
    git.fetch("--all")
    git.reset("--hard")
    git.clean("-f", "-d")
    git.checkout(x["base_commit"])
    shutil.rmtree(repo_dir / ".git")
    return repo_dir


def remove_repo(repo_dir):
    if repo_dir.is_dir():
        shutil.rmtree(repo_dir, ignore_errors=False)

def read_python_script(file_path):
    try:
        with open(file_path, 'r') as file:
            script_body = file.read()
        return script_body
    except FileNotFoundError:
        return "File not found."
    except IOError:
        return "An error occurred while reading the file."

def apply_patch(patch_path):
    command = ['git', 'apply', str(patch_path)]
    result = subprocess.run(command)

def revert_patch(patch_path):
    command = ['git', 'apply', '-R', str(patch_path)]
    result = subprocess.run(command)

def get_dependency(target_file, list_of_files):
    subprocess_arg_list = []
    subprocess_arg_list.append("code2flow")
    subprocess_arg_list.extend(list_of_files)
    subprocess_arg_list.append("-o")
    target_file_name = target_file.split("/")[-1]
    target_file_name = target_file_name.split(".")[0]
    subprocess_arg_list.append(f"./tmp_code2flow_dir/{target_file_name}.json")

    directory_path = f"./tmp_code2flow_dir"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    try:
        sub_result = subprocess.run(
            subprocess_arg_list,
            capture_output=True,
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        exit(1)

    with open(f"./tmp_code2flow_dir/{target_file_name}.json", "r") as f:
        graph_json = json.load(f)['graph']
    file_to_node = {}
    node_to_file = {}
    for node in graph_json['nodes']:
        file_name = graph_json['nodes'][node]['name'].split("::")[0]
        file_to_node[file_name] = node
        node_to_file[node] = file_name
    node_to_node_graph = defaultdict(set)
    for edge in graph_json['edges']:
        node_to_node_graph[edge['source']].add(edge['target'])
    
    set_of_dependent_files = set()
    for node in node_to_node_graph:
        if str(node_to_file[node]) == target_file_name:
            for outward_node in node_to_node_graph[node]:
                if str(node_to_file[outward_node]) != target_file_name:
                    set_of_dependent_files.add(str(node_to_file[outward_node]))
    
    return len(set_of_dependent_files)
    
#get_dependency("core.py", [])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='auto')
    args = parser.parse_args()

    # read in swe-bench dataset
    swebench_lite = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split='test')
    swebench_lite_df = pd.DataFrame.from_dict(swebench_lite, orient='columns')

    if args.method == 'auto':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240621_autocoderover-v20240620/all_preds.jsonl"
    elif args.method == 'aider':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240523_aider/all_preds.jsonl"
    elif args.method == 'codestory-mixed':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240702_codestory_aide_mixed/all_preds.jsonl"
    elif args.method == 'Demo':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240627_abanteai_mentatbot_gpt4o/all_preds.jsonl"
    elif args.method == 'droid':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240617_factory_code_droid/all_preds.jsonl"
    elif args.method == 'opus_func_margin':
        prediction_path = "~/predictions/opus_func_margin.jsonl"
    elif args.method == 'lingma':
        prediction_path = '/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240622_Lingma_Agent/all_preds.jsonl'
    else:
        raise ValueError("Invalid method")

    with open(prediction_path, 'r') as f:
        preds = [json.loads(line) for line in f]

    num_dependencies = []
    for index, pred in tqdm(enumerate(preds), total=len(preds)):
        # if index > 10:
        #     break
        instance_id = pred['instance_id']
        index = swebench_lite_df[swebench_lite_df["instance_id"] == pred['instance_id']].index[0]
        input_sample = swebench_lite[int(index)]
        pred['repo'] = input_sample['repo']
        pred['base_commit'] = input_sample['base_commit']
        pred['patch'] = input_sample['patch']
        repo_dir = setup_repo(pred)
        os.chdir(repo_dir)
        result = {}
        patch = pred['model_patch']
        file_lines = extract_modified_lines(patch)
        patch_path = 'patch.patch'
        if patch is None:
            continue
        with open(patch_path, 'w') as f:
            f.write(patch)
        # related files
        related_files = [filepath for filepath in file_lines]
        line_ranges = [file_lines[file] for file in related_files]
        tmp = {'instance_id': pred['instance_id']}
        nums = []
        for file, ranges in zip(related_files, line_ranges):
            related_file = repo_dir / file
            file_folder = "/".join(str(related_file).split('/')[:-2])
            # Run the find command and capture the output
            result = subprocess.run(['find', file_folder, '-type', 'f', '-name', f'*.py'], capture_output=True, text=True)
            python_files = result.stdout.splitlines()
            try:
                num = get_dependency(str(related_file), python_files)
                nums.append(num)
            except:
                continue
                breakpoint()
        tmp['num_dependences'] = nums
        num_dependencies.append(tmp)
        if index % 10 == 0:
            with open(f'/home/ubuntu/mnt/agent/amazon-Q/NGDEBirds/NGDEBirdsScienceTransforms/src/birds_transforms/examples/external/external_dependencies_{args.method}.jsonl', 'w') as f:
                for result in num_dependencies:
                    f.write(json.dumps(result) + '\n')
    
    with open(f'/home/ubuntu/mnt/agent/amazon-Q/NGDEBirds/NGDEBirdsScienceTransforms/src/birds_transforms/examples/external/external_dependencies_{args.method}.jsonl', 'w') as f:
        for result in num_dependencies:
            f.write(json.dumps(result) + '\n')