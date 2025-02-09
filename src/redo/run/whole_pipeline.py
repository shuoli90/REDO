import os
os.environ["ANTHROPIC_API_KEY"] = YOUR_ANTRHOPIC_API_KEY
import datasets
import pandas as pd
from pathlib import Path

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
import subprocess

from git import Repo
import argparse
from tqdm import tqdm
import json
import pandas as pd
from collections import Counter
import time
import re

OPUS = "claude-3-opus-20240229"
OPUS_BR = "anthropic.claude-3-opus-20240229-v1:0"
HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
SONNET_3_5 = "claude-3-5-sonnet-20240620"


SYSTEM_TEMPLATE = """
You are an experienced program analyzer who can identify potential runtime errors without running the programs, \
and also prune unlikely ones.
"""

TEMPLATE = """
A modification patch is proposed to resolve an issue with the current github repo. \
This modification might introduce runtime errors that cannot be captured by static analysis tools. \
Your task is to check whether such runtime errors exist. Typical runtime errors include TypeError, ValueError, AttributeError, and IndexError.

First, you are provided with the problem statement, which describes the issue and hints on how the modification patch will be tested. \
The problem statement is as follows:
<Problem Statement>
{problem_statement}
</Problem Statement>

Then, you will be provided with the original implementation of python scripts containing modified functions:
<Original Implementation>
{original_implementation}
</Original Implementation>

Finaly, the modification patch is given below:
<Modificatoin Patch>
{modification_patch}
</Modification Patch>

First, please check if there are potential runtime errors, please list their error type and reasoning in the <Runtime Errors></Runtime Errors> section, with the format [ErrorType]:[Reasoning]. \
If there are no potential runtime errors, please return 'Safe'; otherwise, please return 'Unsafe'. The conclusion should be wrapped by <Conclusion></Conclusion>.
"""

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

# def majority_vote(lst):
#     count = Counter(lst)
#     majority_count = len(lst) // 2 + 1
    
#     for elem, freq in count.items():
#         if freq >= majority_count:
#             return elem
    
#     return None  # No majority element

# def extract_potential_runtime_errors(markdown_script):
#     # Regular expression to match the "Potential Runtime Errors" section
#     section_regex = r"### Remaining Errors\s*([\s\S]*?)(?=\n### |\Z)"
    
#     # Search for the "Potential Runtime Errors" section
#     section_match = re.search(section_regex, markdown_script)
    
#     if section_match:
#         section_content = section_match.group(1).strip()
        
#         # Find all bullet points in the section
#         bullet_points = re.findall(r"(\d+\..*?:\n(?:\s*-\s.*\n?)*)", section_content)
        
#         return section_content, bullet_points
#     else:
#         return None, []

def pyright_analysis(file_path):
    command = ['pyright', file_path, '--outputjson']
    result = subprocess.run(command, capture_output=True, text=True)
    sa_output = eval(result.stdout)['generalDiagnostics']
    issues = list(set([issue['message'] for issue in sa_output]))
    return issues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_id', type=str, default="", help='instance id of the sample')
    parser.add_argument('--method', type=str, default='autocoderover')
    parser.add_argument('-g', '--golden', action='store_true')
    parser.add_argument('-a', '--anthropic', action='store_true', help="Use Anthropic models")
    args = parser.parse_args()

    if args.method == 'autocoderover':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240621_autocoderover-v20240620/all_preds.jsonl"
    elif args.method == 'aider':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240523_aider/all_preds.jsonl"
    elif args.method == 'codestory-mixed':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240702_codestory_aide_mixed/all_preds.jsonl"
    elif args.method == 'Demo':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240627_abanteai_mentatbot_gpt4o/all_preds.jsonl"
    elif args.method == 'droid':
        prediction_path = "/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240617_factory_code_droid/all_preds.jsonl"
    elif args.method == 'lingma':
        prediction_path = '/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240622_Lingma_Agent/all_preds.jsonl'
    elif args.method == 'marscode':
        prediction_path = '/home/ubuntu/mnt/agent/experiments/evaluation/lite/20240723_marscode-agent-dev/all_preds.jsonl'
    else:
        raise ValueError("Invalid method")

    results = []
    # read predictions
    preds = pd.read_json(prediction_path, lines=True)
    predicted_file = preds.iloc[0]['model_name_or_path']

    # read in swe-bench dataset
    swebench_lite = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split='test')
    swebench_lite_df = pd.DataFrame.from_dict(swebench_lite, orient='columns')

    for idx, pred in tqdm(preds.iterrows(), total=len(preds)):

        if args.instance_id != "" and args.instance_id != pred['instance_id']:
            continue

        print('instance_id:', pred['instance_id'])
        index = swebench_lite_df[swebench_lite_df["instance_id"] == pred['instance_id']].index[0]
        input_sample = swebench_lite[int(index)]
        pred['repo'] = input_sample['repo']
        pred['base_commit'] = input_sample['base_commit']
        pred['patch'] = input_sample['patch']
        
        repo_dir = setup_repo(pred)
        os.chdir(repo_dir)

        result = {}
        if args.golden:
            patch = pred['patch']
        else:
            patch = pred['model_patch']
        patch_path = 'patch.patch'
        if patch is None:
            continue
        with open(patch_path, 'w') as f:
            f.write(patch)

        file_lines = extract_modified_lines(patch)
        # related files
        related_files = [filepath for filepath in file_lines]
        line_ranges = [file_lines[file] for file in related_files]

        # static analysis check 
        original_issues = []
        original_scripts = []
        for file in related_files:
            try:
                original_script = read_python_script(file)
                original_issues.extend(pyright_analysis(file))
                original_scripts.append(original_script)
            except:
                continue

        # apply predicted patch
        modified_issues = []
        modified_scripts = []
        apply_patch(patch_path)
        for file in related_files:
            modified_script = read_python_script(file)
            try:
                # then use pyright to check semantic errors
                modified_issues.extend(pyright_analysis(file))
                modified_scripts.append(modified_script)
            except:
                continue
        original_issues_list.append(original_issues)
        modified_issues_list.append(modified_issues)
        
        # check if there are new issues in predicted_issues
        diffs = []
        for issue in modified_issues:
            if issue not in original_issues:
                diffs.append(issue)
        isSame = True if len(diffs) == 0 else False
        if isSame:
            print('No new issues found by static analysis')
        else:
            print('Found the following error by static analysis')
            print(diffs)
            continue

        conclusions = []
        scripts = []
        for file, ranges in zip(related_files, line_ranges):
            # if file is not a python script, skip
            if not file.lower().endswith('.py'):
                continue
            related_file = repo_dir / file
            locs = identify_code_location(related_file, ranges)
            # script = " ".join([loc['location']['code_block'] for loc in locs])
            script = read_python_script(related_file)
            scripts.append(f"filename: {file}" + '/n' + script)
        script = "/n/n/n".join(scripts)
        prompt = TEMPLATE.format(
            problem_statement=input_sample['problem_statement'],
            original_implementation=script,
            modification_patch=patch
        )

        max_retries = 10
        delay = 20
        for attempt in range(max_retries):
            try:
                response = prompt_anthropic(
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

        conclusion = extract_tag_list("Conclusion", response)
        if len(conclusion) == 0:
            conclusion = "Safe"
        else:
            conclusion = conclusion[0].strip()
        remaining_issues = extract_tag_list("Runtime Errors", response)
        if len(remaining_issues) == 0:
            remaining_issues = "None"
        
        print('*' * 20)
        print('LLM conclusion', conlusion)