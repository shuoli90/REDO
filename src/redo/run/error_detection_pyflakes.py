import os
os.environ["ANTHROPIC_API_KEY"] = YOUR_ANTRHOPIC_API_KEY
import git
import ast
import pandas as pd
import datasets
from pathlib import Path
import argparse
from typing import DefaultDict
from utils import (
    find_intervals, 
    identify_code_location, 
)
import shutil
import time
from pathlib import Path

from git import Repo
from loguru import logger
import subprocess
import builtins
import json
from tqdm import tqdm
from pyflakes.api import check
from pyflakes.reporter import Reporter
import sys
from io import StringIO

# Create an in-memory file-like object
output = StringIO()
error = StringIO()

# Initialize the Reporter
reporter = Reporter(output, error)


def run_pyflakes(code):
    # Run pyflakes on the code
    check(code, "<input>", reporter)

    # Retrieve the output and error messages
    output_str = output.getvalue()
    error_str = error.getvalue()
    return output_str, error_str

class FunctionAndAttributeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.attributes = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            self.functions.append(node.func.attr)
        elif isinstance(node.func, ast.Name):
            self.functions.append(node.func.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.attributes.append(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.attributes.append(node.id)
        self.generic_visit(node)

def extract_functions_and_attributes(code):
    tree = ast.parse(code)
    visitor = FunctionAndAttributeVisitor()
    visitor.visit(tree)
    return visitor.functions, visitor.attributes

def filter_builtins(functions):
    builtin_function_names = dir(builtins)
    return [func for func in functions if func not in builtin_function_names]


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

def pyright_analysis(file_path):
    command = ['pyright', file_path, '--outputjson']
    result = subprocess.run(command, capture_output=True, text=True)
    sa_output = eval(result.stdout)['generalDiagnostics']
    issues = list(set([issue['message'] for issue in sa_output]))
    return issues

def apply_patch(patch_path):
    command = ['git', 'apply', str(patch_path)]
    result = subprocess.run(command)

def revert_patch(patch_path):
    command = ['git', 'apply', '-R', str(patch_path)]
    result = subprocess.run(command)

def find_function_in_repo(repo_path, function_name):
    """Find if a function is defined anywhere in a Python repository."""
    function_locations = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=file_path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                                function_locations.append(file_path)
                    except SyntaxError:
                        continue

    return function_locations

def find_classes_with_attribute(repo_path, attribute_name):
    """Find classes in a Python repository that contain a specific attribute."""
    classes_with_attribute = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=file_path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                for class_node in node.body:
                                    if isinstance(class_node, ast.Assign):
                                        for target in class_node.targets:
                                            if isinstance(target, ast.Name) and target.id == attribute_name:
                                                classes_with_attribute.append((node.name, file_path))
                    except SyntaxError:
                        continue

    return classes_with_attribute

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_id', type=str, default="", help='instance id of the sample')
    parser.add_argument('--predictions_file', type=str, default='~/predictions/opus_func_margin.jsonl')
    parser.add_argument('-', '--golden', action='store_true')
    args = parser.parse_args()

    syntax_instances = []
    name_attribute_instances = []
    warning_instances = []
    original_issues_list = []
    modified_issue_list = []

    swebench_lite = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split='test')
    swebench_lite_df = pd.DataFrame.from_dict(swebench_lite, orient='columns')

    # read predictions
    preds = pd.read_json(args.predictions_file, lines=True)
    predicted_file = preds.iloc[0]['model_name_or_path']
    # predicted_file = preds.iloc[0]['model_pr']
    # iterate all rows in preds
    for idx, pred in tqdm(preds.iterrows(), total=len(preds)):
        # select the row in preds where whose instance id equals to instance_id
        # pred = preds[preds['instance_id'] == args.instance_id].iloc[0]
        # if idx > 2:
        #     break
        # if idx == 0:
        #     continue
        if args.instance_id != "":
            if pred['instance_id'] != args.instance_id:
                continue

        print('instance_id:', pred['instance_id'])
        index = swebench_lite_df[swebench_lite_df["instance_id"] == pred['instance_id']].index[0]
        input_sample = swebench_lite[int(index)]
        pred['repo'] = input_sample['repo']
        pred['base_commit'] = input_sample['base_commit']
        pred['patch'] = input_sample['patch']

        # setup repo
        repo_dir = setup_repo(input_sample)
        os.chdir(repo_dir)
        
        # identify predicted patch info
        if args.golden:
            patch = pred['patch']
            patch_path = repo_dir / 'golden.patch'
        else:
            patch = pred['model_patch']
            patch_path = repo_dir / 'predicted.patch'
        if patch is None:
            continue
        with open(patch_path, 'w') as f:
            f.write(patch)
        file_lines = extract_modified_lines(patch)
        related_files = [filepath for filepath in file_lines]
        line_ranges = [file_lines[file] for file in related_files]

        original_issues = []
        original_scripts = []
        for file in related_files:
            original_script = read_python_script(file)
            # try:
            output_str, error_str = run_pyflakes(original_script)
            original_issues.extend(output_str.splitlines())
            original_scripts.append(original_script)
            # except:
            #     continue

        # apply predicted patch
        modified_issues = []
        modified_scripts = []
        apply_patch(patch_path)
        for file in related_files:
            modified_script = read_python_script(file)
            # try:
            # then use pyright to check semantic errors
            output_str, error_str = run_pyflakes(modified_script)
            modified_issues.extend(output_str.splitlines())
            modified_scripts.append(modified_script)
            # except:
            #     continue
        
        original_issues_list.append(original_issues)
        modified_issue_list.append(modified_issues)

        # check if there are new issues in predicted_issues
        diffs = []
        for issue in modified_issues:
            if issue not in original_issues:
                diffs.append(issue)
        isSame = True if len(diffs) == 0 else False
        if isSame:
            print('No new issues found')
        else:
            name_attribute_instances.append(pred['instance_id'])
            print('predicted_name_attribute_instances', name_attribute_instances)
        remove_repo(repo_dir)
    print('finished')
    print('predicted syntax instances:', syntax_instances)
    print('predicted_name_attribute_instances', name_attribute_instances)
    results = {
        'predicted_syntax_instances': syntax_instances,
        'predicted_name_attribute_instances': name_attribute_instances,
        'warning_instances': warning_instances,
        'original_issues_lsit': original_issues_list,
        'modified_issues_list': modified_issue_list
    }
    # write results to a json file
    if args.golden:
        with open(f'/home/ubuntu/mnt/agent/amazon-Q/NGDEBirds/NGDEBirdsScienceTransforms/src/birds_transforms/examples/golden_results_pyflakes.json', 'w') as f:
            json.dump(results, f)
    else:
        with open(f'/home/ubuntu/mnt/agent/amazon-Q/NGDEBirds/NGDEBirdsScienceTransforms/src/birds_transforms/examples/predicted_{predicted_file}_results_pyflakes.json', 'w') as f:
            json.dump(results, f)



    


    