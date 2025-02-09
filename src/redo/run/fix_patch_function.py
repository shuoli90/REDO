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
from typing import Any, Dict, List, Set, Tuple, Union
from difflib import unified_diff

OPUS = "claude-3-opus-20240229"
OPUS_BR = "anthropic.claude-3-opus-20240229-v1:0"
HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
SONNET_3_5 = "claude-3-5-sonnet-20240620"

SYSTEM_TEMPLATE = """
You are an experienced program analyzer who can fix the code according to previously detected runtime errors.
"""

EDIT_CODE_CHUNKS_TEMPLATE = """
Your task is to update the provided code files to prevent the previously detected runtime errors. You will \
be provided with relevant code chunks and identified errors.

Begin your response by providing a simple smoke test to test the updated code within <test></test> \
tags. The rest of your response should provide the updated code to prevent the runtime errors, matching \
the exact format of the provided <code> below, including the <code> and <file> tags, and the name \
and start_line attributes. If a code chunk does not need any modification, it can be omitted from \
your response. Each code chunk you update to solve the problem must be rewritten in full, \
including lines that are unchanged. The name and start_line XML attributes in your response should \
always match those in the code below exactly - do not change them. For example, if 100 lines of \
code are passed for a code chunk, but you only modify 5 lines, you must still include the full \
code chunk in your response with the original start_line attribute. If you are able to solve the \
problem, provide <outcome>Complete</outcome> in your response, otherwise provide \
<outcome>Incomplete</outcome>, along with brief feedback and next steps within \
<assessment></assessment> tags.

Below is a simple example of a valid response:
<example>
<smoke_test>
from path.to.file import combine_numbers
combine_numbers(123, 456)
</smoke_test>
As requested, in the updated code below, I've rewritten the full chunks provided, even those parts \
that remain unchanged, such as the load_file function.
<code>
<file name="path/to/file1.py" start_line="5">
import numpy as np
</file>
<file name="path/to/file1.py" start_line="23">
def load_file(path):
    with open(path, "r") as f:
        content = f.read()
    return content

def combine_numbers(a, b):
    return {{
        "sum": a + b,
        "difference": a - b,
        "product": a * b,
        "quotient": a / b,
        "geometric_mean": geometric_mean(a, b),
    }}
</file>
</code>
<outcome>Incomplete</outcome>
<assessment>
Although this patch adds a geometric mean calculation, it does not import the required function to \
the file. The next step is to import the `geometric_mean` function to path/to/file1.py
</assessment>
</example>

Here are the detected runtime errors:
<Rumetime errors>
{remaining_issues}
</Runtime errors>

Here are the code chunks:
<code>
{code}
</code>\
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

from collections import Counter
def majority_vote(lst):
    count = Counter(lst)
    majority_count = len(lst) // 2 + 1
    
    for elem, freq in count.items():
        if freq >= majority_count:
            return elem
    
    return None  # No majority element

import re
from collections import defaultdict

def extract_modified_files_and_line_numbers(diff):
    """
    Extract modified files and the corresponding line numbers of newly modified code from a git diff output.

    Args:
        diff (str): The git diff output as a string.

    Returns:
        dict: A dictionary with file names as keys and lists of modified line numbers as values.
    """
    file_line_numbers = defaultdict(list)
    current_file = None
    in_diff = False
    current_line_number = 0
    
    # Process the diff line by line
    for line in diff.splitlines():
        if line.startswith('diff --git'):
            # New file in the diff
            current_file = re.search(r'a\/(.*?)\s', line).group(1)
        elif line.startswith('@@'):
            # Extract the starting line number from the diff chunk header
            match = re.search(r'\+(\d+)', line)
            if match:
                hunk_start_line = int(match.group(1))
                current_line_number = hunk_start_line
            in_diff = True
        elif in_diff:
            if line.startswith('+') and not line.startswith('+++'):
                # New line added, store the line number
                file_line_numbers[current_file].append(current_line_number)
                current_line_number += 1
            elif line.startswith('-'):
                # A line that was removed, do not increase line count
                continue
            elif line.startswith(' '):
                # Context line, increase line count
                current_line_number += 1
            else:
                # This marks the end of the diff chunk
                in_diff = False

    return file_line_numbers
    
def augment_intervals(intervals, total_lines, context=2):
    augmented_intervals = []
    
    for start, end in intervals:
        # Calculate augmented start and end with context
        augmented_start = max(1, start - context)
        augmented_end = min(total_lines, end + context)
        augmented_intervals.append((augmented_start, augmented_end))
    
    return augmented_intervals

def merge_line_numbers(line_numbers):
    if not line_numbers:
        return []

    line_numbers.sort()
    
    ranges = []
    start = end = line_numbers[0]

    for num in line_numbers[1:]:
        if num == end + 1:
            end = num
        else:
            ranges.append((start, end))
            start = end = num

    ranges.append((start, end))
    return ranges

def extract_code_by_intervals(code_lines, intervals):
    extracted_chunks = []
    for start, end in intervals:
        chunk = code_lines[start-1:end]  # Convert to 0-based index
        extracted_chunks.append("\n".join(chunk))
    return extracted_chunks

def get_patch(old_file_contents: Dict[str, str], new_file_contents: Dict[str, str], context_lines: int = 3):
    """Generate a patch from old and new versions of a set of files."""
    patches = []
    for file_name, new_file_content in new_file_contents.items():
        # if "/test/" in file_name or "/tests/" in file_name or "/testing/" in file_name:
        #     continue
        old_file_content = old_file_contents.get(file_name)
        diff_gen = unified_diff(
            (old_file_content or "").splitlines(keepends=True),
            new_file_content.splitlines(keepends=True),
            fromfile=f"a/{file_name}",
            tofile=f"b/{file_name}",
            n=context_lines,
        )
        # Work around for https://bugs.python.org/issue2142
        diff_lines = []
        for line in diff_gen:
            if line.endswith("\n"):
                diff_lines.append(line)
            else:
                diff_lines.append(line + "\n")
                diff_lines.append("\\ No newline at end of file\n")
        diff = "".join(diff_lines)
        if diff.strip() != "":
            patches.append((file_name, diff))
    patch = "".join([p[1] for p in patches])
    patch = patch.rstrip("\n") + "\n"
    return patch

def generate_patch(original_scripts, modified_scripts, chunks, remaining_issues):

    code_chunks = []
    for file in chunks:
        for chunk, interval in chunks[file]:
            start = interval[0]
            code_chunks.append(
                f'<file name="{file}" start_line="{start}">\n{chunk}\n</file>'
            )

    prompt = EDIT_CODE_CHUNKS_TEMPLATE.format(
            remaining_issues=remaining_issues,
            code="\n\n".join(code_chunks),
        )
    # response = prompt_anthropic(prompt, "", SONNET_3_5)
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

    chunk_code = extract_tag_list("file", response, remove_leading_newline=True)
    chunk_files = extract_tag_attributes("file", "name", response)
    chunk_starts = extract_tag_attributes("file", "start_line", response)

    code_chunks_per_file: Dict = {}
    for code, file_name, start in zip(chunk_code, chunk_files, chunk_starts):
        if not start.isdigit():
            logger.warning("Malformed start line integer returned by LLM, skipping")
            continue
        # Add the chunk to the list of chunks for the file
        current_code_chunks_for_file = code_chunks_per_file.get(file_name, [])
        current_code_chunks_for_file.append((code, int(start)))
        code_chunks_per_file[file_name] = current_code_chunks_for_file

    new_code, old_code = {}, {}
    for file_name, code_chunks_starts in code_chunks_per_file.items():
        if file_name not in modified_scripts:
            continue

        # Get matching file (guaranteed by the checks above)
        original_code = original_scripts[file_name]
        old_code[file_name] = original_code
        original_code_lines = original_code.splitlines(keepends=True)

        code = modified_scripts[file_name]
        code_lines = code.splitlines(keepends=True)
        # Sort code chunks in decreasing order to apply to the code from the bottom up
        # This prevents earlier chunks from mangling line numbers for later chunks
        code_chunks_starts = sorted(code_chunks_starts, key=lambda x: -x[1])
        for new_chunk, start in code_chunks_starts:
            chunk_matches = [
                line_range
                for _, line_range in chunks[file_name]
                if line_range[0] == start
            ]

            if not chunk_matches:
                logger.error("LLM modified chunk start line, failing")
                raise ValueError("LLM modified chunk start line, failing")
            end = chunk_matches[0][1]

            updated_code: List[str] = []
            if start > 0:
                updated_code.extend(code_lines[:start-1])
            updated_code.extend(new_chunk.splitlines(keepends=True))
            # updated_code.extend(chunk_matched.splitlines(keepends=True))
            updated_code += '\n'
            updated_code.extend(code_lines[end:])
            code_lines = updated_code

        new_code[file_name] = "".join(code_lines)

    patch = get_patch(old_code, new_code, context_lines=10)
    return patch, response

import ast
def find_functions_and_classes(filepath):
    with open(filepath, "r") as file:
        tree = ast.parse(file.read(), filename=filepath)
    
    functions_and_classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start_lineno = node.lineno
            end_lineno = None
            for child in ast.walk(node):
                if hasattr(child, 'lineno'):
                    end_lineno = max(end_lineno or 0, child.lineno)
            functions_and_classes.append({
                "name": node.name,
                "type": "class" if isinstance(node, ast.ClassDef) else "function",
                "start_lineno": start_lineno,
                "end_lineno": end_lineno
            })
    
    return functions_and_classes

def extract_code_block(filepath, start_lineno, end_lineno):
    with open(filepath, "r") as file:
        lines = file.readlines()
        code_block = ''.join(lines[start_lineno-1:end_lineno])
    return code_block

def identify_code_location(filepath, line_ranges, item_type='function'):
    functions_and_classes = find_functions_and_classes(filepath)
    
    code_locations = []
    for start_line, end_line in line_ranges:
        relevant_items = []
        for item in functions_and_classes:
            if item['type'] == item_type and ((item["start_lineno"] <= start_line <= (item["end_lineno"] or float('inf'))) or \
               (item["start_lineno"] <= end_line <= (item["end_lineno"] or float('inf'))) or \
               (start_line <= item["start_lineno"] and end_line >= item["end_lineno"])):
                relevant_items.append(item)

        if relevant_items:
            largest_item = max(relevant_items, key=lambda x: x["end_lineno"] - x["start_lineno"])
            code_block = extract_code_block(filepath, largest_item["start_lineno"], largest_item["end_lineno"])
            code_locations.append({
                "line_range": (start_line, end_line),
                "location": {
                    "name": largest_item["name"],
                    "type": largest_item["type"],
                    "file": filepath,
                    "start_lineno": largest_item["start_lineno"],
                    "end_lineno": largest_item["end_lineno"],
                    "code_block": code_block
                }
            })
    return code_locations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='autocoderover')
    args = parser.parse_args()

    method = args.method
    # read in swe-bench dataset
    swebench_lite = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split='test')
    swebench_lite_df = pd.DataFrame.from_dict(swebench_lite, orient='columns')

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

    with open(f'predicted_{method}_results_pylint.json', 'r') as f:
        static_predicted_pylint = json.load(f)
    static_predicted_all_pylint = []
    for key in static_predicted_pylint:
        if 'predicted' in key:
            static_predicted_all_pylint.extend(static_predicted_pylint[key])
    static_predicted_results_pylint = list(set(static_predicted_all_pylint))

    with open(f'predicted_{method}_results_new.json', 'r') as f:
        static_predicted_pyright = json.load(f)
    static_predicted_all_sa = []
    for key in static_predicted_pyright:
        if 'predicted' in key:
            static_predicted_all_sa.extend(static_predicted_pyright[key])
    static_predicted_results_pyright = list(set(static_predicted_all_sa))

    if method == 'autocoderover':
        method_tmp = 'autocoderover-v20240620-gpt-4o-2024-05-13'
        file_path = f'runtime_detection_predicted_{method_tmp}_results_newpatch_singleshot.jsonl'
    else:
        file_path = f'runtime_detection_predicted_{method}_results_newpatch_singleshot.jsonl'
    predicted_results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            predicted_results.append(json.loads(line.strip()))
    
    modified_records = []
    for idx, ant_prediction in tqdm(enumerate(predicted_results), total=len(predicted_results)):
        if idx > 100:
            break
        instance_id = ant_prediction['instance_id']
        index = preds.index[preds['instance_id'] == instance_id].tolist()[0]
        pred = preds[preds['instance_id']==instance_id].iloc[0]
        record = swebench_lite_df[swebench_lite_df['instance_id']==instance_id].iloc[0]

        if ant_prediction['conclusion'] == 'Safe' and instance_id not in all_instances:
            continue
        
        # set up repo
        pred['repo'] = record['repo']
        pred['base_commit'] = record['base_commit']
        pred['patch'] = record['patch']
        repo_dir = setup_repo(pred)
        os.chdir(repo_dir)

        # retrieve modified files according to the patch
        file_line_numbers = extract_modified_files_and_line_numbers(pred['model_patch'])

        # retrieve original implementation
        original_scripts = {}
        for file in file_line_numbers:
            original_script = read_python_script(file)
            original_scripts[file] = original_script
        
        # apply original (previously problematic) patch; 
        patch = pred['model_patch']
        patch_path = repo_dir / 'predicted.patch'
        with open(patch_path, 'w') as f:
            f.write(patch)
        apply_patch(patch_path)

        # retrieve the modified lines
        # chunks = {}
        # modified_scripts = {}
        # for file in file_line_numbers:
        #     modified_script = read_python_script(file)
        #     intervals = merge_line_numbers(file_line_numbers[file])
        #     total_lines = len(modified_script.splitlines())
        #     augmented_intervals = augment_intervals(intervals, total_lines, context=15)
        #     extracted_code_chunks = extract_code_by_intervals(modified_script.splitlines(), augmented_intervals)
        #     chunks[file] = list(zip(extracted_code_chunks, augmented_intervals))
        #     modified_scripts[file] = modified_script
        
        chunks = {}
        modified_scripts = {}
        for file in file_line_numbers:
            modified_script = read_python_script(file)
            intervals = merge_line_numbers(file_line_numbers[file])
            total_lines = len(modified_script.splitlines())
            augmented_intervals = augment_intervals(intervals, total_lines, context=0)
            extracted_code_chunks = identify_code_location(
                filepath=file, 
                line_ranges=augmented_intervals)
            # extracted_code_chunks = extract_code_by_intervals(modified_script.splitlines(), augmented_intervals)
            # chunks[file] = list(zip(extracted_code_chunks, augmented_intervals))
            chunks[file] = [(chunk['location']['code_block'], (chunk['location']['start_lineno'], chunk['location']['end_lineno'])) for chunk in extracted_code_chunks]
            modified_scripts[file] = modified_script
        
        # revert modification
        revert_patch(patch_path)
        
        if instance_id in static_predicted_results_pyright:
            original_issues = static_predicted_pyright['original_issues'][index]
            modified_issues = static_predicted_pyright['modified_issues'][index]
            remaining_issues = [item for item in modified_issues if item not in original_issues]
        else:
            # retrieve previously detected errors
            remaining_issues = [item.strip() for item in ant_prediction['remaining_issues']]
        remaining_issues = "\n".join(remaining_issues)

        # generate and save new patch 
        new_patch, response = generate_patch(original_scripts, modified_scripts, chunks, remaining_issues)    
        if new_patch == '\n':
            continue
        
        record = {}
        record['model_patch'] = new_patch
        record['instance_id'] = instance_id
        record['model_name_or_path'] = f'{method}_modified_function'
        modified_records.append(record)

        if idx % 10 == 0:
            with open(f'/home/ubuntu/mnt/agent/amazon-Q/NGDEBirds/NGDEBirdsScienceTransforms/src/birds_transforms/examples/modified_predictions/modified_patches_{method}_function.jsonl', 'w') as f:
                for record in modified_records:
                    f.write(json.dumps(record) + '\n')
         
    
    with open(f'/home/ubuntu/mnt/agent/amazon-Q/NGDEBirds/NGDEBirdsScienceTransforms/src/birds_transforms/examples/modified_predictions/modified_patches_{method}_function.jsonl', 'w') as f:
        for record in modified_records:
            f.write(json.dumps(record) + '\n')

