from utils import clone_repo, checkout_commit, repo_to_top_folder, apply_patch, install_dependency, \
    load_stripped_files, get_function_line_ranges, del_folder, shallow_clone_commit
from datasets import load_dataset
import json
from tqdm import tqdm
import os
'''
How to locate code from swebench:
1. clone the project (using the infotmation from the dataset).
2. check out the buggy version.
3. apply the patch to the buggy version.
4. locate the line number of the patched function.
'''
def load_patch_info(instance_id):
    path_json = "./stats/patches.json"
    patch_data = json.load(open(path_json, 'r'))[instance_id]
    result = []
    for p in patch_data["patch"]:
        d = {
            'path': p['file'],
            'lines':[]
        }
        for h in p['hunks']:
            lines = []
            for i in h['changes']:
                if i['type'] == 'add':
                    lines.append(i['content'])
            d['lines'].append(lines)
        result.append(d)
    return result

def find_consecutive_lines_start(lines, lines_to_match):
    """
    Find the starting line number of n consecutive lines that match the given lines.

    Parameters:
        file_path (str): Path to the file.
        lines_to_match (list): List of consecutive lines to locate.

    Returns:
        int: The starting line number of the first match, or -1 if no match is found.
    """
    n = len(lines_to_match)


    for i in range(len(lines) - n + 1):
        if all(lines[j].strip() == lines_to_match[j - i] for j in range(i, i + n)):
            return i + 1  # Line numbers are 1-based
    return -1  # No match found


def extract_start_lines(instance_id, repo_path):
    patch_data = load_patch_info(instance_id)
    ## iterate all the modified files:
    results = []
    for p in patch_data:
        starting_lines = []
        if not p['path']:
            continue
        file_path = f"{repo_path}/{p['path']}"
        if not os.path.exists(file_path):
            continue
        stripped_lines = load_stripped_files(file_path)
        for i in p["lines"]:
            lineno = find_consecutive_lines_start(stripped_lines, i)
            starting_lines.append(lineno)
        results.append({
            'file_path': file_path,
            'starting_lines': starting_lines
        })
    return results
        
        

if __name__ == "__main__":
    project_name = "astropy"
    repo_playground = "/swebench_playground/fixed" ## need to change to dir under your dir
    path_test_patch = f"{repo_playground}/test_patch.txt"
    path_code_patch = f"{repo_playground}/patch.txt"
    summary = {}
    ds = load_dataset("princeton-nlp/SWE-bench", split="test",  cache_dir="/home/shared/huggingface")
    
    summary_path = f"stats/function_locations/{project_name}-fix.jsonl"
    
    candidate_list = []
    count = 0
    for i in range(len(ds)):
        instance_id = ds[i]['instance_id']
        if project_name in instance_id:
            candidate_list.append(ds[i])
          
    for i in tqdm(candidate_list):
        count += 1
        # if count < 340:
        #     continue
        instance_id = i['instance_id']
        results = []
        repo_name = i['repo']
        repo_path = f"{repo_playground}/{repo_to_top_folder[repo_name]}"
        commit_id = i['base_commit']
        patch = i['patch']
        test_patch = i['test_patch']

        # clone_repo(repo_name, repo_playground)
        # checkout_commit(repo_path, commit_id)
        shallow_clone_commit(repo_name, repo_path, commit_id)
        with open(path_test_patch, 'w') as wr:
            wr.write(test_patch)
        with open(path_code_patch, 'w') as wr:
            wr.write(patch)
            
        apply_patch(repo_path, path_test_patch)
        apply_patch(repo_path, path_code_patch)
        # install_dependency(repo_name, repo_path)
        
        ## results after parsing the text
        text_results = extract_start_lines(instance_id, repo_path)
        print(text_results)
        for t in text_results:
            file_path = t['file_path']
            d = {
                'file_path': file_path,
                'functions':[]
            }
            starting_lines = t['starting_lines']
            defined_functions = get_function_line_ranges(file_path)
            for s in starting_lines:
                # print(defined_functions)
                for fun in defined_functions:
                    if s >= fun['start_line'] and s <= fun['end_line']:
                        d['functions'].append(
                            {
                                'name': fun['name'],
                                'start_line': fun['start_line'],
                                'end_line': fun['end_line']
                            }
                        )
            results.append(d)
        data = {"instance_id": instance_id, "data": results}
        with open(summary_path, "a+") as jsonl_file:
            jsonl_file.write(json.dumps(data)+"\n")
        # del_folder(repo_path)
        
