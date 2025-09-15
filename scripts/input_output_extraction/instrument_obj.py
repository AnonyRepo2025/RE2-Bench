from datasets import load_dataset
import os
import argparse
from tqdm import tqdm
from utils import del_folder, shallow_clone_commit, apply_patch, git_diff, parse_diff_to_jsonl, run_swebench_input_output,\
    load_jsonl, find_last_import_node, find_leading_whitespace, find_future_import_line
'''
Load the data from Huggingface, which contains commint_id, repo name for each problem.
Clone the repo to a local dir, then checkout to that commit.
'''
def init_playground(pid):
    playground_path = "../swebench_playground/obj"
    ds = load_dataset("princeton-nlp/SWE-bench", split="test",  cache_dir="/home/shared/huggingface")
    for i in range(len(ds)):
        instance_id = ds[i]['instance_id']
        if pid == instance_id:
            meta_data = ds[i]
    commit_id = meta_data['base_commit']
    patch = meta_data['patch']
    test_patch = meta_data['test_patch']
    repo_name = meta_data['repo']
    repo_path = f"{playground_path}/{pid}"
    path_test_patch = f"{playground_path}/test_patch.txt"
    path_code_patch = f"{playground_path}/patch.txt"
    
    if os.path.exists(repo_path):
        del_folder(repo_path)
        
    
    shallow_clone_commit(repo_name, repo_path, commit_id)
    with open(path_test_patch, 'w') as wr:
        wr.write(test_patch)
    with open(path_code_patch, 'w') as wr:
        wr.write(patch)
    apply_patch(repo_path, path_code_patch)
    apply_patch(repo_path, path_test_patch)


'''
After the code instrumentation, use git diff to get the diff.
Parse the diff to jsonl file, which is required by the swebench pipeline.
Run swebench, using the jsonl file.
The decorator in the instrumented code has some print statements, which will print some text to the log produced by swebench pipeline.
'''
def create_diff(pid, timeout):
    playground_path = "../swebench_playground/obj"
    git_diff_base = "../swebench_playground/obj/diff-base"
    jsonl_base = "../swebench_playground/obj/jsonl-base/"
    diff_path = f"{git_diff_base}/{pid}.txt"
    jsonl_path = f"{jsonl_base}/{pid}.jsonl"
    repo_path = f"{playground_path}/{pid}"
    git_diff(repo_path, diff_path)
    parse_diff_to_jsonl(diff_path, jsonl_path, pid, 'playground')
    del_folder(repo_path)
    run_swebench_input_output(pid, jsonl_path, timeout)
    print(f"{playground_path}/{pid}/test_output.txt")

def load_inspect_code():
    p = "./input_output_extraction/inspect_code.py"
    inspect_code = open(p, 'r').read()
    return inspect_code


def main(dataset, timeout):
    inspect_code = load_inspect_code()
    location_path = f"../stats/function_locations/{dataset}-fix.jsonl"
    base_name = f"/home/changshu/CODEMIND/scripts/swebench/swebench_playground/fixed/{dataset}"
    playground_base = "../swebench_playground/obj"
    data = load_jsonl(location_path)
    for d in tqdm(data):
        log_dir = f"/home/changshu/SWE-bench/logs/run_evaluation/OBJ_PLAYGROUND/playground/{d}" ## this path is related to the location of the swebench project
        if os.path.exists(log_dir):
            continue
        init_playground(d) ##initiate the playground
        playground_path = os.path.join(playground_base, d)
        flag = False
        ## For each modified function add a decorator.
        for file in data[d]:
            inst_lines = []
            file_path = file['file_path'].replace(base_name,playground_path)
            last_statement_lno = int(find_future_import_line(file_path))
            fun_names = [i['name'] for i in file['functions']]
            fun_lnos = [i['start_line'] for i in file['functions']]
            if fun_lnos:
                inspect_code_updated = inspect_code.replace("FILE_NAME", file_path)
                flag = True
                lines = open(file_path, 'r').read().split("\n")
                for lineno, line in enumerate(lines, start=1):
                    if lineno == last_statement_lno:
                        inst_lines.append(line)
                        inst_lines.append(inspect_code_updated)
                    elif lineno in fun_lnos:
                        leading_whitespce = find_leading_whitespace(line)
                        curidx = len(inst_lines)-1
                        while(inst_lines[curidx].startswith(leading_whitespce+"@")):
                            if "@property" in inst_lines[curidx] or "@staticmethod" in inst_lines[curidx] or "@classmethod" in inst_lines[curidx]:
                                break
                            curidx-=1
                        inst_lines.insert(curidx+1, leading_whitespce + "@inspect_code")
                        
                        inst_lines.append(line)
                    else:
                        inst_lines.append(line)
                if last_statement_lno == -1:
                    inst_lines.insert(0, inspect_code_updated)
                inst_code = "\n".join(inst_lines)
                with open(file_path, 'w') as wr:
                    wr.write(inst_code)
        if flag:
            create_diff(d,timeout)
                    
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--project")
    parser.add_argument('-t', "--timeout", default=1800)
    args = parser.parse_args()
    project = args.project
    timeout = args.timeout
    main(project, timeout)    
    
