import ast
import json
import subprocess
import os
import shutil
import astor
import jsonlines
import astunparse
from datasets import load_dataset

repo_to_top_folder = {
    "django/django": "django",
    "sphinx-doc/sphinx": "sphinx",
    "scikit-learn/scikit-learn": "scikit-learn",
    "sympy/sympy": "sympy",
    "pytest-dev/pytest": "pytest",
    "matplotlib/matplotlib": "matplotlib",
    "astropy/astropy": "astropy",
    "pydata/xarray": "xarray",
    "mwaskom/seaborn": "seaborn",
    "psf/requests": "requests",
    "pylint-dev/pylint": "pylint",
    "pallets/flask": "flask",
}


def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        data[result['instance_id']] = result['data']
    return data
        

class MethodAnalyzer(ast.NodeVisitor):
    def __init__(self, target_method_name):
        self.target_method_name = target_method_name
        self.body_start_line = None
        self.return_lines = []

    def visit_FunctionDef(self, node):
        if node.name == self.target_method_name:
            # Get the first statement in the body (if any) to determine body start
            if node.body:
                self.body_start_line = node.body[0].lineno
            
            # Traverse the body to find return statements
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Return):
                    self.return_lines.append(sub_node.lineno)
        # Continue traversal
        self.generic_visit(node)

class MethodBodyLocator(ast.NodeVisitor):
    def __init__(self):
        self.locations = {}
        self.class_stack = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        self._record_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._record_function(node)
        self.generic_visit(node)

    def _record_function(self, node):
        if node.body:
            first_stmt = node.body[0]
            full_name = ".".join(self.class_stack + [node.name]) if self.class_stack else node.name
            # self.locations.append({
            #     "function_name": full_name,
            #     "start_line": first_stmt.lineno
            # })
            self.locations[first_stmt.lineno] = full_name

def find_start_return(file_path, method_name):
    ## extract the start of method body and the location of return statements.
    source_code = open(file_path, 'r').read()
    tree = ast.parse(source_code)
    analyzer = MethodAnalyzer(method_name)
    analyzer.visit(tree)
    return analyzer.body_start_line, analyzer.return_lines

def find_leading_whitespace(line):
    stripped_line = line.lstrip('\t ')  # Remove leading whitespace
    leading_whitespace = line[:len(line) - len(stripped_line)]
    return leading_whitespace

def find_start_lines(file_path):
    source_code = open(file_path, 'r').read()
    tree = ast.parse(source_code)
    locator = MethodBodyLocator()
    locator.visit(tree)
    return locator.locations

def del_folder(folder_path):
    try:
    # Remove the folder and all its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been deleted.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied: Unable to delete '{folder_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def shallow_clone_commit(repo_name, repo_playground, commit):
    git_link = f"https://github.com/{repo_name}.git"
    cmd_path = "/home/changshu/CODEMIND/scripts/swebench/input_pipeline/shallow_clone.sh"
    try:
        print(f"shallow clone and checkout {repo_name}-{commit}")
        subprocess.run(["bash", cmd_path, git_link, commit, repo_playground])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def apply_patch(repo_path, diff_path):
    '''apply .diff to the origin code'''
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Apply patch in repository at {diff_path}...")
        subprocess.run(["git", "-C", repo_path, "apply", diff_path], check=True)
        print("Patch applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def git_diff(repo_path, diff_path):
    os.chdir(repo_path)
    subprocess.run(f'git diff > {diff_path}', shell=True)

def parse_diff_to_jsonl(diff_path, jsonlpath, instance_id, model_name_or_path):
    patch_text = open(diff_path, 'r').read()
    data = [
        {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": patch_text            
        }
    ]
    with jsonlines.open(jsonlpath, mode='w') as writer:
        writer.write_all(data)    

def run_swebench(instance_id, predictions_path, timeout):
    swebench_root = "/home/changshu/SWE-bench"
    predict_path = predictions_path
    run_id = "DEP"
    dataset = "princeton-nlp/SWE-bench"
    os.chdir(swebench_root)
    run_cmd = f"python -m swebench.harness.run_evaluation --predictions_path {predict_path} --max_worker 1 --instance_ids {instance_id} --run_id {run_id} --dataset_name {dataset} --timeout {timeout}"
    # print(run_cmd)
    subprocess.run(run_cmd, shell=True)



def init_playground(pid, ds):
    playground_path = "/home/changshu/CODEMIND/scripts/swebench/swebench_playground/dep"
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
if __name__ == "__main__":
    ## test
    # file_path = "/home/changshu/CODEMIND/scripts/swebench/swebench_playground/obj/sympy__sympy-16781/sympy/printing/dot.py"
    # fun_name = "dotnode"
    # print(find_start_return(file_path,fun_name))
    
    file_path = "/home/changshu/CODEMIND/scripts/swebench/swebench_playground/obj/sympy__sympy-16781/sympy/printing/fcode.py"
    r = find_start_lines(file_path)
    print(r)