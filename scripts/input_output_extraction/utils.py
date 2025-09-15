import subprocess
import os
import ast
import shutil
import astor
import jsonlines
import astunparse
import json

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


inspect_code="""

import inspect
def log_method(func):
    def wrapper(*args, **kwargs):
        data = {"name": func.__name__}
        args_names = inspect.getfullargspec(func).args
        if len(args) > 0 and hasattr(args[0], '__dict__') and args_names[0] == 'self':
            instance = args[0]
            instance_attrs = {k: str(v) for k, v in vars(instance).items()}
            data["self"] = instance_attrs
        
        if len(args_names) > 0:
            if args_names[0] == 'self':
                other_args = dict(zip(args_names[1:], args[1:]))
            else:
                other_args = dict(zip(args_names, args))
        else:
            other_args = {}
        
        other_args_cleaned = {}
        for k in other_args:
            other_args_cleaned[k] = str(other_args[k])
        data["args"] = other_args_cleaned
        cleaned_kwargs = {}
        if kwargs:
            for k in kwargs:
                cleaned_kwargs[k] = str(kwargs[k])
            data["kwargs"] = cleaned_kwargs
        print("[CALL]")
        result = func(*args, **kwargs)
        data["return"] = result
        package_name = "PACKAGE_NAME"
        print("@[DATA]@", package_name,"[SEP]",data,"[/SEP]")
        return result
    return wrapper

"""
def clone_repo(repo_name, repo_playground):
    try:

        print(
            f"Cloning repository from https://github.com/{repo_name}.git to {repo_playground}/{repo_to_top_folder[repo_name]}..."
        )
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                f"{repo_playground}/{repo_to_top_folder[repo_name]}",
            ],
            check=True,
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def checkout_commit(repo_path, patch_path):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param path_diff: path to git diff
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {patch_path} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", patch_path], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


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

def install_dependency(repo_name, repo_path):
    os.chdir(repo_path)
    ## navigate into the project directory
    if repo_name == "django/django":
        subprocess.run(["pip", "install", "-e", "."])
    if repo_name == "matplotlib/matplotlib":
        subprocess.run(["pip", "install", "-e", "."])
    if repo_name == "scikit-learn/scikit-learn":
        subprocess.run(["pip", "install", "-e", "."])

def load_stripped_files(file_path):
    stripped_lines = []
    lines = open(file_path, 'r').readlines()
    for line in lines:
        stripped_lines.append(line.strip())
    return stripped_lines 

def find_leading_whitespace(line):
    stripped_line = line.lstrip('\t ')  # Remove leading whitespace
    leading_whitespace = line[:len(line) - len(stripped_line)]
    return leading_whitespace

def get_function_line_ranges(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            start_line = node.lineno
            end_line = max(
                [child.lineno for child in ast.walk(node) if hasattr(child, "lineno")],
                default=start_line
            )
            functions.append({"name": function_name, "start_line": start_line, "end_line": end_line})
    return functions

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

def insert_inspect_code(pkg_name):
    code_to_insert = """
import inspect

def log_method(func):
    def wrapper(*args, **kwargs):
        data = {"name": func.__name__}
        args_names = inspect.getfullargspec(func).args
        if len(args) > 0 and hasattr(args[0], '__dict__') and args_names[0] == 'self':
            instance = args[0]
            instance_attrs = {k: str(v) for k, v in vars(instance).items()}
            data["self"] = instance_attrs
        
        if len(args_names) > 0:
            if args_names[0] == 'self':
                other_args = dict(zip(args_names[1:], args[1:]))
            else:
                other_args = dict(zip(args_names, args))
        else:
            other_args = {}
        
        other_args_cleaned = {}
        for k in other_args:
            other_args_cleaned[k] = str(other_args[k])
        data["args"] = other_args_cleaned
        cleaned_kwargs = {}
        if kwargs:
            for k in kwargs:
                cleaned_kwargs[k] = str(kwargs[k])
            data["kwargs"] = cleaned_kwargs
        print("[CALL]")
        result = func(*args, **kwargs)
        data["return"] = result
        package_name = "PACKAGE_NAME"
        print("@[DATA]@", package_name,"[SEP]",data,"[/SEP]")
        return result
    return wrapper
    """
    code_to_insert = code_to_insert.replace("PACKAGE_NAME", pkg_name)
    return code_to_insert
    # insert_after_imports(file_path, code_to_insert)
    # code = open(file_path, 'r').read()
    # new_code = code_to_insert + "\n" + code
    # with open(file_path, 'w') as wr:
    #     wr.write(new_code)


def find_last_import_node(file_path):
    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    last_import = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import = node

    if last_import:
        # Get the line number where the import ends
        end_lineno = getattr(last_import, 'end_lineno', last_import.lineno)
        return end_lineno
    else:
        return -1


def find_future_import_line(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=file_path)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                return node.lineno
    except SyntaxError:
        pass

    return -1

def insert_after_imports(filename, new_code):
    with open(filename, "r", encoding="utf-8") as f:
        source_code = f.read()
    tree = ast.parse(source_code)

    last_import_index = -1
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_index = index
    new_code_ast = ast.parse(new_code).body
    if last_import_index != -1:
        tree.body[last_import_index + 1:last_import_index + 1] = new_code_ast
    else:
        tree.body = new_code_ast + tree.body
    new_source_code = astunparse.unparse(tree)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(new_source_code)

class FunctionDecoratorAdder(ast.NodeTransformer):
    def __init__(self, functions, decorator_name):
        self.functions = functions
        self.decorator_name = decorator_name

    def visit_FunctionDef(self, node):
        # Check if the function matches the name and line number
        for func in self.functions:
            if node.name == func["name"] and node.lineno == func["start_line"]:
                # Add the decorator if it doesn't already exist
                if not any(
                    isinstance(decorator, ast.Name) and decorator.id ==self.decorator_name
                    for decorator in node.decorator_list
                ):
                    decorator = ast.Name(id=self.decorator_name, ctx=ast.Load())
                    node.decorator_list.insert(0, decorator)
                break
        return node


class FunctionDecoratorAdder_CES(ast.NodeTransformer):
    """add decorators for CES, no need to consider the line number"""
    def __init__(self, functions, decorator_name):
        self.functions = functions
        self.decorator_name = decorator_name

    def visit_FunctionDef(self, node):
        # Check if the function matches the name and line number
        for func in self.functions:
            if node.name == func:
                # Add the decorator if it doesn't already exist
                if not any(
                    isinstance(decorator, ast.Name) and decorator.id ==self.decorator_name
                    for decorator in node.decorator_list
                ):
                    decorator = ast.Name(id=self.decorator_name, ctx=ast.Load())
                    node.decorator_list.insert(0, decorator)
                break
        return node

def add_decorator(file_path, functions_to_modify, decorator_name):
    with open(file_path, "r") as f:
        source_code = f.read()
    tree = ast.parse(source_code)
    modifier = FunctionDecoratorAdder(functions_to_modify, decorator_name)
    modified_tree = modifier.visit(tree)
    modified_code = astor.to_source(modified_tree)
    with open(file_path, "w") as wr:
        wr.write(modified_code)

def add_decorator_ces(file_path, functions_to_modify, decorator_name):
    with open(file_path, "r") as f:
        source_code = f.read()
    tree = ast.parse(source_code)
    modifier = FunctionDecoratorAdder_CES(functions_to_modify, decorator_name)
    modified_tree = modifier.visit(tree)
    modified_code = astunparse.unparse(modified_tree)
    with open(file_path, "w") as wr:
        wr.write(modified_code)


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
    
    

def running_tests(repo_name, tests, repo_path):
    os.chdir(repo_path)
    print(repo_path)
    if repo_name == "django/django":
        for test in tests:
            if "(" not in test or ")" not in test:
                continue
            else:
                prefix = test.split("(")[1].split(")")[0]
                test_case = test.split("(")[0].strip().strip('"')
                test_name = f"{prefix}.{test_case}"
                try:
                    subprocess.run(["./tests/runtests.py", "--verbosity", "2", "--settings=test_sqlite", "--parallel", "1", test_name])
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
    # if repo_name == "sympy/sympy":
    #     subprocess.run(["bin/test", "-C",  "--verbose", "sympy/core/tests/test_basic.py"])
    if repo_name == "matplotlib/matplotlib" or repo_name == "scikit-learn/scikit-learn":
        for test in tests:
            test = test.strip().strip('"').strip("'")
            print(f"pytest {test}")
            try:
                subprocess.run(["pytest", test])
            except Exception as e:
                print(f"An unexpected error occurred: {e}")       

def run_swebench_input_output(instance_id, predictions_path, timeout):
    swebench_root = "/home/changshu/SWE-bench"
    predict_path = predictions_path
    run_id = "OBJ_PLAYGROUND"
    dataset = "princeton-nlp/SWE-bench"
    os.chdir(swebench_root)
    run_cmd = f"python -m swebench.harness.run_evaluation --predictions_path {predict_path} --max_worker 1 --instance_ids {instance_id} --run_id {run_id} --dataset_name {dataset} --timeout {timeout}"
    # print(run_cmd)
    subprocess.run(run_cmd, shell=True)

# def run_swebench_input_output(instance_id, predictions_path):
#     swebench_root = "/home/changshu/SWE-bench"
#     predict_path = predictions_path
#     run_id = "2025JAN"
#     dataset = "princeton-nlp/SWE-bench"
#     os.chdir(swebench_root)
#     run_cmd = f"python -m swebench.harness.run_evaluation --predictions_path {predict_path} --max_worker 1 --instance_ids {instance_id} --run_id {run_id} --dataset_name {dataset}"
#     # print(run_cmd)
#     subprocess.run(run_cmd, shell=True)
    
def run_swebench_input_output_ces(instance_id, predictions_path, run_id):
    swebench_root = "/home/changshu/SWE-bench" ## change the swebench_root
    predict_path = predictions_path
    run_id = run_id
    dataset = "princeton-nlp/SWE-bench"
    os.chdir(swebench_root)
    run_cmd = f"python -m swebench.harness.run_evaluation --predictions_path {predict_path} --max_worker 1 --instance_ids {instance_id} --run_id {run_id} --dataset_name {dataset}"
    # print(run_cmd)
    subprocess.run(run_cmd, shell=True)



def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        data[result['instance_id']] = result['data']
    return data