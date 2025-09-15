import os
import json 


import ast
import inspect
from swebench_utils import init_playground
from code_utils import strip_comments_and_docstrings, extract_method_code
from datasets import load_dataset
import textwrap

# def extract_method_code(file_path, method_name, class_name=None):
#     with open(file_path, "r", encoding="utf-8") as f:
#         code = f.read()

#     tree = ast.parse(code)
#     lines = code.splitlines()

#     def get_source_segment(node):
#         start = node.lineno - 1
#         end = node.end_lineno
#         return "\n".join(lines[start:end])

#     # Case 1: Top-level function
#     if class_name is None:
#         for node in tree.body:
#             if isinstance(node, ast.FunctionDef) and node.name == method_name:
#                 return get_source_segment(node)
#         return None

#     # Case 2: Method inside a class
#     for node in tree.body:
#         if isinstance(node, ast.ClassDef) and node.name == class_name:
#             for item in node.body:
#                 if isinstance(item, ast.FunctionDef) and item.name == method_name:
#                     return get_source_segment(item)
#     return None

def get_dependency(instance_id, file_path, function_name):
    file_path ="/" + file_path.removesuffix(".py").replace(".", "/") + ".py"
    project_name = instance_id.split("-")[0].split("__")[-1]
    dependency_json_path=f"/home/changshu/CODEMIND/scripts/swebench/dependency_pipeline/data/{project_name}.json"
    dependencies_all = []
    if not os.path.exists(path=dependency_json_path):
        print(f"{dependency_json_path} can not be found!")
        return [] 
    with open(dependency_json_path, 'r') as f:
        dependency_data = json.load(f)
    try:
        dependencies = dependency_data[instance_id][file_path][function_name]
        for deps in dependencies:
            for dep in deps:
                if dep not in dependencies_all:
                    dependencies_all.append(dep)
        return dependencies_all
    except:
        print(f"Can not find {instance_id} {file_path} {function_name} in {dependency_json_path}")
        return []



def create_repos():
    ds = load_dataset("princeton-nlp/SWE-bench", split="test",  cache_dir="/home/shared/huggingface")
    existed_instances = os.listdir("/home/changshu/CODEMIND/scripts/swebench/swebench_playground/dep")
    for d in os.listdir("/home/changshu/RE2-Bench/dataset/re2-bench/code"):
        if "@@" in d:
            isntance_id = d.split("@@")[0]
            if isntance_id not in existed_instances:
                init_playground(isntance_id, ds)
            
# dependencies = get_dependency(
#     instance_id = "astropy__astropy-8747",
#     file_path = "astropy.units.quantity.py",
#     function_name = "__array_ufunc__"
# )
    

# code = extract_method_code(
#     file_path = "/home/changshu/CODEMIND/scripts/swebench/swebench_playground/dep/astropy__astropy-12842/astropy/time/core.py",
#     method_name = "_construct_from_dict",
#     class_name = "TimeInfoBase"
# )

# print(code)

def main(instance_id, file_path, function_name, code_path):
    dependency_results = {}
    code_ut = open(code_path, 'r').read()
    
    dep_file_root = "/home/changshu/CODEMIND/scripts/swebench/swebench_playground/dep"
    dependencies = get_dependency(
        instance_id = instance_id,
        file_path = file_path, 
        function_name = function_name
    )
    # print(dep_file_root)
    for dep in dependencies:
        # print(dep)
        dep_file_path = dep.split("@@")[0].removesuffix('.py').replace(".", "/") + ".py"
        dep_file_path = f"{dep_file_root}/{instance_id}{dep_file_path}"
        
        method_name = dep.split("@@")[-1] if "." not in dep.split("@@")[-1] else dep.split("@@")[-1].split(".")[-1]
        class_name = dep.split("@@")[-1].split(".")[0] if "." in dep.split("@@")[-1] else None
        
        # print(method_name, class_name)
        dep_code = extract_method_code(
            file_path = dep_file_path,
            method_name = method_name,
            class_name = class_name
        )
        if dep_code is None:
            print(f"Can not extract code for {dep} in {dep_file_path}")
            continue
        if dep_code in code_ut:
            print(f"Dependency {dep} is found in the UT code!")
        else:
            dedented_dep_code = textwrap.dedent(dep_code).strip() ## remove leading spaces and extra indentation
            try:
                cleaned_dep_code = strip_comments_and_docstrings(dedented_dep_code) ## remove comments and docstrings
            except:
                cleaned_dep_code = dedented_dep_code
            dependency_results[dep] = cleaned_dep_code
    ## now write the dependency results to a json file
    json_root = "../dataset/re2-bench/dependency"
    json_file_path = f"{json_root}/{instance_id}@@{file_path}@@{function_name}.json"
    with open(json_file_path, 'w') as f:
        json.dump(dependency_results, f, indent=4)
    
        

# main(
#     instance_id = "astropy__astropy-7606",
#     file_path = "astropy.units.core.py",
#     function_name = "__div__",
#     code_path = "/home/changshu/RE2-Bench/dataset/re2-bench/code/astropy__astropy-7606@@astropy.units.core.py@@__div__.py"
# )

if __name__ == "__main__":
    # create_repos()
    for d in os.listdir("../dataset/re2-bench/code"):
        if "@@" in d:
            instance_id = d.split("@@")[0]
            file_path = d.split("@@")[1]
            function_name = d.split("@@")[-1].removesuffix(".py")
            code_path = f"../dataset/re2-bench/code/{d}"
            print(f"Processing {instance_id} {file_path} {function_name}")
            main(
                instance_id = instance_id,
                file_path = file_path,
                function_name = function_name,
                code_path = code_path
            )
        
