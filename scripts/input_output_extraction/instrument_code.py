import ast
import os
import subprocess
import sys
import argparse

def get_function_locations(code, class_name, function_name):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef) and body_item.name == function_name:
                    return body_item.lineno
    return -1

                    
def get_all_def(code: str):
    tree = ast.parse(code)
    result = {"__global__": {}}

    for node in tree.body:
        # top-level def
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["__global__"][node.name] = node.lineno

        elif isinstance(node, ast.ClassDef):
            methods = {}
            for body_item in node.body:
                if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if body_item.name != "__init__":  # keep if you want to skip __init__
                        methods[body_item.name] = body_item.lineno
            # Only add the class key if it actually has methods
            if methods:
                result[node.name] = methods

    # Drop "__global__" if empty so a class-only file returns just classes
    if not result["__global__"]:
        result.pop("__global__")

    return result

def instrument_code(code, line_nos):
    new_lines = []
    lines = code.splitlines()
    for idx, line in enumerate(lines, start=1):
        if idx not in line_nos:
            new_lines.append(line)
        else:
            indent_str = line[:len(line) - len(line.lstrip())]
            new_lines.append(indent_str + "@inspect_code")
            new_lines.append(line)
    new_code = "\n".join(new_lines)
    return new_code

def instrument(className, line_nos, code_path, test_path, root_inst, root_result, entry_point=""):
    
    inspect_code_path = "./input_output_extraction/inspect_code_generic.py"
    inspect_code = open(inspect_code_path, 'r').read()
    modified_code_path = os.path.join(root_inst, f"{className}-inst.py")
    result_folder = root_result
    inspect_code = inspect_code.replace("{CODE_FOLDER}", result_folder)
    inspect_code = inspect_code.replace("{CLASS_NAME}", className)
    code = open(code_path, 'r').read()
    test_code = open(test_path, 'r').read()
    entry = f"check({entry_point})" if entry_point else ""
    new_code = inspect_code + "\n\n" + instrument_code(code, line_nos) + "\n\n" + test_code + "\n" + entry
    with open(modified_code_path, 'w') as wr:
        wr.write(new_code)
    
    
    # execute the instrumented code with pytest
        
    try:
        if "HumanEval" in className or "cruxeval" in className:
            cmd = [
                "python",
                modified_code_path
            ]
        else:
            cmd = [
                "pytest",
                modified_code_path
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return 1
    except Exception as err:
        print(err)
        return 2
    
    
    
    
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--codeDir', type=str, default="", help="root dir of the code")
    parser.add_argument('--testDir', type=str, default="", help="root dir of the tests")
    parser.add_argument('--entryDir', type=str, default="", help="root dir of the entry points")
    parser.add_argument('--instDir', type=str, default="", help="root dir of the instrumented code")
    parser.add_argument('--resultDir', type=str, default="", help="root dir of the .jsonl results")
    
    args = parser.parse_args()
    root = args.codeDir
    root_test_suite = args.testDir
    root_entry_point = args.entryDir
    root_inst = args.instDir
    root_result = args.resultDir
    
    if not os.path.exists(root_inst):
        os.makedirs(root_inst)
    if not os.path.exists(root_result):
        os.makedirs(root_result)
    
    for d in os.listdir(root):
        if d == "__pycache__" or d.endswith(".png"): continue
        file_path = os.path.join(root,d)
        test_path = os.path.join(root_test_suite, d)
        entry_path = os.path.join(root_entry_point, d.replace('.py','.txt'))
        
        if os.path.exists(entry_path):
            entry_point = open(entry_path, 'r').read()
        else:
            entry_point = ""
        # print(f"###{file_path}###")
        try:
            code = open(file_path, 'r').read()
            defs = get_all_def(code)
            for className in defs:
                linos = [defs[className][k] for k in defs[className]]
                if className != "__global__":
                    instrument(className, linos, file_path, test_path, root_inst, root_result, entry_point)
                else:
                    instrument(d.rstrip(".py"), linos, file_path, test_path, root_inst, root_result, entry_point)
        except Exception as e:
            print(f"###{file_path}###")
            print(e)