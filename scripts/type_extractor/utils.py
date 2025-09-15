import ast
import json
import os
from collections import deque
from typing import Any, List, Callable, Dict
import numpy as np

difficulty_levels = ['high', 'medium', 'low']
dataset_dir_path = '../../dataset'
EVAL_PROJECTS_RAW_DATASET_DIR_PATH = '../../dataset/raw'


def infer_constant(node: ast.Constant):
    val = node.value
    if val is None:
        return "Any"  # type_description="the value in ground truth is null")
    if isinstance(val, str) and val.__contains__("<html>"):
        return "html code as string"
    elif isinstance(val, str) and val.__contains__("<p>"):
        return "html tag as string"
    return type(val).__name__


def infer_itertable(node: ast.List | ast.Set | ast.Tuple, type_name: str):
    return infer_iterable_value(node.elts, type_name, is_ast=True)


def infer_iterable_value(value: list | set | tuple | deque[str | int], type_name: str, is_ast: bool = False):
    if not value:
        return f"empty {type_name} in ground truth"
    func = infer_type if is_ast else infer_type_from_value
    elem_types = [func(elt) for elt in value]
    base_type = elem_types[0]
    if all(element_type == base_type for element_type in elem_types):
        return f'{type_name} of {len(elem_types)} {base_type}s'
    if any(isinstance(element_type, dict) for element_type in elem_types):
        return elem_types
    return f'{type_name} of different types containing {len(elem_types)} elements{list(dict.fromkeys(elem_types))}'


def infer_dict(node: ast.Dict):
    return infer_dict_value(node.keys, node.values, True)


def infer_dict_value(keys, values, is_ast: bool = False):
    if not keys:
        return "in ground truth values, it is an empty dict"
    types = {}
    for k, v in zip(keys, values):
        inferring_func = infer_type_from_value if not is_ast else infer_type
        key_value = k if not is_ast else k.value
        types[key_value] = inferring_func(v)
    return types


def infer_str_value(value: str):
    return str.__name__


def infer_primitives_value(value: str):
    return type(value).__name__


def infer_type_from_value(value: Any,
                          infer_str_fun: Callable[[str], str] = infer_str_value,
                          infer_primitives_fun: Callable[[str], str] = infer_primitives_value,
                          infer_iterable_fun: Callable = infer_iterable_value,
                          infer_dict_fun: Callable = infer_dict_value):
    if value is None:
        return "null value in the ground truth"
    elif type(value) in [bool, int, float]:
        return infer_primitives_fun(value)
    elif isinstance(value, str):
        return infer_str_fun(value)
    elif isinstance(value, list):
        return infer_iterable_fun(value, type_name=list.__name__)
    elif isinstance(value, dict):
        return infer_dict_fun(value.keys(), value.values())
    elif isinstance(value, tuple):
        return infer_iterable_fun(value, tuple.__name__)
    elif isinstance(value, deque):
        return infer_iterable_fun(value, type(value).__name__)
    elif isinstance(value, np.ndarray):
        return infer_iterable_fun(value, np.ndarray.__name__)
    else:
        return "Any"


def infer_type(node: ast.AST):
    if isinstance(node, ast.Constant):
        return infer_constant(node)
    elif isinstance(node, ast.List):
        return infer_itertable(node, list.__name__)
    elif isinstance(node, ast.Tuple):
        return infer_itertable(node, tuple.__name__)
    elif isinstance(node, ast.Set):
        return infer_itertable(node, set.__name__)
    elif isinstance(node, ast.Dict):
        return infer_dict(node)
    elif isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            func_name = ".".join(reversed(parts))
        return func_name or "Any"
    elif isinstance(node, ast.BinOp):
        left_type = infer_type(node.left)
        right_type = infer_type(node.right)
        if left_type == right_type:
            return left_type
        return f"{left_type} | {right_type}"
    elif isinstance(node, ast.UnaryOp):
        return infer_type(node.operand)
    elif isinstance(node, ast.Subscript):
        return "Subscript"
    else:
        return "Any"  # type_description="not detected type by our type extractor")


class TypeCase:
    def __init__(self, input_type, output_type, input_content, output_content):
        self.input_type = input_type
        self.output_type = output_type
        self.input_json = read_str_as_json(input_content)
        self.output_json = read_str_as_json(output_content)


def read_str_as_json(input_content: str):
    try:
        return json.loads(input_content)
    except Exception:
        return input_content


def create_input_output_json(type_case: TypeCase, case_dir_path: str):
    input_output_json_data = {'types': {'inputs': type_case.input_type, 'outputs': type_case.output_type},
                              'values': {'inputs': type_case.input_json, 'outputs': type_case.output_json}}
    input_output_json_path = os.path.join(case_dir_path, 'input-output.json')
    with open(input_output_json_path, 'w') as f:
        json.dump(input_output_json_data, f, indent=2)


def get_all_called_functions_names(code: str) -> List[str]:
    tree = ast.parse(code)
    return [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)]


def get_all_function_names(code: str) -> List[str]:
    tree = ast.parse(code)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def read_input_values_for_function(content: str, func_name: str):
    try:
        tree = ast.parse(content, mode='eval') if not content.endswith(';') else ast.parse(content)
        expr = tree.body if isinstance(tree, ast.Expression) else tree.body[0].value
        if isinstance(expr, ast.Call) and getattr(expr.func, 'id', '') == func_name:
            return [ast.literal_eval(arg) for arg in expr.args]
        else:
            print(f"No valid function call to `{func_name}` found.")
    except Exception as e:
        print("Error parsing input file:", e)
    return None


def get_function_args(code: str, func_name: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [arg.arg for arg in node.args.args]
    return []


def get_file_content(file_name: str, case_dir_path: str):
    file_path = os.path.join(case_dir_path, file_name)
    with open(file_path, 'r') as f:
        return f.read().strip()


def get_input_output_code_content(case_dir_path: str):
    input_content = get_file_content('input.txt', case_dir_path)
    output_content = get_file_content('output.txt', case_dir_path)
    code_content = get_file_content('main.py', case_dir_path)
    return input_content, output_content, code_content


def read_input_values_for_dotted_function(content: str, func_name: str):
    try:
        tree = ast.parse(content)

        expr = tree.body[0].value if isinstance(tree, ast.Module) else tree.body

        def get_full_func_name(node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return get_full_func_name(node.value) + '.' + node.attr
            else:
                return ''

        if isinstance(expr, ast.Call):
            call_name = get_full_func_name(expr.func)
            if call_name == func_name:
                result = [infer_type(arg) for arg in expr.args]
                return result
            else:
                print(f"Function call found but name '{call_name}' != '{func_name}'")
        else:
            print("No valid function call found in expression.")
    except Exception as e:
        print("Error parsing input:", e)
    return None


def get_project_cases_per_level(project_name: str, level_dir_path: str) -> list[str]:
    return [case_name for case_name in os.listdir(level_dir_path) if case_name.startswith(project_name)]


def load_jsonl_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def save_jsonl_file(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
