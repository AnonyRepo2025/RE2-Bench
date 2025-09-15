import json
import ast
import os
from typing import Optional, Union, Iterable

class _DocstringStripper(ast.NodeTransformer):
    """Remove first-statement string literals used as docstrings."""
    def _strip_in(self, node):
        if node.body and isinstance(node.body[0], ast.Expr):
            v = node.body[0].value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                node.body = node.body[1:]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip_in(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_in(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_in(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip_in(node)

def strip_comments_and_docstrings(source: str, keep_shebang: bool = True) -> str:
    shebang = ""
    lines = source.splitlines()
    if keep_shebang and lines and lines[0].startswith("#!"):
        shebang = lines[0] + "\n"
        source_body = "\n".join(lines[1:])
    else:
        source_body = source

    tree = ast.parse(source_body)
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    cleaned = ast.unparse(tree)

    return shebang + cleaned

def read_sampling_results():
    samples = []
    sample_path = "../dataset/re2-bench/sampled_problems.json"
    with open(sample_path, 'r') as f:
        sampling_results = json.load(f)
        for i in sampling_results:
            for k in sampling_results[i]:
                samples.append(
                    {
                        "difficulty": i,
                        "id": k,
                        "source": sampling_results[i][k]['benchmark'],
                        "io_id": sampling_results[i][k]['input-output']
                    }
                )
    return samples

def read_code(index, source):
    if source in ["Swebench", "Avatar", "HumanEval"]:
        code_path = f"../dataset/re2-bench/code/{index}.py"
    elif source == "Classeval":
        code_path = f"../dataset/re2-bench/code/{index.split("@")[-1]}.py"
    elif source == "cruxeval":
        code_path = f"../dataset/re2-bench/code/{index.replace("sample", "cruxeval")}.py"
    else:
        print(f"Can not find source for {index}")
    if not os.path.exists(code_path):
        print(f"Can not find code path for {code_path}")
        return ""
    code = open(code_path, 'r').read()
    ## remove docstring and comments
    cleaned_code = strip_comments_and_docstrings(code)
    return cleaned_code
            

def read_io(index, source, io_id):
    ## return the input, output, and the entry point.
    input_ouput_root = f"../dataset/re2-bench/input-output/{source}"
    if source == "Swebench":
        modified_filename = index.split("@@")[0] + "@@" + index.split("@@")[1].rstrip(".py") + "." + index.split("@@")[2] + ".jsonl"
    elif source == "cruxeval":
        modified_filename = index.replace("sample", "cruxeval") + ".jsonl"
    elif source == "HumanEval":
        modified_filename = index + ".jsonl"
    elif source == "Classeval":
        modified_filename = index.split("@")[-1].replace(".", "@") + ".jsonl"
    elif source == "Avatar":
        modified_filename = index + ".jsonl"
    file_path = os.path.join(input_ouput_root, modified_filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        ## sample an input-output pair
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == io_id:
                    io_info = json.loads(line)
                    if source == "Avatar":
                        input_info = io_info["values"]["input"]
                        output_info = io_info["values"]["output"]
                        return {"input": input_info, "output": output_info, "entry": ""}
                    else:
                        input_info = io_info["values"]["inputs"]
                        output_info = io_info["values"]["return"]
                        entry_point = io_info["name"]
                        return {"input": input_info, "output": output_info, "entry": entry_point}

def read_dependency(index):
    ## read the dependency information, only for Swebench
    dependency_string = ""
    dependency_path = f"../dataset/re2-bench/dependency/{index}.json"
    with open(dependency_path, 'r') as f:
        dependency_info = json.load(f)
        for k in dependency_info:
            dep_code = dependency_info[k]
            package_name = k.split("@@")[0].removesuffix(".py") + "." + k.split("@@")[1]
            dependency_string += package_name + "\n\n"
            dependency_string += dep_code + "\n\n"
            counter += 1
    return dependency_string
    
     
def mask_input(data):
    def mask_values(obj):
        if isinstance(obj, dict):
            if not obj:  # empty dict
                return {}
            return {k: mask_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if not obj:  # empty list
                return "XXX"
            return "XXX"
        else:
            return "XXX"
    masked_json = mask_values(data)
    masked_str = json.dumps(masked_json, indent=4).replace('"XXX"', "XXX")
    return masked_str

def mask_output(data):
    if isinstance(data, dict):
        # data = {"output": data}
        return mask_input(data)
    else:
        return """{
    "output": XXX        
}"""

def test_mask_value():
    data = {
    "self": {
        "_tmppath_factory": {
            "_given_basetemp": "/tmp/pytest-of-root/pytest-0/test_mktemp0",
            "_trace": {},
            "_basetemp": None,
            "value": 2
        }
    },
    "args": {
        "basename": "world"
    },
    "kwargs": {}
    }
    mask_input_output(data)




