import ast
import os
import json
from star_plot import star_plot

class NestedLoopDetector(ast.NodeVisitor):
    def __init__(self):
        self.nested_loop_found = False

    def visit_For(self, node):
        # Check for nested loops inside this 'For' loop
        if any(isinstance(child, (ast.For, ast.While)) for child in ast.iter_child_nodes(node)):
            self.nested_loop_found = True
        self.generic_visit(node)  # Continue visiting child nodes

    def visit_While(self, node):
        # Check for nested loops inside this 'While' loop
        if any(isinstance(child, (ast.For, ast.While)) for child in ast.iter_child_nodes(node)):
            self.nested_loop_found = True
        self.generic_visit(node)

def has_nested_loops(code):
    tree = ast.parse(code)
    detector = NestedLoopDetector()
    detector.visit(tree)
    return detector.nested_loop_found

class NestedIfDetector(ast.NodeVisitor):
    def __init__(self):
        self.nested_if_found = False

    def visit_If(self, node):
        # Check if there's a nested 'if' statement within this 'if' statement's body
        if any(isinstance(child, ast.If) for child in ast.iter_child_nodes(node)):
            self.nested_if_found = True
        self.generic_visit(node)  # Continue visiting child nodes

def has_nested_if(code):
    tree = ast.parse(code)
    detector = NestedIfDetector()
    detector.visit(tree)
    return detector.nested_if_found

class UnnestedIfDetector(ast.NodeVisitor):
    def __init__(self):
        self.unnested_if_found = False

    def visit_If(self, node):
        # Check if this 'if' statement has no nested 'if' in its body
        if not any(isinstance(child, ast.If) for child in ast.iter_child_nodes(node)):
            self.unnested_if_found = True
        self.generic_visit(node)  # Continue visiting child nodes

def has_unnested_if(code):
    tree = ast.parse(code)
    detector = UnnestedIfDetector()
    detector.visit(tree)
    return detector.unnested_if_found

class ForLoopDetector(ast.NodeVisitor):
    def __init__(self):
        self.for_loop_found = False

    def visit_For(self, node):
        # If a 'for' loop is found, set the flag to True
        self.for_loop_found = True
        # No need to go further since we found a for loop
        return  # Exit early after finding the first 'for' loop

def has_for_loop(code):
    tree = ast.parse(code)
    detector = ForLoopDetector()
    detector.visit(tree)
    return detector.for_loop_found

class WhileLoopDetector(ast.NodeVisitor):
    def __init__(self):
        self.while_loop_found = False

    def visit_While(self, node):
        # If a 'while' loop is found, set the flag to True
        self.while_loop_found = True
        # No need to go further since we found a while loop
        return  # Exit early after finding the first 'while' loop

def has_while_loop(code):
    tree = ast.parse(code)
    detector = WhileLoopDetector()
    detector.visit(tree)
    return detector.while_loop_found



class TryExceptDetector(ast.NodeVisitor):
    def __init__(self):
        self.try_except_found = False

    def visit_Try(self, node):
        # If a 'try' block is found, set the flag to True
        self.try_except_found = True
        # No need to go further since we found a try-except block
        return  # Exit early after finding the first try-except block

def has_try_except(code):
    tree = ast.parse(code)
    detector = TryExceptDetector()
    detector.visit(tree)
    return detector.try_except_found

# Example usage
def extract_classeval(transform=False):
    overall_results = {}
    root = "../dataset"
    for c in ['high', 'medium', 'low']:
        folder = os.path.join(root, c)
        for d in os.listdir(folder):
            file_path = os.path.join(folder, d, 'main.py')
            code = open(file_path, 'r').read()
            results = {
                "nested": 0,
                "if": 0,
                "for": 0,
                "while": 0,
                "try": 0,
                "switch": 0,
                "basic": 1,
                "nested_if": 0
            }
            try:
                if has_nested_loops(code):
                    results["nested"] = 1
                    results["basic"] = 0
                if has_unnested_if(code):
                    results["if"] = 1
                    results["basic"] = 0
                if has_for_loop(code):
                    results["for"] = 1
                    results["basic"] = 0
                if has_while_loop(code):
                    results["while"] = 1
                    results["basic"] = 0
                if has_try_except(code):
                    results["try"] = 1
                    results["basic"] = 0
                if has_nested_if(code):
                    results["nested_if"] = 1
                    results["basic"] = 0
            except:
                print(file_path)
            overall_results[d] = results
    path = "../summary/constructs.json"
    with open(path, 'w') as wr:
        json.dump(overall_results, wr, indent=4)
    # print(overall_results)

def extract_constructs(data_path, label_path, cat):
    constraints = os.listdir(f"../dataset/{cat}")
    result = {}
    correct_ids = []
    total_count = {
        'nested': 0, 'if': 0, 'for': 0, 'while': 0, 'try': 0, 'switch': 0, 'basic': 0, "nested_if": 0
    }
    correct_count = {
        'nested': 0, 'if': 0, 'for': 0, 'while': 0, 'try': 0, 'switch': 0, 'basic': 0, "nested_if": 0
    }
    ## loading data
    with open(data_path, 'r') as reader:
        data_constructs = json.load(reader)
    with open(label_path, 'r') as reader1:
        data_labels = json.load(reader1)
    
    for k in data_constructs.keys():
        if k not in constraints:
            continue
        for j in data_constructs[k].keys():
            total_count[j] += data_constructs[k][j]
        if k not in data_labels: continue
        if data_labels[k] == 1:
            for j in data_constructs[k].keys():
                correct_count[j] += data_constructs[k][j]
    for k in total_count.keys():
        if total_count[k]>1:
            result[k] = correct_count[k] / total_count[k]
    return result


def main(task, cat):
    data_path = "../Results/summary/constructs.json"
    label_path_omini = f"../Results/summary/{task}_o4-mini-2025-04-16.json"
    label_path_gemini2 = f"../Results/summary/{task}_gemini-2.5-pro-preview-03-25.json"
    label_path_deepseek = f"../Results/summary/{task}_deepseek-r1.json"
    label_path_gptf = f"../Results/summary/{task}_gpt-4.1-2025-04-14.json"
    label_path_gemini1 = f"../Results/summary/{task}_gemini-1.5-pro.json"
    label_path_deepseekcoder = f"../Results/summary/{task}_deepseek-coder-33b-instruct.json"
    
    omini = extract_constructs(data_path, label_path_omini, cat)
    gemini2 = extract_constructs(data_path, label_path_gemini2, cat)
    deepseekr1 = extract_constructs(data_path, label_path_deepseek, cat)
    gpt4 = extract_constructs(data_path, label_path_gptf, cat)
    gemini1 = extract_constructs(data_path, label_path_gemini1, cat)
    deeepseekcoder = extract_constructs(data_path, label_path_deepseekcoder, cat)
    label = [k for k in omini.keys()]
    label_new = []
    for l in label:
        if l == 'nested':
            label_new.append("NL")
        if l == 'if':
            label_new.append("I")
        if l == 'for':
            label_new.append("F")
        if l == "while":
            label_new.append("W")
        if l == 'try':
            label_new.append("T")
        if l == 'basic':
            label_new.append("B")
        if l == "switch":
            label_new.append("S")
        if l == 'nested_if':
            label_new.append("NI")
    print(label_new)
    omini_task = [omini[l] for l in label]
    gemini2_task = [gemini2[l] for l in label]
    deepseekr1_task = [deepseekr1[l] for l in label]
    gpt4_task = [gpt4[l] for l in label]
    gemini1_task = [gemini1[l] for l in label]
    deepseekcoder_task = [deeepseekcoder[l] for l in label]

    # deepseekcoder_task[0] = 0
    label_new = ['I', 'F', 'W', 'T', 'B', 'NI']
    data = [
        label_new,
        (
            task, [
                omini_task,
                gemini2_task,
                deepseekr1_task,
                gpt4_task,
                gemini1_task,
                deepseekcoder_task
            ]
        )
    ]
    # print(data)
    labels = ('O4-mini','Gemini-2.5-Pro', 'DeepSeek-R1', 'GPT-4.1', 'Gemini-1.5-Pro', 'DeepSeek-Coder-Inst-33b')
    star_plot(data, len(label), labels, f"{task}_{cat}")


main("output", "low")   