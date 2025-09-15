def load_original_code(code_path):
    with open(code_path, "r", encoding="utf-8") as f:
        return f.read()


def get_python_code_name(problem_id, benchmark):
    if benchmark == "Classeval":
        file_name_plus_function_name = problem_id.split('@')[1]
        return f"{file_name_plus_function_name.split('.')[0]}.py"
    elif benchmark == "Avatar":
        return f"{problem_id}.py"
    elif benchmark == "HumanEval":
        return f"{problem_id}.py"
    elif benchmark == 'cruxeval':
        return f"{problem_id.replace('sample', 'cruxeval')}.py"
    elif benchmark == 'Swebench':
        return f"{problem_id}.py"
    return None


def get_code_path(problem_id, benchmark):
    python_file_name = get_python_code_name(problem_id, benchmark)
    if benchmark == 'Classeval':
        return f'../../dataset/original_datasets/ClassEval/code/{python_file_name}'
    return f'../../dataset/re2-bench/code/{python_file_name}'
