import json


def get_class_name(case_name: str, benchmark: str):
    if benchmark == 'Avatar':
        return None
    elif benchmark == 'Classeval':
        return case_name.split("@")[0]
    elif benchmark == 'cruxeval':
        return None
    elif benchmark == 'HumanEval':
        return None
    return None


def read_case_name(problem_id, benchmark):
    if benchmark == "Swebench":
        modified_filename = problem_id.split("@@")[0] + "@@" + problem_id.split("@@")[1].rstrip(".py") + "." + \
                            problem_id.split("@@")[2] + ".jsonl"
    elif benchmark == "cruxeval":
        modified_filename = problem_id.replace("sample", "cruxeval") + ".jsonl"
    elif benchmark == "HumanEval":
        modified_filename = problem_id + ".jsonl"
    elif benchmark == "Classeval":
        modified_filename = problem_id.split("@")[-1].replace(".", "@") + ".jsonl"
    elif benchmark == "Avatar":
        modified_filename = problem_id + ".jsonl"
    return modified_filename


def get_case_index(case_id, difficulty_level):
    with open("../../dataset/re2-bench/sampled_problems.json", 'r') as f:
        sampled_data = json.load(f)
    case_index_and_project = sampled_data[difficulty_level][case_id]
    project = case_index_and_project['benchmark']
    index = case_index_and_project['input-output']
    return index, project


class CaseInfo:
    def __init__(self, case_name, case_index, problem_id, benchmark, task, code_path):
        self.case_name = case_name
        self.case_index = case_index
        self.problem_id = problem_id
        self.benchmark = benchmark
        self.task = task
        self.code_path = code_path
