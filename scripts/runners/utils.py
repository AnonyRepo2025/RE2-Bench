import json
import os.path

dataset_dir_path = '../../dataset'
models = ['deepseek-coder-33b-instruct', 'deepseek-r1', 'gemini-1.5-pro', 'gemini-2.5-pro-preview-03-25',
          'gpt-4.1-2025-04-14', 'o4-mini-2025-04-16']
input_experiments_dir_path = '../../Results/experiment_results_input'
output_experiments_dir_path = '../../Results/experiment_results_output'
summary_dir_path = '../../Results/summary'


def get_failed_input_cases_names(project_name: str, model_name: str):
    case_success_map = get_model_summary_cases(model_name, 'input')
    return extract_project_failed_cases_names(case_success_map, project_name)


def get_failed_output_cases_names(project_name: str, model_name: str):
    case_success_map = get_model_summary_cases(model_name, 'output')
    return extract_project_failed_cases_names(case_success_map, project_name)


def get_model_summary_cases(model_name: str, task_type: str) -> dict[str, int]:
    summary_file_path = os.path.join(summary_dir_path, f'{task_type}_{model_name}.json')
    with open(summary_file_path, 'r') as f:
        return json.load(f)


def extract_project_failed_cases_names(cases: dict[str, int], project_name: str) -> list[str]:
    return [case_name for case_name, is_success in cases.items() if case_name.startswith(project_name) and not is_success]


def get_ground_truth_input(case_name: str, level: str):
    input_file_path = os.path.join(dataset_dir_path, level, case_name, 'input.txt')
    with open(input_file_path, 'r') as f:
        return f.read().strip()


def get_ground_truth_output(case_name: str, level: str):
    output_file_path = os.path.join(dataset_dir_path, level, case_name, 'output.txt')
    with open(output_file_path, 'r') as f:
        return f.read().strip()


def get_code_content(case_name: str, level: str):
    code_file_path = os.path.join(dataset_dir_path, level, case_name, 'main.py')
    with open(code_file_path, 'r') as f:
        return f.read().strip()


def get_level_of_case(case_name: str):
    levels = ['high', 'low', 'medium']
    for level in levels:
        level_dir_path = os.path.join(dataset_dir_path, level)
        if os.listdir(level_dir_path).__contains__(case_name):
            return level
    return None

def get_generated_input(model: str, level: str, case_name: str):
    experiment_generated_input_json_file_path = os.path.join(input_experiments_dir_path, model, level, case_name, 'response.json')
    with open(experiment_generated_input_json_file_path, 'r') as f:
        return json.load(f)["input"]