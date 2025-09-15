import json
import os
import re
import subprocess
from collections import defaultdict

from scripts.runners.utils import models, get_failed_input_cases_names, get_level_of_case, \
    get_ground_truth_output, get_code_content, get_generated_input, get_ground_truth_input


# convert Test.test to Test().test todo: probably, we should delete this function after Changshu edits its pipeline for class eval
def class_eval_correct_class_initiation_generated_input(generated_input: str) -> str:
    pattern = r'(\b[A-Z][A-Za-z0-9_]*)\.(\w+)\('
    corrected_input = re.sub(pattern, r'\1().\2(', generated_input)
    return corrected_input


def run_class_eval_generated_input(original_code: str, generated_input: str):
    correct_input = class_eval_correct_class_initiation_generated_input(generated_input)
    runnable_code = f"{original_code}\nprint({correct_input})\n"
    temp_python_file_path = './temp.py'
    with open(temp_python_file_path, 'w') as f:
        f.write(runnable_code)
    output = subprocess.run(
        ['python', temp_python_file_path],
        text=True,
        capture_output=True,
        timeout=15,
        check=True
    )
    os.remove(temp_python_file_path)
    return output.stdout.strip()


def run_human_eval_generated_input(original_code: str, generated_input: str):
    runnable_code = f"{original_code}\nprint({generated_input})\n"
    temp_python_file_path = './temp.py'
    with open(temp_python_file_path, 'w') as f:
        f.write(runnable_code)
    output = subprocess.run(
        ['python', temp_python_file_path],
        text=True,
        capture_output=True,
        timeout=15,
        check=True
    )
    os.remove(temp_python_file_path)
    return output.stdout.strip()


def is_the_case_false_negative(generated_output: str, ground_truth_output: str):
    return generated_output == ground_truth_output


def detect_human_eval_false_negatives():
    project_name = 'HumanEval'
    results = defaultdict(lambda: defaultdict(list))
    false_negatives_number = 0
    for model in models:
        failed_cases = get_failed_input_cases_names(project_name, model)
        for case in failed_cases:
            level = get_level_of_case(case)
            ground_truth_input = get_ground_truth_input(case, level)
            ground_truth_output = get_ground_truth_output(case, level)
            code_content = get_code_content(case, level)
            generated_input = get_generated_input(model, level, case)
            try:
                output_of_generated_input = run_human_eval_generated_input(code_content, generated_input)
            except subprocess.TimeoutExpired as e:
                print(f"{case}-{level}-{model} encountered an error")
                print(e)
                continue
            except subprocess.CalledProcessError as e:
                print(f"{case}-{level}-{model} encountered an error")
                print(e)
                continue
            if is_the_case_false_negative(output_of_generated_input, ground_truth_output):
                print(
                    f"{case}-{level}-{model}-{generated_input}-{ground_truth_input}-{output_of_generated_input}-{ground_truth_output}")
                results[model][level].append(case)
                false_negatives_number += 1
    with open('./human_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"total number of false negatives {project_name}: {false_negatives_number}")


def install_requirements_class_eval():
    requirements_file_path = './class-eval-requirements.txt'
    try:
        subprocess.run(
            ['python', '-m', 'pip', 'install', '-r', requirements_file_path],
            check=True,
            text=True
        )
        print("Successfully installed ClassEval requirements...")
    except Exception as e:
        pass


def detect_class_eval_false_negatives():
    project_name = 'ClassEval'
    results = defaultdict(lambda: defaultdict(list))
    false_negatives_number = 0
    for model in models:
        failed_cases = get_failed_input_cases_names(project_name, model)
        for case in failed_cases:
            level = get_level_of_case(case)
            ground_truth_input = get_ground_truth_input(case, level)
            ground_truth_output = get_ground_truth_output(case, level)
            code_content = get_code_content(case, level)
            generated_input = get_generated_input(model, level, case)
            try:
                output_of_generated_input = run_class_eval_generated_input(code_content, generated_input)
            except subprocess.TimeoutExpired as e:
                print(f"{case}-{level}-{model} encountered an error")
                continue
            except subprocess.CalledProcessError as e:
                print(f"{case}-{level}-{model} encountered an error")
                continue
            if is_the_case_false_negative(output_of_generated_input, ground_truth_output):
                print(
                    f"{case}-{level}-{model}-{generated_input}-{ground_truth_input}-{output_of_generated_input}-{ground_truth_output}")
                results[model][level].append(case)
                false_negatives_number += 1
    with open('./class_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"total number of false negatives {project_name}: {false_negatives_number}")


def detect_crux_eval_false_negatives():
    project_name = 'cruxeval'
    results = defaultdict(lambda: defaultdict(list))
    false_negatives_number = 0
    for model in models:
        failed_cases = get_failed_input_cases_names(project_name, model)
        for case in failed_cases:
            level = get_level_of_case(case)
            ground_truth_input = get_ground_truth_input(case, level)
            ground_truth_output = get_ground_truth_output(case, level)
            code_content = get_code_content(case, level)
            generated_input = get_generated_input(model, level, case)
            try:
                output_of_generated_input = run_human_eval_generated_input(code_content, generated_input)
            except subprocess.TimeoutExpired as e:
                print(f"{case}-{level}-{model} encountered an Timeout error")
                continue
            except subprocess.CalledProcessError as e:
                print(f"{case}-{level}-{model} encountered an error")
                print(e.stderr)
                continue
            if is_the_case_false_negative(output_of_generated_input, ground_truth_output):
                print(
                    f"{case}-{level}-{model}-{generated_input}-{ground_truth_input}-{output_of_generated_input}-{ground_truth_output}")
                results[model][level].append(case)
                false_negatives_number += 1
    with open('./crux_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"total number of false negatives {project_name}: {false_negatives_number}")


def run_avatar_generated_input(original_code: str, generated_input: str):
    temp_python_file_path = './temp.py'
    if os.path.exists(temp_python_file_path):
        os.remove(temp_python_file_path)
    with open(temp_python_file_path, 'w') as f:
        f.write(original_code)
    output = subprocess.run(
        ['python', temp_python_file_path],
        text=True,
        input=generated_input,
        capture_output=True,
        timeout=15,
        check=True
    )
    os.remove(temp_python_file_path)
    return output.stdout.strip()


def detect_avatar_false_negatives():
    project_name = 'Avatar'
    results = defaultdict(lambda: defaultdict(list))
    false_negatives_number = 0
    for model in models:
        failed_cases = get_failed_input_cases_names(project_name, model)
        for case in failed_cases:
            level = get_level_of_case(case)
            ground_truth_input = get_ground_truth_input(case, level)
            ground_truth_output = get_ground_truth_output(case, level)
            code_content = get_code_content(case, level)
            try:
                generated_input = get_generated_input(model, level, case)
            except FileNotFoundError as e:
                print(f"{case}-{level}-{model} Not generated Input file")
                continue
            try:
                output_of_generated_input = run_avatar_generated_input(code_content, generated_input)
            except subprocess.TimeoutExpired as e:
                print(f"{case}-{level}-{model} encountered an Timeout error")
                continue
            except subprocess.CalledProcessError as e:
                print(f"{case}-{level}-{model}-{generated_input}-{ground_truth_input} encountered an error")
                print(e.stderr)
                continue
            if is_the_case_false_negative(output_of_generated_input, ground_truth_output):
                print(
                    f"{case}-{level}-{model}-{generated_input}-{ground_truth_input}-{output_of_generated_input}-{ground_truth_output}")
                results[model][level].append(case)
                false_negatives_number += 1
    with open('./avatar_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"total number of false negatives {project_name}: {false_negatives_number}")


if __name__ == '__main__':
    detect_human_eval_false_negatives()
    detect_class_eval_false_negatives()
    detect_crux_eval_false_negatives()
    detect_avatar_false_negatives()
