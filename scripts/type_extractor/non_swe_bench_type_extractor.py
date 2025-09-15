import argparse
import glob
import os.path
from typing import Dict, Any
from avatar_utils import get_avatar_input_type, get_avatar_output_type
from scripts.type_extractor.eval_type_utils import extract_eval_project_value_type
from utils import save_jsonl_file, load_jsonl_file


def process_avatar_test_case(test_data: Dict[str, Any]) -> Dict[str, Any]:
    input_value = test_data["input"]
    output_value = test_data["output"]
    return {
        "types": {
            "input": get_avatar_input_type(input_value),
            "output": get_avatar_output_type(output_value),
        },
        "values": {
            "input": input_value,
            "output": output_value,
        },
    }


def extract_avatar_types(raw_files_dir_path: str):
    file_paths = glob.glob(os.path.join(raw_files_dir_path, '*.jsonl'))
    total_files = len(file_paths)
    total_tests = 0
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        try:
            test_cases = load_jsonl_file(file_path)
            processed_cases = [process_avatar_test_case(test_case) for test_case in test_cases]
            save_jsonl_file(file_path, processed_cases)
            total_tests += len(processed_cases)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    print(f"Processed {total_files} files with {total_tests} total test cases.")


def process_eval_project_test_case(test_data: Dict[str, Any]) -> Dict[str, Any]:
    function_name = test_data["name"]
    args = test_data["args"]
    kwargs = test_data["kwargs"]
    return_value = test_data["return"]
    self_props = test_data["self"]
    args_types = extract_eval_project_value_type(args)
    kwargs_types = extract_eval_project_value_type(kwargs)
    return_type = extract_eval_project_value_type(return_value)
    self_props_types = extract_eval_project_value_type(self_props)
    return {
        "name": function_name,
        "types": {
            "inputs": {
                "self": self_props_types,
                "args": args_types,
                "kwargs": kwargs_types,
            },
            "return": return_type,
        },
        "values": {
            "inputs": {
                "self": self_props,
                "args": args,
                "kwargs": kwargs,
            },
            "return": return_value,
        },
    }


def extract_eval_projects_types(raw_files_dir_path: str) -> None:
    file_paths = glob.glob(os.path.join(raw_files_dir_path, '*.jsonl'))
    total_files = len(file_paths)
    total_tests = 0
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        try:
            test_cases = load_jsonl_file(file_path)
            processed_cases = [process_eval_project_test_case(test_case) for test_case in test_cases]
            save_jsonl_file(file_path, processed_cases)
            total_tests += len(processed_cases)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    print(f"Processed {total_files} files with {total_tests} total test cases.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Non Swe Bench type extractor')
    parser.add_argument('--project', required=True,
                        help='project name (HumanEval, ClassEval, CruxEval, Avatar). We consider all other projects as an eval family project')
    parser.add_argument('--path', required=True,
                        help='Path to the non SWE-Bench dataset containing raw JSONL files produced by the data generation pipeline.')
    args = parser.parse_args()
    project = args.project
    path = args.path
    if project == 'Avatar':
        extract_avatar_types(path)
    else:
        extract_eval_projects_types(path)
