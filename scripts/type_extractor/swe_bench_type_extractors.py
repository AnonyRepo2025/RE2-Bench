import argparse
import glob
import os.path
from typing import Dict, Any

from swe_type_utils import extract_swe_type_from_value
from utils import load_jsonl_file, save_jsonl_file


def process_test_case(test_data: Dict[str, Any]) -> Dict[str, Any]:
    excluded_keys = ('name', 'return')
    inputs_values = {k: v for k, v in test_data.items() if k not in excluded_keys}
    input_types = {k: extract_swe_type_from_value(v) for k, v in inputs_values.items()}
    return_type = extract_swe_type_from_value(test_data["return"])
    processed = {
        "types": {"inputs": input_types, "return": return_type},
        "values": {"inputs": inputs_values, "return": test_data["return"]},
    }
    if "name" in test_data:
        processed["name"] = test_data["name"]
    return processed


def extract_swe_bench_types(swe_bench_jsonl_dir: str) -> None:
    file_paths = glob.glob(os.path.join(swe_bench_jsonl_dir, '*.jsonl'))
    total_files = len(file_paths)
    total_tests = 0
    for file_path in file_paths:
        test_cases = load_jsonl_file(file_path)
        processed_cases = [process_test_case(test) for test in test_cases]
        save_jsonl_file(file_path, processed_cases)
        total_tests += len(processed_cases)
    print(f"Processed {total_files} files with {total_tests} total test cases.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract SWE-Bench types')
    parser.add_argument('--path', required=True,
                        help='Path to the SWE-bench dataset containing raw JSONL files produced by the data generation pipeline.')
    args = parser.parse_args()
    extract_swe_bench_types(args.path)
