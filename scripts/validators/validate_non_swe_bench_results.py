import argparse
import json
import subprocess
from json import JSONDecodeError
from typing import List, Any

from collect_fn_reports import add_fn_reports
from collect_scores import add_scores_report
from compare_partially import partial_compare
from detect_false_negative_cases import detect_false_negative
from generate_reports import create_detail_csv_file, ValidationOutputPaths, create_report_json_file
from get_case_info import CaseInfo, get_class_name, get_case_index, read_case_name
from load_original_code import get_code_path
from deserialize import deserialize


def is_complete_correct(partial_rs):
    return 1 if partial_rs == 1 else 0


def get_fn_detector_result(rs, predicted_value, ground_truth_input, ground_truth_output, function_name,
                           case_info: CaseInfo):
    try:
        is_fn = detect_false_negative(predicted_value, ground_truth_input,
                                      function_name, ground_truth_output,
                                      case_info) if rs == 0 else False
    except subprocess.TimeoutExpired as e:
        print(f"{case_info.task}-{case_info.benchmark}-{case_info.case_name}-{case_info.case_index}- timeout")
        is_fn = False
    except subprocess.CalledProcessError as e:
        print(f"{case_info.task}-{case_info.benchmark}-{case_info.case_name}-{case_info.case_index}- error: {e.stderr}")
        is_fn = False
    return is_fn


def validate_prediction_per_case(case_info: CaseInfo, ground_truths, predicted_value):
    ground_truth_input = ground_truths['inputs']
    ground_truth_output = ground_truths['outputs']
    task = case_info.task
    benchmark = case_info.benchmark
    if task == 'input' and benchmark != 'Swebench':
        deserialized_prediction = deserialize(predicted_value)
        deserialized_gt = deserialize(ground_truth_input)
        function_name = ground_truths['function_name']
        count_and_score = {'score': 0.0, 'count': 0}
        partial_compare(deserialized_prediction, deserialized_gt, count_and_score)
        partial_rs = count_and_score['score'] / count_and_score['count']
        rs = is_complete_correct(partial_rs)
        is_fn = get_fn_detector_result(rs, deserialized_prediction, deserialized_gt, ground_truth_output,
                                       function_name,
                                       case_info)
        return {'rs': rs, 'partial_rs': partial_rs, 'is_fn': is_fn}
    elif task == 'input' and benchmark == 'Swebench':
        # todo: deserialize - currently we don't deserialize because they have numpy arrays and we handle them in partially comparing
        deserialized_predicted_input = predicted_value
        count_and_score = {'score': 0.0, 'count': 0}
        partial_compare(deserialized_predicted_input, ground_truth_input, count_and_score)
        partial_rs = count_and_score['score'] / count_and_score['count']
        rs = is_complete_correct(partial_rs)
        return {'rs': rs, 'partial_rs': partial_rs, 'is_fn': ""}
    elif task == 'output':
        deserialized_predicted_output = predicted_value
        if isinstance(predicted_value, dict) and 'output' in predicted_value.keys():
            deserialized_predicted_output = predicted_value['output']
        count_and_score = {'score': 0.0, 'count': 0}
        partial_compare(deserialized_predicted_output, ground_truth_output, count_and_score)
        partial_rs = count_and_score['score'] / count_and_score['count']
        rs = is_complete_correct(partial_rs)
        return {'rs': rs, 'partial_rs': partial_rs}
    return None


def load_result_jsons_model(model_summary_path: str) -> List[Any]:
    results = []
    with open(model_summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception as e:
                    print(f"Oh not json parsable {line}")
    return results


def load_ground_truths(benchmark, case_name, index):
    base_path = f'../../dataset/re2-bench/input-output/{benchmark}/{case_name}'
    with (open(base_path, "r", encoding="utf-8") as f):
        for i, line in enumerate(f):
            if i == index:
                try:
                    master_json = json.loads(line)
                except JSONDecodeError as e:
                    print(case_name, index)
                values = master_json['values']
                return {'inputs': values['inputs'] if 'inputs' in values.keys() else values['input'],
                        'outputs': values['return'] if 'return' in values.keys() else values['output'],
                        'function_name': master_json['name'] if 'name' in master_json.keys() else None,
                        'class_name': get_class_name(case_name, benchmark)}

    raise IndexError(f"Index {index} is out of range for {base_path}")


def validate(model_summary_path, task, output_paths):
    model_task_cases_results = load_result_jsons_model(model_summary_path)
    models_task_results = {}
    results = []
    for case_result in model_task_cases_results:
        problem_id = case_result['problem_id']
        difficulty_level = case_result['difficulty']
        case_index, benchmark = get_case_index(problem_id, difficulty_level)
        case_name = read_case_name(problem_id, benchmark)
        ground_truths = load_ground_truths(benchmark, case_name, case_index)
        code_path = get_code_path(problem_id, benchmark)
        predicted_value = case_result['prediction']
        case_info = CaseInfo(case_name, case_index, problem_id, benchmark, task, code_path)
        case_result = validate_prediction_per_case(case_info, ground_truths, predicted_value)
        result_obj = {
            'case_name': case_name,
            'benchmark': benchmark,
            'rs': case_result['rs'],
            'partial_rs': case_result['partial_rs'],
            'difficulty_level': difficulty_level,
        }
        if 'is_fn' in case_result.keys():
            result_obj['is_fn'] = case_result['is_fn']
        results.append(result_obj)

    difficulty_levels = set([results_item['difficulty_level'] for results_item in results])
    add_scores_report(results, models_task_results, difficulty_levels)
    if task == 'input':
        add_fn_reports(models_task_results, results, difficulty_levels)
    create_detail_csv_file(results, output_paths)
    create_report_json_file(models_task_results, output_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Non Swe Bench type extractor')
    parser.add_argument('--summary', required=True,
                        help='path to the summary file of the model and task.')
    parser.add_argument('--output_path', required=True,
                        help='The file path that you want the results in it. (It should be csv)')
    parser.add_argument('--task', required=True, help='Task type between input and output')
    parser.add_argument('--metadata_path', required=True, help='Path to metadata csv file')
    args = parser.parse_args()
    output_path = args.output_path
    task_type = args.task
    summary_path = args.summary
    metadata_path = args.metadata_path
    paths = ValidationOutputPaths(output_path, metadata_path)
    validate(summary_path, task_type, paths)
