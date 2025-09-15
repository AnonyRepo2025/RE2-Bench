import argparse

from generate_reports import ValidationOutputPaths, create_detail_csv_file, create_report_json_file
from collect_fn_reports import add_fn_reports
from collect_scores import add_scores_report
from get_case_info import get_case_index, read_case_name, CaseInfo
from load_original_code import get_code_path
from validate_non_swe_bench_results import load_result_jsons_model, load_ground_truths, validate_prediction_per_case


def preprocess_output_prediction(value):
    if not isinstance(value, dict):
        return {'output': value}
    else:
        return value


def preprocess_input_prediction(value, benchmark, ground_truth):
    if not isinstance(value, dict):
        raise Exception(f"we aren't supposed to have this {value}")
    if benchmark == 'Avatar':
        if 'input' in value.keys():
            return value
        elif value == dict():
            return value
        elif isinstance(value, dict) and len(value) == 1:
            return {'input': value[list(value.keys())[0]]}
        else:
            raise Exception(f"we aren't supposed to have this {value}")
    else:
        result = {}
        gt_keys = {'self': [], 'args': [], 'kwargs': []}
        for k, v in ground_truth.items():
            if v is None:
                gt_keys[k] = []
            else:
                keys = list(v.keys())
                gt_keys[k] = keys
        copied_value = value.copy()
        if 'self' in value.keys():
            result['self'] = value['self'] if isinstance(value['self'], dict) else {}
            copied_value.pop('self')
        if 'args' in value.keys():
            result['args'] = value['args'] if isinstance(value['args'], dict) else {}
            copied_value.pop('args')
        if 'kwargs' in value.keys():
            result['kwargs'] = value['kwargs'] if isinstance(value['kwargs'], dict) else {}
            copied_value.pop('kwargs')
        for k, v in copied_value.items():
            selected_key = None
            for base_key, base_value in gt_keys.items():
                if k in base_value:
                    selected_key = base_key
                    break
            if selected_key is None:
                selected_key = 'kwargs'
            if selected_key not in result.keys():
                result[selected_key] = {}
            result[selected_key][k] = v
        if 'self' not in result.keys():
            result['self'] = {}
        if 'args' not in result.keys():
            result['args'] = {}
        if 'kwargs' not in result.keys():
            result['kwargs'] = {}
        return result

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
        if task == 'input':
            predicted_value = preprocess_input_prediction(predicted_value, benchmark, ground_truths['inputs'])
        else:
            predicted_value = preprocess_output_prediction(predicted_value)
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


if __name__ == '__main__':
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
