from collections import defaultdict


def add_scores_report(results, models_task_results, difficulty_levels):
    get_count_difficulty_based(results, models_task_results)
    models_task_results['total_rs'] = get_total_rs(results)
    models_task_results['total_partial_rs'] = get_total_partial_rs(results)
    get_difficult_level_based_rs(difficulty_levels, models_task_results, results)
    get_difficulty_level_based_partial_rs(difficulty_levels, models_task_results, results)


def get_total_partial_rs(results):
    total_partial_rs = 0.0
    for result in results:
        total_partial_rs += result['partial_rs']
    total_partial_rs /= len(results)
    return total_partial_rs


def get_total_rs(results):
    total_rs = 0
    for result in results:
        total_rs += result['rs']
    total_rs /= len(results)
    return total_rs


def get_difficult_level_based_rs(difficulty_levels, models_task_results, results):
    for difficulty_level in difficulty_levels:
        models_task_results[f"{difficulty_level}_rs"] = get_per_difficulty_level_rs(results, difficulty_level)


def get_per_difficulty_level_rs(results, difficulty_level):
    count = 0
    total_rs = 0.0
    for result in results:
        if result['difficulty_level'] == difficulty_level:
            count += 1
            total_rs += result['rs']
    if count != 0:
        total_rs /= count
    return total_rs


def get_difficulty_level_based_partial_rs(difficulty_levels, models_task_results, results):
    for difficulty_level in difficulty_levels:
        models_task_results[f"{difficulty_level}_partial_rs"] = get_per_difficulty_partial_rs(results, difficulty_level)


def get_per_difficulty_partial_rs(results, difficulty_level):
    count = 0
    total_partial_rs = 0.0
    for result in results:
        if result['difficulty_level'] == difficulty_level:
            count += 1
            total_partial_rs += result['partial_rs']
    if count != 0:
        total_partial_rs /= count

    return total_partial_rs


def get_count_difficulty_based(results, models_task_results):
    per_difficulty_number_cases = defaultdict(int)
    for result in results:
        per_difficulty_number_cases[result['difficulty_level']] += 1
    for difficulty_level_name in per_difficulty_number_cases.keys():
        models_task_results[f'{difficulty_level_name}_number'] = per_difficulty_number_cases[difficulty_level_name]
