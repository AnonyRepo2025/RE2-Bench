def add_fn_reports(models_task_results, results, difficulty_levels):
    get_difficult_level_based_sum_fn_and_rs(models_task_results, results, difficulty_levels)
    models_task_results['total_rs_fn'] = get_total_sum_fn_and_rs(results)
    get_difficulty_level_based_sum_fn_and_partial_rs(difficulty_levels, models_task_results, results)
    models_task_results['total_partial_rs_fn'] = get_total_sum_fn_and_partial_rs(results)
    models_task_results['#fn'] = get_fn_counts(results)


def get_per_difficulty_level_sum_fn_and_rs(results, difficulty_level):
    count = 0
    total_score = 0.0
    for result in results:
        if result['difficulty_level'] == difficulty_level:
            count += 1
            total_score += 1 if result['is_fn'] == True else result['rs']
    if count != 0:
        total_score /= count
    return total_score


def get_difficult_level_based_sum_fn_and_rs(models_task_results, results, difficulty_levels):
    for difficulty_level in difficulty_levels:
        models_task_results[f'{difficulty_level}_rs_fn'] = get_per_difficulty_level_sum_fn_and_rs(results, difficulty_level)


def get_total_sum_fn_and_rs(results):
    total_rs = 0.0
    for result in results:
        total_rs += result['rs'] if result['is_fn'] != True else 1
    return total_rs / len(results)


def get_per_difficulty_level_sum_fn_and_partial_rs(results, difficulty_level):
    count = 0
    total_score = 0.0
    for result in results:
        if result['difficulty_level'] == difficulty_level:
            count += 1
            total_score += 1 if result['is_fn'] else result['partial_rs']
    if count != 0:
        total_score /= count
    return total_score


def get_difficulty_level_based_sum_fn_and_partial_rs(difficulty_levels, models_task_results, results):
    for difficulty_level in difficulty_levels:
        models_task_results[f'{difficulty_level}_partial_rs_fn'] = get_per_difficulty_level_sum_fn_and_partial_rs(
            results, difficulty_level
        )


def get_total_sum_fn_and_partial_rs(results):
    total_partial_rs = 0.0
    for result in results:
        total_partial_rs += result['partial_rs'] if result['is_fn'] != True else 1
    return total_partial_rs / len(results)


def get_fn_counts(results):
    fn_counts = 0
    for result in results:
        if 'is_fn' in result.keys() and result['is_fn'] == True:
            fn_counts += 1
    return fn_counts
