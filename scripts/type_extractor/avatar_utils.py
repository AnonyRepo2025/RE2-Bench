import re


def get_avatar_input_type(input_content) -> str:
    if type(input_content) in (int, float, bool):
        return type(input_content).__name__
    elif isinstance(input_content, str):
        list_of_numbers_pattern = r'^-?\d+( -?\d+)+$'
        number_pattern = r'^\d+$'
        if re.fullmatch(list_of_numbers_pattern, input_content):
            number_integers = len(input_content.split())
            input_type = f'sequence of {number_integers} integers separated by space'
        elif re.fullmatch(number_pattern, input_content):
            input_type = 'single integer'
        else:
            input_type = 'single string'
        return input_type
    return type(input_content).__name__


def get_avatar_output_type(output_content) -> str:
    if type(output_content) in (bool, int, float):
        return type(output_content).__name__
    elif isinstance(output_content, str):
        float_pattern = r'^-?\d+\.\d+$'
        single_int_pattern = r'^-?\d+$'
        # todo: probably we can improve multi rows because it is printing digit by digit, so each row is not a number.
        multi_rows_an_int_pattern = r'^(-?\d+\n)*-?\d+$'
        seq_ints_pattern = r'^-?\d+( +-?\d+)*$'
        multi_rows_multi_ints_pattern = r'^(-?\d+( +-?\d+)*\n)*-?\d+( +-?\d+)*\n?$'
        if re.fullmatch(float_pattern, output_content):
            output_type = float.__name__
        elif re.fullmatch(single_int_pattern, output_content):
            output_type = int.__name__
        elif re.fullmatch(multi_rows_an_int_pattern, output_content):
            rows = output_content.strip().split('\n')
            output_type = f'{len(rows)} rows each row contains an integer'
        elif re.fullmatch(seq_ints_pattern, output_content):
            number_integers = len(output_content.split())
            output_type = f'sequence of {number_integers} integers separated by space'
        elif re.fullmatch(multi_rows_multi_ints_pattern, output_content):
            rows = [row.strip().split() for row in output_content.strip().split("\n")]
            row_counts = len(rows)
            col_counts = len(rows[0])
            output_type = f'{row_counts} rows, each row has {col_counts} integers separated by space'
        else:
            output_type = str.__name__
        return output_type
    return type(output_content).__name__
