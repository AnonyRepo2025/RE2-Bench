import os
import subprocess
from get_case_info import CaseInfo, get_class_name
from load_original_code import load_original_code


def detect_false_negative(predicted_input, gt_input, function_name, gt_output,
                          case_info: CaseInfo):
    benchmark = case_info.benchmark
    case_name = case_info.case_name
    code_path = case_info.code_path
    original_code = load_original_code(code_path)
    if benchmark == 'Avatar' or 'input' in predicted_input.keys() or 'input' in gt_input.keys():
        if os.path.exists('./temp.py'):
            os.remove('./temp.py')
        with open('./temp.py', 'w') as f:
            f.write(original_code)
        if 'input' in predicted_input.keys():
            input_value = predicted_input['input']
        else:
            input_value = "\n"
        passed_input = input_value if isinstance(input_value, str) else str(input_value)
        output = subprocess.run(
            ['python', './temp.py'],
            text=True,
            input=passed_input,
            capture_output=True,
            timeout=15,
            check=True
        )
        os.remove('./temp.py')
        output_value = output.stdout.strip()
        if str(gt_output).strip() == output_value and str(gt_input) != passed_input:
            return True
        else:
            return False
    else:
        args = predicted_input['args'] if 'args' in predicted_input.keys() else {}
        kwargs = predicted_input['kwargs'] if 'kwargs' in predicted_input.keys() else {}
        self_values = predicted_input['self'] if 'self' in predicted_input.keys() else {}
        attached_code = open("./constructor.py", 'r').read().strip()
        class_name = get_class_name(case_name, benchmark)
        if class_name is not None:
            attached_code += '\n\n' + f'''self_instance = construct_instance("{code_path}", "{class_name}", {self_values})\nprint(self_instance.{build_call_code(function_name, args, kwargs)})'''
        else:
            attached_code += '\n\n' + f'''print({build_call_code(function_name, args, kwargs)})'''
        with open('./temp.py', 'w') as f:
            f.write(f"{original_code}\n{attached_code}")
        output = subprocess.run(
            ['python', './temp.py'],
            text=True,
            capture_output=True,
            timeout=15,
            check=True
        )
        output_value = output.stdout.strip()
        if output_value == str(gt_output).strip() and gt_input != predicted_input:
            return True
        else:
            return False


def build_call_code(function_name, args, kwargs):
    result = ''
    for k, v in args.items():
        if isinstance(v, str):
            escaped = v.encode("unicode_escape").decode("utf-8")
            if escaped.__contains__('"'):
                result += f"{k}='{escaped}',"
            else:
                result += f'{k}="{escaped}",'
        elif callable(v):
            result += f"{k}=eval('{v._source_string}')"
        else:
            result += f"{k}={v},"

    for k, v in kwargs.items():
        if isinstance(v, str):
            escaped = v.encode("unicode_escape").decode("utf-8")
            if escaped.__contains__('"'):
                result += f"{k}='{escaped}',"
            else:
                result += f'{k}="{escaped}",'
        else:
            result += f"{k}={v},"

    return f"{function_name}({result})"
