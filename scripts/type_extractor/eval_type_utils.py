from utils import infer_type_from_value
import re


def infer_eval_iterable_fun(value, type_name):
    if not value:
        return type_name
    elem_types = [infer_type_from_value(elt, infer_str_fun=infer_eval_str_fun, infer_dict_fun=infer_eval_dict_fun,
                                        infer_iterable_fun=infer_eval_iterable_fun) for elt in value]
    base_type = elem_types[0]
    if all(element_type == base_type for element_type in elem_types):
        return f'{type_name} of {len(elem_types)} {base_type}'
    if any(isinstance(element_type, dict) for element_type in elem_types):
        return elem_types
    return f'{type_name} of different types containing {len(elem_types)} elements {list(dict.fromkeys(elem_types))}'


def infer_eval_dict_fun(keys, values):
    if not keys:
        return dict.__name__
    types = {}
    for k, v in zip(keys, values):
        types[k] = infer_type_from_value(v, infer_str_fun=infer_eval_str_fun, infer_dict_fun=infer_eval_dict_fun,
                                         infer_iterable_fun=infer_eval_iterable_fun)
    return types


def infer_eval_str_fun(value: str):
    if value.startswith("<class"):
        return type.__name__
    elif value == 'set()':
        return set.__name__
    object_pattern = r"^<(\w+) object"
    object_detecting_match = re.match(object_pattern, value)
    if object_detecting_match:
        class_name = object_detecting_match.group(1)
        return class_name
    set_pattern = r"^\{\s*\w+\s*,\s*\w+\s*\}$"
    set_detecting_match = re.match(set_pattern, value)
    if set_detecting_match:
        return set.__name__
    return str.__name__


def extract_eval_project_value_type(value):
    return infer_type_from_value(value, infer_str_fun=infer_eval_str_fun, infer_dict_fun=infer_eval_dict_fun,
                                 infer_iterable_fun=infer_eval_iterable_fun)
