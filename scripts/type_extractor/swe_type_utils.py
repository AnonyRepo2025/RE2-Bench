from utils import infer_type_from_value
import numpy as np
import xarray as xr
import re
import ast


def infer_numpy_arrays(value):
    s_clean = value.replace('\n', ' ').replace('...,', '').replace("nan", "None")
    s_clean = s_clean.replace('...', '')
    s_comma = re.sub(r'(?<=[\w.])\s+(?=[\w.\-])', ', ', s_clean)
    s_comma_lists = re.sub(r'\]\s+\[', '], [', s_comma)
    numpy_array = np.array(eval(s_comma_lists))
    flat = numpy_array.flatten()
    types = set(type(x).__name__ for x in flat if x is not None)
    if not types:
        return "np.ndarray[empty]"
    type_repr = "|".join(sorted(types))
    return f"np.ndarray[{type_repr}]"


def infer_random_state(value: str):
    if value.__contains__("RandomState"):
        return "numpy.random.RandomState"
    raise Exception()


def infer_xarray_variable(value: str) -> str:
    clean_value = value.strip().strip('"').replace('\\"', '"')
    dims_match = re.search(r"\((.*?)\)", clean_value)
    if not dims_match:
        raise ValueError("Could not find dimensions in repr")
    dims_str = dims_match.group(1).strip()
    dims = []
    shape = []
    for part in dims_str.split(","):
        name, size = part.split(":")
        dims.append(name.strip())
        shape.append(int(size.strip()))
    arr_match = re.search(r"array\((.*)\)", clean_value, re.DOTALL)
    if not arr_match:
        raise ValueError("Could not find array data in repr")
    arr_str = arr_match.group(1).strip()
    data = np.array(ast.literal_eval(arr_str))
    variable = xr.Variable(dims, data)
    return f"xarray.Variable(dims={variable.dims}, shape={variable.shape}, size={variable.size})"


def infer_swe_bench_dict_value(keys, values):
    if not values:
        return "in ground truth values, it is an empty dict"
    types = {}
    for k, v in zip(keys, values):
        types[k] = infer_type_from_value(v,
                                         infer_dict_fun=infer_swe_bench_dict_value,
                                         infer_str_fun=infer_swe_bench_str_value,
                                         infer_iterable_fun=infer_swe_bench_iterable_value)
    return types


def is_there_a_dict_in_list(list_of_types: list):
    for item in list_of_types:
        if isinstance(item, dict):
            return True
        if isinstance(item, list):
            does_have_dict = is_there_a_dict_in_list(item)
            if does_have_dict:
                return True
    return False


def infer_swe_bench_iterable_value(value, type_name):
    if not value:
        return f"in ground truth values, it is an empty {type_name}"
    element_types = [infer_type_from_value(element, infer_dict_fun=infer_swe_bench_dict_value,
                                           infer_str_fun=infer_swe_bench_str_value,
                                           infer_iterable_fun=infer_swe_bench_iterable_value) for element in value]
    base_type = element_types[0]
    if all(element_type == base_type for element_type in element_types):
        return f'{type_name} of {len(element_types)} {base_type}'
    if is_there_a_dict_in_list(element_types):
        return element_types
    return f'{type_name} of different types containing {len(element_types)} elements {list(dict.fromkeys(element_types))}'


def infer_swe_bench_str_value(value):
    try:
        extracted_value_from_string = ast.literal_eval(value)
        if isinstance(extracted_value_from_string, set):
            return 'in ground truth values, it is an empty set'
        if isinstance(extracted_value_from_string, list):
            return infer_type_from_value(extracted_value_from_string,
                                         infer_dict_fun=infer_swe_bench_dict_value,
                                         infer_str_fun=infer_swe_bench_str_value,
                                         infer_iterable_fun=infer_swe_bench_iterable_value)
    except Exception:
        try:
            return infer_numpy_arrays(value)
        except Exception:
            pass
        try:
            return infer_random_state(value)
        except Exception:
            pass
        try:
            return infer_xarray_variable(value)
        except Exception:
            return str.__name__
    return str.__name__


def extract_swe_type_from_value(value):
    return infer_type_from_value(value,
                                 infer_dict_fun=infer_swe_bench_dict_value,
                                 infer_str_fun=infer_swe_bench_str_value,
                                 infer_iterable_fun=infer_swe_bench_iterable_value)
