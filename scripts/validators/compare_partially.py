import ast
import numbers
import re
import types


def partial_compare(predicted_value, ground_truth_value, count_and_score: dict):
    if isinstance(predicted_value, str) and looks_like_numpy_dump(predicted_value):
        predicted_value = get_numpy(predicted_value)
    if isinstance(ground_truth_value, str) and looks_like_numpy_dump(ground_truth_value):
        ground_truth_value = get_numpy(ground_truth_value)
    if isinstance(predicted_value, str) and looks_like_set(predicted_value):
        predicted_value = parse_set(predicted_value)
    if isinstance(ground_truth_value, str) and looks_like_set(ground_truth_value):
        ground_truth_value = parse_set(ground_truth_value)
    if isinstance(predicted_value, str) and looks_like_dict(predicted_value):
        predicted_value = parse_dict(predicted_value)
    if isinstance(ground_truth_value, str) and looks_like_dict(ground_truth_value):
        ground_truth_value = parse_dict(ground_truth_value)
    if isinstance(predicted_value, str) and looks_like_dump(predicted_value):
        predicted_value = parse_dump(predicted_value)
    if isinstance(ground_truth_value, str) and looks_like_dump(ground_truth_value):
        ground_truth_value = parse_dump(ground_truth_value)
    if type(predicted_value) != type(ground_truth_value):
        if isinstance(predicted_value, str):
            try:
                value = eval(predicted_value)
                partial_compare(value, ground_truth_value, count_and_score)
                return
            except Exception as e:
                pass
        count_and_score['count'] += count_predictable_attributes(ground_truth_value)
        return
    if isinstance(predicted_value, str):
        score = string_similarity(predicted_value.strip(), ground_truth_value.strip())
        count_and_score['score'] += score
        count_and_score['count'] += 1
        return
    if isinstance(predicted_value, numbers.Number):
        score = 1 if predicted_value == ground_truth_value else 0
        count_and_score['score'] += score
        count_and_score['count'] += 1
        return
    if isinstance(predicted_value, bool):
        score = 1.0 if predicted_value == ground_truth_value else 0.0
        count_and_score['score'] += score
        count_and_score['count'] += 1
        return
    if isinstance(predicted_value, list):
        total = len(ground_truth_value)
        if total == 0:
            count_and_score['count'] += 1
            count_and_score['score'] += 1.0 if not predicted_value else 0.0
            return
        old_score = count_and_score['score']
        for i in range(min(len(predicted_value), total)):
            partial_compare(predicted_value[i], ground_truth_value[i], count_and_score)
        if len(predicted_value) == 0:
            count_and_score['count'] += total
            return
        if len(predicted_value) > len(ground_truth_value):
            count_and_score['score'] = old_score
            return
        return
    if isinstance(predicted_value, set) or isinstance(predicted_value, frozenset):
        total = len(ground_truth_value)
        if total == 0:
            count_and_score['count'] += 1
            count_and_score['score'] += 1.0 if not predicted_value else 0.0
            return
        matches = ground_truth_value & predicted_value
        count_and_score['count'] += total
        if len(predicted_value) > total:
            return
        count_and_score['score'] += len(matches)
        return
    if isinstance(predicted_value, dict):
        total = len(ground_truth_value)
        if total == 0:
            count_and_score['score'] += 1.0 if not predicted_value else 0.0
            count_and_score['count'] += 1
            return
        old_score = count_and_score['score']
        for k, v in ground_truth_value.items():
            if k == 'operation' and k in predicted_value.keys():
                if isinstance(ground_truth_value[k], types.FunctionType) and isinstance(predicted_value[k],
                                                                                        types.FunctionType):
                    partial_compare(predicted_value[k]._source_string, ground_truth_value[k]._source_string,
                                    count_and_score)
                else:
                    partial_compare(str(predicted_value[k]), str(v), count_and_score)
            elif k in predicted_value:
                partial_compare(predicted_value[k], v, count_and_score)
            else:
                count_and_score['count'] += 1

        if len(predicted_value.keys()) > len(ground_truth_value.keys()):
            count_and_score['score'] = old_score
            return
        return
    if predicted_value is None and ground_truth_value is None:
        count_and_score['count'] += 1
        count_and_score['score'] += 1
        return
    count_and_score['score'] += 1.0 if predicted_value == ground_truth_value else 0.0
    count_and_score['count'] += 1
    return


def count_predictable_attributes(value):
    if isinstance(value, (str, int, float, bool, types.NoneType, types.FunctionType)):
        return 1
    elif isinstance(value, list):
        if value == list():
            return 1
        count = 0
        for item in value:
            count += count_predictable_attributes(item)
        return count
    elif isinstance(value, dict):
        if value == dict():
            return 1
        count = 0
        for k, v in value.items():
            count += count_predictable_attributes(v)
        return count
    elif isinstance(value, (set, frozenset)):
        if value == set() or value == frozenset():
            return 1
        return len(value)
    else:
        return 1


def string_similarity(a: str, b: str) -> float:
    return 1 if a == b else 0


import re

def looks_like_dump(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return False
    return bool(re.search(r"'[^']*'|True|False", s))

def parse_dump(s: str):
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError("String must start with '[' and end with ']'")

    s = s[1:-1].strip()
    tokens = re.findall(r"'[^']*'|True|False", s)
    parsed = []
    for t in tokens:
        if t == "True":
            parsed.append(True)
        elif t == "False":
            parsed.append(False)
        else:  # quoted string
            parsed.append(t[1:-1])  # strip quotes
    return parsed


import math


def looks_like_numpy_dump(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return False
    number_pattern = r"nan|inf|[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|\d+"
    if not re.search(number_pattern, s, re.IGNORECASE):
        return False
    if "\n" in s:
        return True
    if "[" in s[1:-1] and "]" in s[1:-1]:
        return True
    return True


def get_numpy(data_str: str):
    s = data_str.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    rows = re.split(r"\]\s*\[|\n", s)
    result = []
    for row in rows:
        tokens = re.findall(r"nan|[-+]?inf|[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|\d+", row, flags=re.IGNORECASE)
        parsed = []
        for t in tokens:
            t_lower = t.lower()
            if t_lower == "nan":
                parsed.append(None)
            elif t_lower == "inf":
                parsed.append(math.inf)
            elif t_lower == "-inf":
                parsed.append(-math.inf)
            else:
                parsed.append(float(t))
        if parsed:
            result.append(parsed)
    if len(result) == 1:
        return result[0]
    return result


def looks_like_set(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if s.startswith("set(") and s.endswith(")"):
        return True
    if s.startswith("frozenset(") and s.endswith(")"):
        return True
    if s.startswith("{") and s.endswith("}"):
        try:
            val = ast.literal_eval(s)
            return isinstance(val, set)
        except Exception:
            return False

    return False


def parse_set(s: str):
    s = s.strip()
    try:
        if s.startswith("frozenset(") and s.endswith(")"):
            return eval(s)

        if s.startswith("set(") and s.endswith(")"):
            inner = s[len("set("):-1]
            return set(ast.literal_eval(inner))

        if s.startswith("{") and s.endswith("}"):
            return eval(s)
    except Exception:
        return s
    raise ValueError(f"String does not look like a set: {s}")


def looks_like_dict(s: str) -> bool:
    if s.startswith("{") and s.endswith("}"):
        try:
            value = eval(s)
            return isinstance(value, dict)
        except Exception:
            return False
    return False


def parse_dict(s: str):
    return eval(s)
