import json
import sys
from typing import Any

def get_type_from_value(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, complex):
        return "complex"
    elif isinstance(value, str):
        return "str"
    elif isinstance(value, list):
        # Heuristic: If the list has exactly 2 elements of different types, treat it as a tuple
        if len(value) == 2:
            first_type = get_type_from_value(value[0])
            second_type = get_type_from_value(value[1])
            if first_type != second_type:
                return f"Tuple[{first_type}, {second_type}]"
        # Default: it's a general list
        types = {get_type_from_value(v) for v in value}
        if len(types) == 1:
            return f"List[{next(iter(types))}]"
        return f"List[Union[{', '.join(sorted(types))}]]"
    elif isinstance(value, set):
        types = {get_type_from_value(v) for v in value}
        return f"Set[{', '.join(sorted(types)) if types else 'Any'}]"
    elif isinstance(value, frozenset):
        types = {get_type_from_value(v) for v in value}
        return f"FrozenSet[{', '.join(sorted(types)) if types else 'Any'}]"
    elif isinstance(value, tuple):
        return f"Tuple[{', '.join(get_type_from_value(v) for v in value)}]"
    elif isinstance(value, dict):
        keys = {get_type_from_value(k) for k in value.keys()}
        vals = {get_type_from_value(v) for v in value.values()}
        return f"Dict[{', '.join(sorted(keys))}, {', '.join(sorted(vals))}]"
    elif value is None:
        return "None"
    return type(value).__name__

def annotate_json(value: Any) -> Any:
    value_type = get_type_from_value(value)

    if isinstance(value, dict):
        annotated = {k: annotate_json(v) for k, v in value.items()}
        return {"type": value_type, "value": annotated}
    elif isinstance(value, list):
        annotated = [annotate_json(item) for item in value]
        return {"type": value_type, "value": annotated}
    elif isinstance(value, tuple):
        annotated = [annotate_json(item) for item in value]
        return {"type": value_type, "value": annotated}
    elif isinstance(value, set) or isinstance(value, frozenset):
        annotated = [annotate_json(item) for item in value]
        return {"type": value_type, "value": annotated}
    else:
        return {"type": value_type, "value": value}

def analyze_dynamic_json(io_json_path: str):
    with open(io_json_path + "/input-output.json", 'r') as f:
        raw_data = json.load(f)

    annotated_data = {key: annotate_json(value) for key, value in raw_data.items()}

    with open(io_json_path + "/input-output-annotated.json", "w") as f:
        json.dump(annotated_data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dynamic_type_merge.py <io_json_path>")
        sys.exit(1)

    _, io_json_path = sys.argv
    analyze_dynamic_json(io_json_path)

