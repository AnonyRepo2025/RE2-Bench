import re
import types


def deserialize(value):
    if isinstance(value, str):
        import datetime
        import math
        import string
        if value.startswith('datetime.'):
            return eval(value)
        iso_datetime_regex = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
        partial_iso_datetime_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if iso_datetime_regex.match(value) or partial_iso_datetime_regex.match(value):
            return datetime.datetime.fromisoformat(value)
        if value.__contains__("math."):
            try:
                return eval(value)
            except Exception:
                pass
        if value.startswith("["):
            try:
                return eval(value)
            except Exception:
                pass
        if value.startswith('{') and value.endswith('}'):
            try:
                temp_value = eval(value)
                return deserialize(temp_value)
            except Exception:
                pass
        return value
    if isinstance(value, list):
        return [deserialize(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k == 'operation':
                import builtins
                try:
                    fn = eval(v)
                    if isinstance(fn, types.FunctionType):
                        fn._source_string = v
                    if isinstance(fn, types.BuiltinFunctionType):
                        def wrapper(*args, **kwargs):
                            return fn(*args, **kwargs)
                        wrapper._source_string = v
                        fn = wrapper
                    result[k] = fn
                except Exception:
                    result[k] = v
            else:
                result[k] = deserialize(v)
        return result
    return value
