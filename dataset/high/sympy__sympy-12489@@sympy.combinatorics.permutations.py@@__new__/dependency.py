def is_sequence(i, include=None):
    return (hasattr(i, '__getitem__') and
            iterable(i) or
            bool(include) and
            isinstance(i, include))

def iterable(i, exclude=(string_types, dict, NotIterable)):
    if hasattr(i, '_iterable'):
        return i._iterable
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True

def has_variety(seq):
    for i, s in enumerate(seq):
        if i == 0:
            sentinel = s
        else:
            if s != sentinel:
                return True
    return False

def flatten(iterable, levels=None, cls=None):
    if levels is not None:
        if not levels:
            return iterable
        elif levels > 0:
            levels -= 1
        else:
            raise ValueError(
                "expected non-negative number of levels, got %s" % levels)

    if cls is None:
        reducible = lambda x: is_sequence(x, set)
    else:
        reducible = lambda x: isinstance(x, cls)

    result = []

    for el in iterable:
        if reducible(el):
            if hasattr(el, 'args'):
                el = el.args
            result.extend(flatten(el, levels=levels, cls=cls))
        else:
            result.append(el)

    return result

def has_dups(seq):
    from sympy.core.containers import Dict
    from sympy.sets.sets import Set
    if isinstance(seq, (dict, set, Dict, Set)):
        return False
    uniq = set()
    return any(True for s in seq if s in uniq or uniq.add(s))
