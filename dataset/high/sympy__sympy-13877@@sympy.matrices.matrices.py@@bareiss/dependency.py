def _find_reasonable_pivot(col, iszerofunc=_iszero, simpfunc=_simplify):

    newly_determined = []
    col = list(col)
    if all(isinstance(x, (Float, Integer)) for x in col) and any(
            isinstance(x, Float) for x in col):
        col_abs = [abs(x) for x in col]
        max_value = max(col_abs)
        if iszerofunc(max_value):
            if max_value != 0:
                newly_determined = [(i, 0) for i, x in enumerate(col) if x != 0]
            return (None, None, False, newly_determined)
        index = col_abs.index(max_value)
        return (index, col[index], False, newly_determined)

    possible_zeros = []
    for i, x in enumerate(col):
        is_zero = iszerofunc(x)
        if is_zero == False:
            return (i, x, False, newly_determined)
        possible_zeros.append(is_zero)

    if all(possible_zeros):
        return (None, None, False, newly_determined)
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        simped = simpfunc(x)
        is_zero = iszerofunc(simped)
        if is_zero == True or is_zero == False:
            newly_determined.append((i, simped))
        if is_zero == False:
            return (i, simped, False, newly_determined)
        possible_zeros[i] = is_zero
    if all(possible_zeros):
        return (None, None, False, newly_determined)
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        if x.equals(S.Zero):
            possible_zeros[i] = True
            newly_determined.append((i, S.Zero))

    if all(possible_zeros):
        return (None, None, False, newly_determined)
    i = possible_zeros.index(None)
    return (i, col[i], True, newly_determined)