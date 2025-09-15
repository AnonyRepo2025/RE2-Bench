from sympy import Order, S, log, limit, lcm_list, pi, Abs
from sympy.core.basic import Basic
from sympy.core import Add, Mul, Pow
from sympy.logic.boolalg import And
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.numbers import _sympifyit, oo
from sympy.core.sympify import _sympify
from sympy.sets.sets import Interval, Intersection, FiniteSet, Union, Complement, EmptySet
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.utilities import filldedent
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.solvers.solveset import solveset, _has_rational_power
from sympy.solvers.solveset import solveset
from sympy import simplify, lcm_list
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import TrigonometricFunction, sin, cos, csc, sec
from sympy.solvers.decompogen import decompogen
from sympy.core.relational import Relational
from sympy.solvers.solveset import solveset
from sympy.functions.elementary.miscellaneous import real_root
from sympy.solvers.decompogen import compogen
AccumBounds = AccumulationBounds

def periodicity(f, symbol, check=False):
    from sympy import simplify, lcm_list
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.trigonometric import TrigonometricFunction, sin, cos, csc, sec
    from sympy.solvers.decompogen import decompogen
    from sympy.core.relational import Relational

    def _check(orig_f, period):
        new_f = orig_f.subs(symbol, symbol + period)
        if new_f.equals(orig_f):
            return period
        else:
            raise NotImplementedError(filldedent('\n                The period of the given function cannot be verified.\n                When `%s` was replaced with `%s + %s` in `%s`, the result\n                was `%s` which was not recognized as being the same as\n                the original function.\n                So either the period was wrong or the two forms were\n                not recognized as being equal.\n                Set check=False to obtain the value.' % (symbol, symbol, period, orig_f, new_f)))
    orig_f = f
    f = simplify(orig_f)
    period = None
    if symbol not in f.free_symbols:
        return S.Zero
    if isinstance(f, Relational):
        f = f.lhs - f.rhs
    if isinstance(f, TrigonometricFunction):
        try:
            period = f.period(symbol)
        except NotImplementedError:
            pass
    if isinstance(f, Abs):
        arg = f.args[0]
        if isinstance(arg, (sec, csc, cos)):
            arg = sin(arg.args[0])
        period = periodicity(arg, symbol)
        if period is not None and isinstance(arg, sin):
            orig_f = Abs(arg)
            try:
                return _check(orig_f, period / 2)
            except NotImplementedError as err:
                if check:
                    raise NotImplementedError(err)
    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)
        if base_has_sym and (not expo_has_sym):
            period = periodicity(base, symbol)
        elif expo_has_sym and (not base_has_sym):
            period = periodicity(expo, symbol)
        else:
            period = _periodicity(f.args, symbol)
    elif f.is_Mul:
        coeff, g = f.as_independent(symbol, as_Add=False)
        if isinstance(g, TrigonometricFunction) or coeff is not S.One:
            period = periodicity(g, symbol)
        else:
            period = _periodicity(g.args, symbol)
    elif f.is_Add:
        k, g = f.as_independent(symbol)
        if k is not S.Zero:
            return periodicity(g, symbol)
        period = _periodicity(g.args, symbol)
    elif period is None:
        from sympy.solvers.decompogen import compogen
        g_s = decompogen(f, symbol)
        num_of_gs = len(g_s)
        if num_of_gs > 1:
            for index, g in enumerate(reversed(g_s)):
                start_index = num_of_gs - 1 - index
                g = compogen(g_s[start_index:], symbol)
                if g != orig_f and g != f:
                    period = periodicity(g, symbol)
                    if period is not None:
                        break
    if period is not None:
        if check:
            return _check(orig_f, period)
        return period
    return None