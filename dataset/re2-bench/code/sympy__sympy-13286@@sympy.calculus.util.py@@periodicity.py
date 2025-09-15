from sympy import Order, S, log, limit, lcm_list, pi, Abs
from sympy.core.basic import Basic
from sympy.core import Add, Mul, Pow
from sympy.logic.boolalg import And
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.numbers import _sympifyit, oo
from sympy.core.sympify import _sympify
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
                             Complement, EmptySet)
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.utilities import filldedent
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.solvers.solveset import solveset, _has_rational_power
from sympy.solvers.solveset import solveset
from sympy import simplify, lcm_list
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import (
        TrigonometricFunction, sin, cos, csc, sec)
from sympy.solvers.decompogen import decompogen
from sympy.core.relational import Relational
from sympy.solvers.solveset import solveset
from sympy.functions.elementary.miscellaneous import real_root
from sympy.solvers.decompogen import compogen

AccumBounds = AccumulationBounds

def periodicity(f, symbol, check=False):
    """
    Tests the given function for periodicity in the given symbol.

    Parameters
    ==========

    f : Expr.
        The concerned function.
    symbol : Symbol
        The variable for which the period is to be determined.
    check : Boolean
        The flag to verify whether the value being returned is a period or not.

    Returns
    =======

    period
        The period of the function is returned.
        `None` is returned when the function is aperiodic or has a complex period.
        The value of `0` is returned as the period of a constant function.

    Raises
    ======

    NotImplementedError
        The value of the period computed cannot be verified.


    Notes
    =====

    Currently, we do not support functions with a complex period.
    The period of functions having complex periodic values such
    as `exp`, `sinh` is evaluated to `None`.

    The value returned might not be the "fundamental" period of the given
    function i.e. it may not be the smallest periodic value of the function.

    The verification of the period through the `check` flag is not reliable
    due to internal simplification of the given expression. Hence, it is set
    to `False` by default.

    Examples
    ========
    >>> from sympy import Symbol, sin, cos, tan, exp
    >>> from sympy.calculus.util import periodicity
    >>> x = Symbol('x')
    >>> f = sin(x) + sin(2*x) + sin(3*x)
    >>> periodicity(f, x)
    2*pi
    >>> periodicity(sin(x)*cos(x), x)
    pi
    >>> periodicity(exp(tan(2*x) - 1), x)
    pi/2
    >>> periodicity(sin(4*x)**cos(2*x), x)
    pi
    >>> periodicity(exp(x), x)

    """
    from sympy import simplify, lcm_list
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.trigonometric import (
        TrigonometricFunction, sin, cos, csc, sec)
    from sympy.solvers.decompogen import decompogen
    from sympy.core.relational import Relational

    def _check(orig_f, period):
        '''Return the checked period or raise an error.'''
        new_f = orig_f.subs(symbol, symbol + period)
        if new_f.equals(orig_f):
            return period
        else:
            raise NotImplementedError(filldedent('''
                The period of the given function cannot be verified.
                When `%s` was replaced with `%s + %s` in `%s`, the result
                was `%s` which was not recognized as being the same as
                the original function.
                So either the period was wrong or the two forms were
                not recognized as being equal.
                Set check=False to obtain the value.''' %
                (symbol, symbol, period, orig_f, new_f)))

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
            # all but tan and cot might have a
            # a period that is half as large
            # so recast as sin
            arg = sin(arg.args[0])
        period = periodicity(arg, symbol)
        if period is not None and isinstance(arg, sin):
            # the argument of Abs was a trigonometric other than
            # cot or tan; test to see if the half-period
            # is valid. Abs(arg) has behaviour equivalent to
            # orig_f, so use that for test:
            orig_f = Abs(arg)
            try:
                return _check(orig_f, period/2)
            except NotImplementedError as err:
                if check:
                    raise NotImplementedError(err)
            # else let new orig_f and period be
            # checked below

    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        if base_has_sym and not expo_has_sym:
            period = periodicity(base, symbol)

        elif expo_has_sym and not base_has_sym:
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
                if g != orig_f and g != f: # Fix for issue 12620
                    period = periodicity(g, symbol)
                    if period is not None:
                        break

    if period is not None:
        if check:
            return _check(orig_f, period)
        return period

    return None
