from __future__ import print_function, division
from sympy.core.add import Add
from sympy.core.compatibility import as_int, is_sequence, range
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
    divisors, factorint, multiplicity, perfect_power)
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solvers import check_assumptions
from sympy.solvers.solveset import solveset_real
from sympy.utilities import default_sort_key, numbered_symbols
from sympy.utilities.misc import filldedent
from sympy.utilities.iterables import (
        subsets, permute_signs, signed_permutations)
from sympy.core.function import count_ops
from sympy.ntheory.continued_fraction import continued_fraction_periodic
from sympy.simplify.simplify import clear_coefficients
from sympy.utilities.iterables import ordered_partitions

__all__ = ['diophantine', 'classify_diop']
diop_known = {
    "binary_quadratic",
    "cubic_thue",
    "general_pythagorean",
    "general_sum_of_even_powers",
    "general_sum_of_squares",
    "homogeneous_general_quadratic",
    "homogeneous_ternary_quadratic",
    "homogeneous_ternary_quadratic_normal",
    "inhomogeneous_general_quadratic",
    "inhomogeneous_ternary_quadratic",
    "linear",
    "univariate"}
classify_diop.func_doc = '''
    Helper routine used by diop_solve() to find information about ``eq``.

    Returns a tuple containing the type of the diophantine equation
    along with the variables (free symbols) and their coefficients.
    Variables are returned as a list and coefficients are returned
    as a dict with the key being the respective term and the constant
    term is keyed to 1. The type is one of the following:

    * %s

    Usage
    =====

    ``classify_diop(eq)``: Return variables, coefficients and type of the
    ``eq``.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``_dict`` is for internal use: when True (default) a dict is returned,
    otherwise a defaultdict which supplies 0 for missing keys is returned.

    Examples
    ========

    >>> from sympy.solvers.diophantine import classify_diop
    >>> from sympy.abc import x, y, z, w, t
    >>> classify_diop(4*x + 6*y - 4)
    ([x, y], {1: -4, x: 4, y: 6}, 'linear')
    >>> classify_diop(x + 3*y -4*z + 5)
    ([x, y, z], {1: 5, x: 1, y: 3, z: -4}, 'linear')
    >>> classify_diop(x**2 + y**2 - x*y + x + 5)
    ([x, y], {1: 5, x: 1, x**2: 1, y**2: 1, x*y: -1}, 'binary_quadratic')
    ''' % ('\n    * '.join(sorted(diop_known)))
sum_of_powers = power_representation

def diop_solve(eq, param=symbols("t", integer=True)):
    """
    Solves the diophantine equation ``eq``.

    Unlike ``diophantine()``, factoring of ``eq`` is not attempted. Uses
    ``classify_diop()`` to determine the type of the equation and calls
    the appropriate solver function.

    Usage
    =====

    ``diop_solve(eq, t)``: Solve diophantine equation, ``eq`` using ``t``
    as a parameter if needed.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``t`` is a parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine import diop_solve
    >>> from sympy.abc import x, y, z, w
    >>> diop_solve(2*x + 3*y - 5)
    (3*t_0 - 5, -2*t_0 + 5)
    >>> diop_solve(4*x + 3*y - 4*z + 5)
    (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)
    >>> diop_solve(x + 3*y - 4*z + w - 6)
    (t_0, t_0 + t_1, 6*t_0 + 5*t_1 + 4*t_2 - 6, 5*t_0 + 4*t_1 + 3*t_2 - 6)
    >>> diop_solve(x**2 + y**2 - 5)
    {(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)}


    See Also
    ========

    diophantine()
    """
    var, coeff, eq_type = classify_diop(eq, _dict=False)

    if eq_type == "linear":
        return _diop_linear(var, coeff, param)

    elif eq_type == "binary_quadratic":
        return _diop_quadratic(var, coeff, param)

    elif eq_type == "homogeneous_ternary_quadratic":
        x_0, y_0, z_0 = _diop_ternary_quadratic(var, coeff)
        return _parametrize_ternary_quadratic(
            (x_0, y_0, z_0), var, coeff)

    elif eq_type == "homogeneous_ternary_quadratic_normal":
        x_0, y_0, z_0 = _diop_ternary_quadratic_normal(var, coeff)
        return _parametrize_ternary_quadratic(
            (x_0, y_0, z_0), var, coeff)

    elif eq_type == "general_pythagorean":
        return _diop_general_pythagorean(var, coeff, param)

    elif eq_type == "univariate":
        return set([(int(i),) for i in solveset_real(
            eq, var[0]).intersect(S.Integers)])

    elif eq_type == "general_sum_of_squares":
        return _diop_general_sum_of_squares(var, -int(coeff[1]), limit=S.Infinity)

    elif eq_type == "general_sum_of_even_powers":
        for k in coeff.keys():
            if k.is_Pow and coeff[k]:
                p = k.exp
        return _diop_general_sum_of_even_powers(var, p, -int(coeff[1]), limit=S.Infinity)

    if eq_type is not None and eq_type not in diop_known:
            raise ValueError(filldedent('''
    Alhough this type of equation was identified, it is not yet
    handled. It should, however, be listed in `diop_known` at the
    top of this file. Developers should see comments at the end of
    `classify_diop`.
            '''))  # pragma: no cover
    else:
        raise NotImplementedError(
            'No solver has been written for %s.' % eq_type)
