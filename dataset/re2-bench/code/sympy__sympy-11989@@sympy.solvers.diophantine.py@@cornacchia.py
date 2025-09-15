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

def cornacchia(a, b, m):
    """
    Solves `ax^2 + by^2 = m` where `\gcd(a, b) = 1 = gcd(a, m)` and `a, b > 0`.

    Uses the algorithm due to Cornacchia. The method only finds primitive
    solutions, i.e. ones with `\gcd(x, y) = 1`. So this method can't be used to
    find the solutions of `x^2 + y^2 = 20` since the only solution to former is
    `(x, y) = (4, 2)` and it is not primitive. When `a = b`, only the
    solutions with `x \leq y` are found. For more details, see the References.

    Examples
    ========

    >>> from sympy.solvers.diophantine import cornacchia
    >>> cornacchia(2, 3, 35) # equation 2x**2 + 3y**2 = 35
    {(2, 3), (4, 1)}
    >>> cornacchia(1, 1, 25) # equation x**2 + y**2 = 25
    {(4, 3)}

    References
    ===========

    .. [1] A. Nitaj, "L'algorithme de Cornacchia"
    .. [2] Solving the diophantine equation ax**2 + by**2 = m by Cornacchia's
        method, [online], Available:
        http://www.numbertheory.org/php/cornacchia.html

    See Also
    ========
    sympy.utilities.iterables.signed_permutations
    """
    sols = set()

    a1 = igcdex(a, m)[0]
    v = sqrt_mod(-b*a1, m, all_roots=True)
    if not v:
        return None

    for t in v:
        if t < m // 2:
            continue

        u, r = t, m

        while True:
            u, r = r, u % r
            if a*r**2 < m:
                break

        m1 = m - a*r**2

        if m1 % b == 0:
            m1 = m1 // b
            s, _exact = integer_nthroot(m1, 2)
            if _exact:
                if a == b and r < s:
                    r, s = s, r
                sols.add((int(r), int(s)))

    return sols
