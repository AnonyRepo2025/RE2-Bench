from __future__ import print_function, division
from collections import defaultdict
from sympy import SYMPY_DEBUG
from sympy.core import expand_power_base, sympify, Add, S, Mul, Derivative, Pow, symbols, expand_mul
from sympy.core.add import _unevaluated_Add
from sympy.core.compatibility import iterable, ordered, default_sort_key
from sympy.core.evaluate import global_evaluate
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand
from sympy.core.mul import _keep_coeff, _unevaluated_Mul
from sympy.core.numbers import Rational
from sympy.functions import exp, sqrt, log
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.simplify import signsimp
from sympy.simplify.simplify import nsimplify
from sympy.simplify.powsimp import powsimp, powdenest

expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand

def rad_rationalize(num, den):
    """
    Rationalize num/den by removing square roots in the denominator;
    num and den are sum of terms whose squares are positive rationals.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import rad_rationalize
    >>> rad_rationalize(sqrt(3), 1 + sqrt(2)/3)
    (-sqrt(3) + sqrt(6)/3, -7/9)
    """
    if not den.is_Add:
        return num, den
    g, a, b = split_surds(den)
    a = a*sqrt(g)
    num = _mexpand((a - b)*num)
    den = _mexpand(a**2 - b**2)
    return rad_rationalize(num, den)
