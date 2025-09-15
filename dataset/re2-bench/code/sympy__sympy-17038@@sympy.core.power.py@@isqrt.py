from __future__ import print_function, division
from math import log as _log
from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (_coeff_isneg, expand_complex, expand_multinomial,
    expand_mul)
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and
from .compatibility import as_int, range
from .evaluate import global_evaluate
from sympy.utilities.iterables import sift
from mpmath.libmp import sqrtrem as mpmath_sqrtrem
from math import sqrt as _sqrt
from .add import Add
from .numbers import Integer
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
from sympy.functions.elementary.exponential import exp_polar
from sympy.assumptions.ask import ask, Q
from sympy import Abs, arg, exp, floor, im, log, re, sign
from sympy import log
from sympy import arg, exp, log, Mul
from sympy import arg, log
from sympy import exp, log, Symbol
from sympy.functions.elementary.complexes import adjoint
from sympy.functions.elementary.complexes import conjugate as c
from sympy.functions.elementary.complexes import transpose
from sympy import atan2, cos, im, re, sin
from sympy.polys.polytools import poly
from sympy import log
from sympy import exp, log, I, arg
from sympy import ceiling, collect, exp, log, O, Order, powsimp
from sympy import exp, log
from sympy import binomial
from sympy import multinomial_coefficients
from sympy.polys.polyutils import basic_from_dict
from sympy import O
from sympy.ntheory import totient
from sympy import numer, denom, log, sign, im, factor_terms



def isqrt(n):
    """Return the largest integer less than or equal to sqrt(n)."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    n = int(n)

    # Fast path: with IEEE 754 binary64 floats and a correctly-rounded
    # math.sqrt, int(math.sqrt(n)) works for any integer n satisfying 0 <= n <
    # 4503599761588224 = 2**52 + 2**27. But Python doesn't guarantee either
    # IEEE 754 format floats *or* correct rounding of math.sqrt, so check the
    # answer and fall back to the slow method if necessary.
    if n < 4503599761588224:
        s = int(_sqrt(n))
        if 0 <= n - s*s <= 2*s:
            return s

    return integer_nthroot(n, 2)[0]
