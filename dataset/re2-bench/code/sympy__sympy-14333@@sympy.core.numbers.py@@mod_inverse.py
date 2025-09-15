from __future__ import print_function, division
import decimal
import fractions
import math
import warnings
import re as regex
from collections import defaultdict
from .containers import Tuple
from .sympify import converter, sympify, _sympify, SympifyError, _convert_numpy_types
from .singleton import S, Singleton
from .expr import Expr, AtomicExpr
from .decorators import _sympifyit
from .cache import cacheit, clear_cache
from .logic import fuzzy_not
from sympy.core.compatibility import (
    as_int, integer_types, long, string_types, with_metaclass, HAS_GMPY,
    SYMPY_INTS, int_info)
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero as _mpf_zero, _normalize as mpf_normalize,
    prec_to_dps)
from sympy.utilities.misc import debug, filldedent
from .evaluate import global_evaluate
from sympy.utilities.exceptions import SymPyDeprecationWarning
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
from mpmath.libmp.backend import MPZ
from math import gcd as igcd2
import os
import atexit
from sympy.polys.polytools import invert
from .containers import Tuple
from sympy.functions.elementary.complexes import sign
from sympy import Order
from sympy.polys import gcd
from sympy.polys import lcm
from sympy.polys import cofactors
import sage.all as sage
from sympy.ntheory import factorrat
import sage.all as sage
from .containers import Tuple
from .containers import Tuple
from sympy import perfect_power
from sympy.ntheory import isprime
from sympy import Poly
from sympy.polys.polyclasses import ANP, DMP
from sympy.polys.numberfields import minimal_polynomial
from sympy.core.symbol import Symbol
from sympy import Dummy, Poly, PurePoly
from sympy import Poly
from sympy.polys import CRootOf, minpoly
from sympy.functions import re
import sage.all as sage
import sage.all as sage
import sage.all as sage
import sage.all as sage
from sympy import exp
from sympy import sin
from sympy import cos
import sage.all as sage
import sage.all as sage
from sympy import sqrt
import sage.all as sage
import sage.all as sage
import sage.all as sage
import sage.all as sage
import gmpy2 as gmpy
import gmpy

rnd = mlib.round_nearest
_LOG2 = math.log(2)
_errdict = {"divide": False}
_gcdcache = {}
BIGBITS = 5000
converter[float] = converter[decimal.Decimal] = Float
RealNumber = Float
_intcache = {}
_intcache_hits = 0
_intcache_misses = 0
oo = S.Infinity
nan = S.NaN
zoo = S.ComplexInfinity
E = S.Exp1
pi = S.Pi
I = S.ImaginaryUnit
converter[fractions.Fraction] = sympify_fractions
converter[mpnumeric] = sympify_mpmath
converter[complex] = sympify_complex
_intcache[0] = S.Zero
_intcache[1] = S.One
_intcache[-1] = S.NegativeOne
Mul.identity = One()
Add.identity = Zero()

def mod_inverse(a, m):
    """
    Return the number c such that, (a * c) = 1 (mod m)
    where c has the same sign as m. If no such value exists,
    a ValueError is raised.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.core.numbers import mod_inverse

    Suppose we wish to find multiplicative inverse x of
    3 modulo 11. This is the same as finding x such
    that 3 * x = 1 (mod 11). One value of x that satisfies
    this congruence is 4. Because 3 * 4 = 12 and 12 = 1 (mod 11).
    This is the value return by mod_inverse:

    >>> mod_inverse(3, 11)
    4
    >>> mod_inverse(-3, 11)
    7

    When there is a common factor between the numerators of
    ``a`` and ``m`` the inverse does not exist:

    >>> mod_inverse(2, 4)
    Traceback (most recent call last):
    ...
    ValueError: inverse of 2 mod 4 does not exist

    >>> mod_inverse(S(2)/7, S(5)/2)
    7/2

    References
    ==========
    - https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
    - https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    """
    c = None
    try:
        a, m = as_int(a), as_int(m)
        if m != 1 and m != -1:
            x, y, g = igcdex(a, m)
            if g == 1:
                c = x % m
    except ValueError:
        a, m = sympify(a), sympify(m)
        if not (a.is_number and m.is_number):
            raise TypeError(filldedent('''
                Expected numbers for arguments; symbolic `mod_inverse`
                is not implemented
                but symbolic expressions can be handled with the
                similar function,
                sympy.polys.polytools.invert'''))
        big = (m > 1)
        if not (big is S.true or big is S.false):
            raise ValueError('m > 1 did not evaluate; try to simplify %s' % m)
        elif big:
            c = 1/a
    if c is None:
        raise ValueError('inverse of %s (mod %s) does not exist' % (a, m))
    return c
