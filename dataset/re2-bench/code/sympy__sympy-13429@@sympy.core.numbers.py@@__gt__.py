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

class Integer(Rational):
    q = 1
    is_integer = True
    is_number = True
    is_Integer = True
    __slots__ = ['p']
    __long__ = __int__
    @int_trace
    def __new__(cls, i):
        if isinstance(i, string_types):
            i = i.replace(' ', '')
        # whereas we cannot, in general, make a Rational from an
        # arbitrary expression, we can make an Integer unambiguously
        # (except when a non-integer expression happens to round to
        # an integer). So we proceed by taking int() of the input and
        # let the int routines determine whether the expression can
        # be made into an int or whether an error should be raised.
        try:
            ival = int(i)
        except TypeError:
            raise TypeError(
                'Integer can only work with integer expressions.')
        try:
            return _intcache[ival]
        except KeyError:
            # We only work with well-behaved integer types. This converts, for
            # example, numpy.int32 instances.
            obj = Expr.__new__(cls)
            obj.p = ival

            _intcache[ival] = obj
            return obj
    def __mul__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p*other)
            elif isinstance(other, Integer):
                return Integer(self.p*other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.p, other.q, igcd(self.p, other.q))
            return Rational.__mul__(self, other)
        return Rational.__mul__(self, other)
    def __eq__(self, other):
        if isinstance(other, integer_types):
            return (self.p == other)
        elif isinstance(other, Integer):
            return (self.p == other.p)
        return Rational.__eq__(self, other)
    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if other.is_Integer:
            return _sympify(self.p > other.p)
        return Rational.__gt__(self, other)
    def __hash__(self):
        return hash(self.p)