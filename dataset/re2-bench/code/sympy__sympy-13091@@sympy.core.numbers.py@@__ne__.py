from __future__ import print_function, division
import decimal
import fractions
import math
import warnings
import re as regex
from collections import defaultdict
from .containers import Tuple
from .sympify import converter, sympify, _sympify, SympifyError
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

class Float(Number):
    __slots__ = ['_mpf_', '_prec']
    is_rational = None
    is_irrational = None
    is_number = True
    is_real = True
    is_Float = True
    __bool__ = __nonzero__
    __truediv__ = __div__
    __long__ = __int__
    def __ne__(self, other):
        return not self == other