from __future__ import absolute_import, print_function, division
import numbers
import decimal
import fractions
import math
import re as regex
from .containers import Tuple
from .sympify import converter, sympify, _sympify, SympifyError, _convert_numpy_types
from .singleton import S, Singleton
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .decorators import _sympifyit
from .cache import cacheit, clear_cache
from .logic import fuzzy_not
from sympy.core.compatibility import (
    as_int, integer_types, long, string_types, with_metaclass, HAS_GMPY,
    SYMPY_INTS, int_info)
from sympy.core.cache import lru_cache
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
    prec_to_dps, fone, fnone)
from sympy.utilities.misc import debug, filldedent
from .evaluate import global_evaluate
from sympy.utilities.exceptions import SymPyDeprecationWarning
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
from mpmath.libmp.backend import MPZ
from math import gcd as igcd2
from sympy.polys.polytools import invert
from .containers import Tuple
from sympy.functions.elementary.complexes import sign
from sympy import Order
from sympy.polys import gcd
from sympy.polys import lcm
from sympy.polys import cofactors
from sympy.core.evalf import evalf
from sympy.core.numbers import prec_to_dps
import sage.all as sage
from sympy.core.power import integer_log
from sympy.ntheory import factorrat
import sage.all as sage
from .containers import Tuple
from .containers import Tuple
from sympy.ntheory.factor_ import perfect_power
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
from sympy import sqrt, cbrt
import sage.all as sage
from sympy import Sum, Dummy
import sage.all as sage
import sage.all as sage
import gmpy2 as gmpy
import gmpy

rnd = mlib.round_nearest
_LOG2 = math.log(2)
_errdict = {"divide": False}
_floatpat = regex.compile(r"[-+]?((\d*\.\d+)|(\d+\.?))")
BIGBITS = 5000
converter[float] = converter[decimal.Decimal] = Float
RealNumber = Float
oo = S.Infinity
nan = S.NaN
zoo = S.ComplexInfinity
E = S.Exp1
pi = S.Pi
I = S.ImaginaryUnit
converter[fractions.Fraction] = sympify_fractions
converter[mpnumeric] = sympify_mpmath
converter[type(mpmath.rational.mpq(1, 2))] = sympify_mpq
converter[complex] = sympify_complex
Mul.identity = One()
Add.identity = Zero()

class Float(Number):
    __slots__ = ['_mpf_', '_prec']
    is_rational = None
    is_irrational = None
    is_number = True
    is_real = True
    is_extended_real = True
    is_Float = True
    __bool__ = __nonzero__
    __truediv__ = __div__
    __long__ = __int__
    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if isinstance(other, Number) and other != 0 and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
        return Number.__div__(self, other)