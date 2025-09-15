from __future__ import annotations
import numbers
import decimal
import fractions
import math
import re as regex
import sys
from functools import lru_cache
from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
              _sympify, _is_numpy_instance)
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .logic import fuzzy_not
from .kind import NumberKind
from sympy.external.gmpy import SYMPY_INTS, HAS_GMPY, gmpy
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
    prec_to_dps, dps_to_prec)
from sympy.utilities.misc import as_int, debug, filldedent
from .parameters import global_parameters
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
from mpmath.libmp.backend import MPZ
from sympy.polys.polytools import invert
from sympy.functions.elementary.complexes import sign
from sympy.series.order import Order
from sympy.polys.polytools import gcd
from sympy.polys.polytools import lcm
from sympy.polys.polytools import cofactors
from sympy.logic.boolalg import Boolean
from sympy.ntheory.factor_ import factorrat
from sympy.ntheory.factor_ import perfect_power
from sympy.ntheory.primetest import isprime
from sympy.polys.polyclasses import ANP, DMP
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.polys import minpoly
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.miscellaneous import cbrt, sqrt
from .symbol import Dummy
from sympy.concrete.summations import Sum
from .power import integer_log
from sympy.polys.densetools import dup_compose
from .symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.polys.polytools import Poly
from .symbol import Dummy
from sympy.polys.numberfields.minpoly import minpoly

_LOG2 = math.log(2)
_errdict = {"divide": False}
_floatpat = regex.compile(r"[-+]?((\d*\.\d+)|(\d+\.?))")
igcd2 = math.gcd
_sympy_converter[float] = _sympy_converter[decimal.Decimal] = Float
RealNumber = Float
_sympy_converter[int] = Integer
oo = S.Infinity
nan = S.NaN
zoo = S.ComplexInfinity
E = S.Exp1
pi = S.Pi
I = S.ImaginaryUnit
_sympy_converter[fractions.Fraction] = sympify_fractions
_sympy_converter[type(mpmath.rational.mpq(1, 2))] = sympify_mpmath_mpq
_sympy_converter[mpnumeric] = sympify_mpmath
_sympy_converter[complex] = sympify_complex
Mul.identity = One()
Add.identity = Zero()
_illegal = (S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity)

class Integer(Rational):
    q = 1
    is_integer = True
    is_number = True
    is_Integer = True
    __slots__ = ()
    @_sympifyit('other', NotImplemented)
    def __floordiv__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        if isinstance(other, Integer):
            return Integer(self.p // other)
        return divmod(self, other)[0]