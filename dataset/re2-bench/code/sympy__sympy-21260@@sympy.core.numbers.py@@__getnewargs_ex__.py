import numbers
import decimal
import fractions
import math
import re as regex
import sys
from .containers import Tuple
from .sympify import (SympifyError, converter, sympify, _convert_numpy_types, _sympify,
                      _is_numpy_instance)
from .singleton import S, Singleton
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .decorators import _sympifyit
from .cache import cacheit, clear_cache
from .logic import fuzzy_not
from sympy.core.compatibility import (as_int, HAS_GMPY, SYMPY_INTS,
    gmpy)
from sympy.core.cache import lru_cache
from .kind import NumberKind
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
    prec_to_dps)
from sympy.utilities.misc import debug, filldedent
from .parameters import global_parameters
from sympy.utilities.exceptions import SymPyDeprecationWarning
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
from mpmath.libmp.backend import MPZ
from sympy.polys.polytools import invert
from .containers import Tuple
from sympy.functions.elementary.complexes import sign
from sympy import Order
from sympy.polys import gcd
from sympy.polys import lcm
from sympy.polys import cofactors
from sympy.logic.boolalg import Boolean
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
from ..functions.elementary.exponential import log
from . import Add, Mul, Pow
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

rnd = mlib.round_nearest
_LOG2 = math.log(2)
_errdict = {"divide": False}
_floatpat = regex.compile(r"[-+]?((\d*\.\d+)|(\d+\.?))")
igcd2 = math.gcd
converter[float] = converter[decimal.Decimal] = Float
RealNumber = Float
converter[int] = Integer
oo = S.Infinity
nan = S.NaN
zoo = S.ComplexInfinity
E = S.Exp1
pi = S.Pi
I = S.ImaginaryUnit
converter[fractions.Fraction] = sympify_fractions
converter[type(mpmath.rational.mpq(1, 2))] = sympify_mpmath_mpq
converter[mpnumeric] = sympify_mpmath
converter[complex] = sympify_complex
Mul.identity = One()
Add.identity = Zero()

class Float(Number):
    __slots__ = ('_mpf_', '_prec')
    is_rational = None
    is_irrational = None
    is_number = True
    is_real = True
    is_extended_real = True
    is_Float = True
    def __getnewargs_ex__(self):
        return ((mlib.to_pickable(self._mpf_),), {'precision': self._prec})