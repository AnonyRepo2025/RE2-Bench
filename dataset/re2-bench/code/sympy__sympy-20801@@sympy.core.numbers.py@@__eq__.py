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
    def __new__(cls, num, dps=None, prec=None, precision=None):
        if prec is not None:
            SymPyDeprecationWarning(
                            feature="Using 'prec=XX' to denote decimal precision",
                            useinstead="'dps=XX' for decimal precision and 'precision=XX' "\
                                              "for binary precision",
                            issue=12820,
                            deprecated_since_version="1.1").warn()
            dps = prec
        del prec  # avoid using this deprecated kwarg

        if dps is not None and precision is not None:
            raise ValueError('Both decimal and binary precision supplied. '
                             'Supply only one. ')

        if isinstance(num, str):
            # Float accepts spaces as digit separators
            num = num.replace(' ', '').lower()
            # in Py 3.6
            # underscores are allowed. In anticipation of that, we ignore
            # legally placed underscores
            if '_' in num:
                parts = num.split('_')
                if not (all(parts) and
                        all(parts[i][-1].isdigit()
                            for i in range(0, len(parts), 2)) and
                        all(parts[i][0].isdigit()
                            for i in range(1, len(parts), 2))):
                    # copy Py 3.6 error
                    raise ValueError("could not convert string to float: '%s'" % num)
                num = ''.join(parts)
            if num.startswith('.') and len(num) > 1:
                num = '0' + num
            elif num.startswith('-.') and len(num) > 2:
                num = '-0.' + num[2:]
            elif num in ('inf', '+inf'):
                return S.Infinity
            elif num == '-inf':
                return S.NegativeInfinity
        elif isinstance(num, float) and num == 0:
            num = '0'
        elif isinstance(num, float) and num == float('inf'):
            return S.Infinity
        elif isinstance(num, float) and num == float('-inf'):
            return S.NegativeInfinity
        elif isinstance(num, float) and num == float('nan'):
            return S.NaN
        elif isinstance(num, (SYMPY_INTS, Integer)):
            num = str(num)
        elif num is S.Infinity:
            return num
        elif num is S.NegativeInfinity:
            return num
        elif num is S.NaN:
            return num
        elif _is_numpy_instance(num):  # support for numpy datatypes
            num = _convert_numpy_types(num)
        elif isinstance(num, mpmath.mpf):
            if precision is None:
                if dps is None:
                    precision = num.context.prec
            num = num._mpf_

        if dps is None and precision is None:
            dps = 15
            if isinstance(num, Float):
                return num
            if isinstance(num, str) and _literal_float(num):
                try:
                    Num = decimal.Decimal(num)
                except decimal.InvalidOperation:
                    pass
                else:
                    isint = '.' not in num
                    num, dps = _decimal_to_Rational_prec(Num)
                    if num.is_Integer and isint:
                        dps = max(dps, len(str(num).lstrip('-')))
                    dps = max(15, dps)
                    precision = mlib.libmpf.dps_to_prec(dps)
        elif precision == '' and dps is None or precision is None and dps == '':
            if not isinstance(num, str):
                raise ValueError('The null string can only be used when '
                'the number to Float is passed as a string or an integer.')
            ok = None
            if _literal_float(num):
                try:
                    Num = decimal.Decimal(num)
                except decimal.InvalidOperation:
                    pass
                else:
                    isint = '.' not in num
                    num, dps = _decimal_to_Rational_prec(Num)
                    if num.is_Integer and isint:
                        dps = max(dps, len(str(num).lstrip('-')))
                        precision = mlib.libmpf.dps_to_prec(dps)
                    ok = True
            if ok is None:
                raise ValueError('string-float not recognized: %s' % num)

        # decimal precision(dps) is set and maybe binary precision(precision)
        # as well.From here on binary precision is used to compute the Float.
        # Hence, if supplied use binary precision else translate from decimal
        # precision.

        if precision is None or precision == '':
            precision = mlib.libmpf.dps_to_prec(dps)

        precision = int(precision)

        if isinstance(num, float):
            _mpf_ = mlib.from_float(num, precision, rnd)
        elif isinstance(num, str):
            _mpf_ = mlib.from_str(num, precision, rnd)
        elif isinstance(num, decimal.Decimal):
            if num.is_finite():
                _mpf_ = mlib.from_str(str(num), precision, rnd)
            elif num.is_nan():
                return S.NaN
            elif num.is_infinite():
                if num > 0:
                    return S.Infinity
                return S.NegativeInfinity
            else:
                raise ValueError("unexpected decimal value %s" % str(num))
        elif isinstance(num, tuple) and len(num) in (3, 4):
            if type(num[1]) is str:
                # it's a hexadecimal (coming from a pickled object)
                # assume that it is in standard form
                num = list(num)
                # If we're loading an object pickled in Python 2 into
                # Python 3, we may need to strip a tailing 'L' because
                # of a shim for int on Python 3, see issue #13470.
                if num[1].endswith('L'):
                    num[1] = num[1][:-1]
                num[1] = MPZ(num[1], 16)
                _mpf_ = tuple(num)
            else:
                if len(num) == 4:
                    # handle normalization hack
                    return Float._new(num, precision)
                else:
                    if not all((
                            num[0] in (0, 1),
                            num[1] >= 0,
                            all(type(i) in (int, int) for i in num)
                            )):
                        raise ValueError('malformed mpf: %s' % (num,))
                    # don't compute number or else it may
                    # over/underflow
                    return Float._new(
                        (num[0], num[1], num[2], bitcount(num[1])),
                        precision)
        else:
            try:
                _mpf_ = num._as_mpf_val(precision)
            except (NotImplementedError, AttributeError):
                _mpf_ = mpmath.mpf(num, prec=precision)._mpf_

        return cls._new(_mpf_, precision, zero=False)
    def __eq__(self, other):
        from sympy.logic.boolalg import Boolean
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if isinstance(other, Boolean):
            return False
        if other.is_NumberSymbol:
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if other.is_Float:
            # comparison is exact
            # so Float(.1, 3) != Float(.1, 33)
            return self._mpf_ == other._mpf_
        if other.is_Rational:
            return other.__eq__(self)
        if other.is_Number:
            # numbers should compare at the same precision;
            # all _as_mpf_val routines should be sure to abide
            # by the request to change the prec if necessary; if
            # they don't, the equality test will fail since it compares
            # the mpf tuples
            ompf = other._as_mpf_val(self._prec)
            return bool(mlib.mpf_eq(self._mpf_, ompf))
        if not self:
            return not other
        return False    # Float != non-Number