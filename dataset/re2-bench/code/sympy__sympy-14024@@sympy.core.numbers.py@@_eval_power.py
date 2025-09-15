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

class Integer(Rational):
    q = 1
    is_integer = True
    is_number = True
    is_Integer = True
    __slots__ = ['p']
    __long__ = __int__
    def _eval_power(self, expt):
        """
        Tries to do some simplifications on self**expt

        Returns None if no further simplifications can be done

        When exponent is a fraction (so we have for example a square root),
        we try to find a simpler representation by factoring the argument
        up to factors of 2**15, e.g.

          - sqrt(4) becomes 2
          - sqrt(-4) becomes 2*I
          - (2**(3+7)*3**(6+7))**Rational(1,7) becomes 6*18**(3/7)

        Further simplification would require a special call to factorint on
        the argument which is not done here for sake of speed.

        """
        from sympy import perfect_power

        if expt is S.Infinity:
            if self.p > S.One:
                return S.Infinity
            # cases -1, 0, 1 are done in their respective classes
            return S.Infinity + S.ImaginaryUnit*S.Infinity
        if expt is S.NegativeInfinity:
            return Rational(1, self)**S.Infinity
        if not isinstance(expt, Number):
            # simplify when expt is even
            # (-2)**k --> 2**k
            if self.is_negative and expt.is_even:
                return (-self)**expt
        if isinstance(expt, Float):
            # Rational knows how to exponentiate by a Float
            return super(Integer, self)._eval_power(expt)
        if not isinstance(expt, Rational):
            return
        if expt is S.Half and self.is_negative:
            # we extract I for this special case since everyone is doing so
            return S.ImaginaryUnit*Pow(-self, expt)
        if expt.is_negative:
            # invert base and change sign on exponent
            ne = -expt
            if self.is_negative:
                    return S.NegativeOne**expt*Rational(1, -self)**ne
            else:
                return Rational(1, self.p)**ne
        # see if base is a perfect root, sqrt(4) --> 2
        x, xexact = integer_nthroot(abs(self.p), expt.q)
        if xexact:
            # if it's a perfect root we've finished
            result = Integer(x**abs(expt.p))
            if self.is_negative:
                result *= S.NegativeOne**expt
            return result

        # The following is an algorithm where we collect perfect roots
        # from the factors of base.

        # if it's not an nth root, it still might be a perfect power
        b_pos = int(abs(self.p))
        p = perfect_power(b_pos)
        if p is not False:
            dict = {p[0]: p[1]}
        else:
            dict = Integer(b_pos).factors(limit=2**15)

        # now process the dict of factors
        out_int = 1  # integer part
        out_rad = 1  # extracted radicals
        sqr_int = 1
        sqr_gcd = 0
        sqr_dict = {}
        for prime, exponent in dict.items():
            exponent *= expt.p
            # remove multiples of expt.q: (2**12)**(1/10) -> 2*(2**2)**(1/10)
            div_e, div_m = divmod(exponent, expt.q)
            if div_e > 0:
                out_int *= prime**div_e
            if div_m > 0:
                # see if the reduced exponent shares a gcd with e.q
                # (2**2)**(1/10) -> 2**(1/5)
                g = igcd(div_m, expt.q)
                if g != 1:
                    out_rad *= Pow(prime, Rational(div_m//g, expt.q//g))
                else:
                    sqr_dict[prime] = div_m
        # identify gcd of remaining powers
        for p, ex in sqr_dict.items():
            if sqr_gcd == 0:
                sqr_gcd = ex
            else:
                sqr_gcd = igcd(sqr_gcd, ex)
                if sqr_gcd == 1:
                    break
        for k, v in sqr_dict.items():
            sqr_int *= k**(v//sqr_gcd)
        if sqr_int == b_pos and out_int == 1 and out_rad == 1:
            result = None
        else:
            result = out_int*out_rad*Pow(sqr_int, Rational(sqr_gcd, expt.q))
            if self.is_negative:
                result *= Pow(S.NegativeOne, expt)
        return result