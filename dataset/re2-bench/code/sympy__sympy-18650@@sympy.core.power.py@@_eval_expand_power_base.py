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
from .compatibility import as_int, HAS_GMPY, gmpy
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from mpmath.libmp import sqrtrem as mpmath_sqrtrem
from math import sqrt as _sqrt
from .add import Add
from .numbers import Integer
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
from sympy.functions.elementary.exponential import exp_polar
from sympy.core.relational import Relational
from sympy.assumptions.ask import ask, Q
from sympy import arg, exp, floor, im, log, re, sign
from sympy.ntheory import totient
from .mod import Mod
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
from sympy import numer, denom, log, sign, im, factor_terms



class Pow(Expr):
    is_Pow = True
    __slots__ = ('is_commutative',)
    @property
    def base(self):
        return self._args[0]
    @property
    def exp(self):
        return self._args[1]
    def _eval_expand_power_base(self, **hints):
        """(a*b)**n -> a**n * b**n"""
        force = hints.get('force', False)

        b = self.base
        e = self.exp
        if not b.is_Mul:
            return self

        cargs, nc = b.args_cnc(split_1=False)

        # expand each term - this is top-level-only
        # expansion but we have to watch out for things
        # that don't have an _eval_expand method
        if nc:
            nc = [i._eval_expand_power_base(**hints)
                if hasattr(i, '_eval_expand_power_base') else i
                for i in nc]

            if e.is_Integer:
                if e.is_positive:
                    rv = Mul(*nc*e)
                else:
                    rv = Mul(*[i**-1 for i in nc[::-1]]*-e)
                if cargs:
                    rv *= Mul(*cargs)**e
                return rv

            if not cargs:
                return self.func(Mul(*nc), e, evaluate=False)

            nc = [Mul(*nc)]

        # sift the commutative bases
        other, maybe_real = sift(cargs, lambda x: x.is_extended_real is False,
            binary=True)
        def pred(x):
            if x is S.ImaginaryUnit:
                return S.ImaginaryUnit
            polar = x.is_polar
            if polar:
                return True
            if polar is None:
                return fuzzy_bool(x.is_extended_nonnegative)
        sifted = sift(maybe_real, pred)
        nonneg = sifted[True]
        other += sifted[None]
        neg = sifted[False]
        imag = sifted[S.ImaginaryUnit]
        if imag:
            I = S.ImaginaryUnit
            i = len(imag) % 4
            if i == 0:
                pass
            elif i == 1:
                other.append(I)
            elif i == 2:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
            else:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
                other.append(I)
            del imag

        # bring out the bases that can be separated from the base

        if force or e.is_integer:
            # treat all commutatives the same and put nc in other
            cargs = nonneg + neg + other
            other = nc
        else:
            # this is just like what is happening automatically, except
            # that now we are doing it for an arbitrary exponent for which
            # no automatic expansion is done

            assert not e.is_Integer

            # handle negatives by making them all positive and putting
            # the residual -1 in other
            if len(neg) > 1:
                o = S.One
                if not other and neg[0].is_Number:
                    o *= neg.pop(0)
                if len(neg) % 2:
                    o = -o
                for n in neg:
                    nonneg.append(-n)
                if o is not S.One:
                    other.append(o)
            elif neg and other:
                if neg[0].is_Number and neg[0] is not S.NegativeOne:
                    other.append(S.NegativeOne)
                    nonneg.append(-neg[0])
                else:
                    other.extend(neg)
            else:
                other.extend(neg)
            del neg

            cargs = nonneg
            other += nc

        rv = S.One
        if cargs:
            if e.is_Rational:
                npow, cargs = sift(cargs, lambda x: x.is_Pow and
                    x.exp.is_Rational and x.base.is_number,
                    binary=True)
                rv = Mul(*[self.func(b.func(*b.args), e) for b in npow])
            rv *= Mul(*[self.func(b, e, evaluate=False) for b in cargs])
        if other:
            rv *= self.func(Mul(*other), e, evaluate=False)
        return rv