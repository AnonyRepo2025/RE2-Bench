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



class Pow(Expr):
    is_Pow = True
    __slots__ = ['is_commutative']
    @cacheit
    def __new__(cls, b, e, evaluate=None):
        if evaluate is None:
            evaluate = global_evaluate[0]
        from sympy.functions.elementary.exponential import exp_polar

        b = _sympify(b)
        e = _sympify(e)
        if evaluate:
            if e is S.ComplexInfinity:
                return S.NaN
            if e is S.Zero:
                return S.One
            elif e is S.One:
                return b
            # Only perform autosimplification if exponent or base is a Symbol or number
            elif (b.is_Symbol or b.is_number) and (e.is_Symbol or e.is_number) and\
                e.is_integer and _coeff_isneg(b):
                if e.is_even:
                    b = -b
                elif e.is_odd:
                    return -Pow(-b, e)
            if S.NaN in (b, e):  # XXX S.NaN**x -> S.NaN under assumption that x != 0
                return S.NaN
            elif b is S.One:
                if abs(e).is_infinite:
                    return S.NaN
                return S.One
            else:
                # recognize base as E
                if not e.is_Atom and b is not S.Exp1 and not isinstance(b, exp_polar):
                    from sympy import numer, denom, log, sign, im, factor_terms
                    c, ex = factor_terms(e, sign=False).as_coeff_Mul()
                    den = denom(ex)
                    if isinstance(den, log) and den.args[0] == b:
                        return S.Exp1**(c*numer(ex))
                    elif den.is_Add:
                        s = sign(im(b))
                        if s.is_Number and s and den == \
                                log(-factor_terms(b, sign=False)) + s*S.ImaginaryUnit*S.Pi:
                            return S.Exp1**(c*numer(ex))

                obj = b._eval_power(e)
                if obj is not None:
                    return obj
        obj = Expr.__new__(cls, b, e)
        obj = cls._exec_constructor_postprocessors(obj)
        if not isinstance(obj, Pow):
            return obj
        obj.is_commutative = (b.is_commutative and e.is_commutative)
        return obj
    @property
    def base(self):
        return self._args[0]
    @property
    def exp(self):
        return self._args[1]
    def as_base_exp(self):

        b, e = self.args
        if b.is_Rational and b.p == 1 and b.q != 1:
            return Integer(b.q), -e
        return b, e
    def _eval_is_rational(self):
        if (self.exp.is_integer and self.base.is_rational
                and fuzzy_not(fuzzy_and([self.exp.is_negative, self.base.is_zero]))):
            return True
        p = self.func(*self.as_base_exp())  # in case it's unevaluated
        if not p.is_Pow:
            return p.is_rational
        b, e = p.as_base_exp()
        if e.is_Rational and b.is_Rational:
            # we didn't check that e is not an Integer
            # because Rational**Integer autosimplifies
            return False
        if e.is_integer:
            if b.is_rational:
                if fuzzy_not(b.is_zero) or e.is_nonnegative:
                    return True
                if b == e:  # always rational, even for 0**0
                    return True
            elif b.is_irrational:
                return e.is_zero