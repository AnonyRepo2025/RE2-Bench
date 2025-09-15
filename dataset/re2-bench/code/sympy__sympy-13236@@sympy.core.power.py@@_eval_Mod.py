from __future__ import print_function, division
from math import log as _log
from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (_coeff_isneg, expand_complex, expand_multinomial,
    expand_mul)
from .logic import fuzzy_bool, fuzzy_not
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
from sympy import ceiling, collect, exp, log, O, Order, powsimp
from sympy import exp, log
from sympy import binomial
from sympy import multinomial_coefficients
from sympy.polys.polyutils import basic_from_dict
from sympy import O
from sympy import numer, denom, log, sign, im, factor_terms



class Pow(Expr):
    is_Pow = True
    __slots__ = ['is_commutative']
    def _eval_Mod(self, q):
        if self.exp.is_integer and self.exp.is_positive:
            if q.is_integer and self.base % q == 0:
                return S.Zero

            '''
            For unevaluated Integer power, use built-in pow modular
            exponentiation.
            '''
            if self.base.is_Integer and self.exp.is_Integer and q.is_Integer:
                return pow(int(self.base), int(self.exp), int(q))