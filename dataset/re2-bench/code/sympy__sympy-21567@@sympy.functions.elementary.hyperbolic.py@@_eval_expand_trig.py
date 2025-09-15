from sympy.core.logic import FuzzyBool
from sympy.core import S, sympify, cacheit, pi, I, Rational
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError, _coeff_isneg
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.integers import floor
from sympy.core.logic import fuzzy_or, fuzzy_and
from sympy import sin
from sympy import cos, sin
from sympy import Order
from sympy import cos
from sympy import cos, sin
from sympy import Order
from sympy import tan
from sympy import bernoulli
from sympy import cos, sin
from sympy import Order
from sympy import sinh, cos
from sympy import cot
from sympy import bernoulli
from sympy import cos, sin
from sympy import Order
from sympy import bernoulli
import sage.all as sage
from sympy.functions.combinatorial.numbers import euler
import sage.all as sage
from sympy import asin
from sympy import Order
from sympy import Order
from sympy import atan
from sympy import Order
from sympy import acot
from sympy import Order
from sympy import symmetric_poly
from sympy import symmetric_poly
from sympy.calculus.util import AccumBounds
from sympy.functions.combinatorial.numbers import nC
from sympy import binomial
from sympy.functions.elementary.complexes import Abs
from sympy.calculus.util import AccumBounds



class ReciprocalHyperbolicFunction(HyperbolicFunction):
    _reciprocal_of = None
    _is_even = None  # type: FuzzyBool
    _is_odd = None  # type: FuzzyBool
    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal("_eval_expand_trig", **hints)