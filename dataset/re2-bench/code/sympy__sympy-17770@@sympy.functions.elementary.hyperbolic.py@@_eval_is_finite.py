from __future__ import print_function, division
from sympy.core import S, sympify, cacheit, pi, I, Rational
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError, _coeff_isneg
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.integers import floor
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
from sympy import cos, sinh
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
from sympy.calculus.util import AccumBounds
from sympy.functions.elementary.complexes import Abs
from sympy.calculus.util import AccumBounds



class sinh(HyperbolicFunction):
    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite