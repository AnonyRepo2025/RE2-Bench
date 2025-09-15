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



class tanh(HyperbolicFunction):
    def _eval_is_real(self):
        from sympy import cos, sinh
        arg = self.args[0]
        if arg.is_real:
            return True

        re, im = arg.as_real_imag()

        # if denom = 0, tanh(arg) = zoo
        if re == 0 and im % pi == pi/2:
            return None

        # check if im is of the form n*pi/2 to make sin(2*im) = 0
        # if not, im could be a number, return False in that case
        return (im % (pi/2)).is_zero