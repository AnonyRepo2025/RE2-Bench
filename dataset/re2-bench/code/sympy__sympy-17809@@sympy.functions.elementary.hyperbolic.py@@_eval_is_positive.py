from __future__ import print_function, division
from sympy.core import S, sympify, cacheit, pi, I, Rational
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError, _coeff_isneg
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.integers import floor
from sympy import pi, Eq
from sympy.logic import Or, And
from sympy.core.logic import fuzzy_or, fuzzy_and, fuzzy_bool
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



class cosh(HyperbolicFunction):
    def _eval_is_positive(self):
        arg = self.args[0]

        if arg.is_real:
            return True

        re, im = arg.as_real_imag()
        im_mod = im % (2*pi)

        if im_mod == 0:
            return True

        if re == 0:
            if im_mod < pi/2 or im_mod > 3*pi/2:
                return True
            elif im_mod >= pi/2 or im_mod <= 3*pi/2:
                return False

        return fuzzy_or([fuzzy_and([fuzzy_bool(Eq(re, 0)),
                         fuzzy_or([fuzzy_bool(im_mod < pi/2),
                                   fuzzy_bool(im_mod > 3*pi/2)])]),
                         fuzzy_bool(Eq(im_mod, 0))])