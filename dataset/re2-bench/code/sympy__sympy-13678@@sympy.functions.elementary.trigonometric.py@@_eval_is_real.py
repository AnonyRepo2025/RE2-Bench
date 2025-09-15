from __future__ import print_function, division
from sympy.core.add import Add
from sympy.core.basic import sympify, cacheit
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.numbers import igcdex, Rational, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Wild
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh,
    coth, HyperbolicFunction, sinh, tanh)
from sympy.sets.sets import FiniteSet
from sympy.utilities.iterables import numbered_symbols
from sympy.core.compatibility import range
from sympy.core.relational import Ne
from sympy.functions.elementary.piecewise import Piecewise
from sympy.calculus import AccumBounds
from sympy import expand_mul
from sympy.functions.special.polynomials import chebyshevt, chebyshevu
from sympy import Order
from sympy.functions.special.polynomials import chebyshevt
from sympy.calculus.util import AccumBounds
from sympy.functions.special.polynomials import chebyshevt
from sympy.functions.special.polynomials import chebyshevt
from sympy import Order
from sympy.calculus.util import AccumBounds
from sympy import bernoulli
from sympy import im, re
from sympy import Order
from sympy.calculus.util import AccumBounds
from sympy import bernoulli
from sympy import Order
from sympy import im, re
from sympy.functions.combinatorial.numbers import euler
from sympy import bernoulli
from sympy.functions.special.bessel import jn
from sympy import Order
from sympy import Order
from sympy import Order
from sympy import Order
from sympy import Order
from sympy import Order
from sympy import Heaviside, im, re
from sympy import arg
from sympy.ntheory import factorint
from sympy import symmetric_poly
from sympy import symmetric_poly



class asec(InverseTrigonometricFunction):
    def _eval_is_real(self):
        x = self.args[0]
        if x.is_real is False:
            return False
        return fuzzy_or(((x - 1).is_nonnegative, (-x - 1).is_nonnegative))