from __future__ import print_function, division
from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError, _coeff_isneg
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
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



def _peeloff_ipi(arg):
    """
    Split ARG into two parts, a "rest" and a multiple of I*pi/2.
    This assumes ARG to be an Add.
    The multiple of I*pi returned in the second position is always a Rational.

    Examples
    ========

    >>> from sympy.functions.elementary.hyperbolic import _peeloff_ipi as peel
    >>> from sympy import pi, I
    >>> from sympy.abc import x, y
    >>> peel(x + I*pi/2)
    (x, I*pi/2)
    >>> peel(x + I*2*pi/3 + I*pi*y)
    (x + I*pi*y + I*pi/6, I*pi/2)
    """
    for a in Add.make_args(arg):
        if a == S.Pi*S.ImaginaryUnit:
            K = S.One
            break
        elif a.is_Mul:
            K, p = a.as_two_terms()
            if p == S.Pi*S.ImaginaryUnit and K.is_Rational:
                break
    else:
        return arg, S.Zero

    m1 = (K % S.Half)*S.Pi*S.ImaginaryUnit
    m2 = K*S.Pi*S.ImaginaryUnit - m1
    return arg - m2, m2
