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



class coth(HyperbolicFunction):
    @classmethod
    def eval(cls, arg):
        from sympy import cot
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg is S.Zero:
                return S.ComplexInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = arg.as_coefficient(S.ImaginaryUnit)

            if i_coeff is not None:
                if _coeff_isneg(i_coeff):
                    return S.ImaginaryUnit * cot(-i_coeff)
                return -S.ImaginaryUnit * cot(i_coeff)
            else:
                if _coeff_isneg(arg):
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    cothm = coth(m)
                    if cothm is S.ComplexInfinity:
                        return coth(x)
                    else: # cothm == 0
                        return tanh(x)

            if arg.func == asinh:
                x = arg.args[0]
                return sqrt(1 + x**2)/x

            if arg.func == acosh:
                x = arg.args[0]
                return x/(sqrt(x - 1) * sqrt(x + 1))

            if arg.func == atanh:
                return 1/arg.args[0]

            if arg.func == acoth:
                return arg.args[0]