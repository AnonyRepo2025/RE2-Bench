from __future__ import print_function, division
from sympy.core import Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import binomial, factorial, RisingFactorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sec
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import (
    jacobi_poly,
    gegenbauer_poly,
    chebyshevt_poly,
    chebyshevu_poly,
    laguerre_poly,
    hermite_poly,
    legendre_poly
)
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum
from sympy import Sum

_x = Dummy('x')

class laguerre(OrthogonalPolynomial):
    _ortho_poly = staticmethod(laguerre_poly)
    @classmethod
    def eval(cls, n, x):
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        if not n.is_Number:
            # Symbolic result L_n(x)
            # L_{n}(-x)  --->  exp(-x) * L_{-n-1}(x)
            # L_{-n}(x)  --->  exp(x) * L_{n-1}(-x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return exp(x)*laguerre(-n - 1, -x)
            # We can evaluate for some special values of x
            if x == S.Zero:
                return S.One
            elif x == S.NegativeInfinity:
                return S.Infinity
            elif x == S.Infinity:
                return S.NegativeOne**n * S.Infinity
        else:
            if n.is_negative:
                return exp(x)*laguerre(-n - 1, -x)
            else:
                return cls._eval_at_order(n, x)