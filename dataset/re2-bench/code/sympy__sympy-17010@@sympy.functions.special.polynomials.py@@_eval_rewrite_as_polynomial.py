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
    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        from sympy import Sum
        # Make sure n \in N_0
        if n.is_negative:
            return exp(x) * self._eval_rewrite_as_polynomial(-n - 1, -x, **kwargs)
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        k = Dummy("k")
        kern = RisingFactorial(-n, k) / factorial(k)**2 * x**k
        return Sum(kern, (k, 0, n))