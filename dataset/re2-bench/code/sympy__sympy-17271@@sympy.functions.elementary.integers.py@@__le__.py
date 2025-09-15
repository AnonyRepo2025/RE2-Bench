from __future__ import print_function, division
from sympy.core import Add, S
from sympy.core.evalf import get_integer_part, PrecisionExhausted
from sympy.core.function import Function
from sympy.core.logic import fuzzy_or
from sympy.core.numbers import Integer
from sympy.core.relational import Gt, Lt, Ge, Le, Relational
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy import im
from sympy import AccumBounds, im



class floor(RoundFunction):
    _dir = -1
    def __le__(self, other):
        if self.args[0] == other and other.is_real:
            return S.true
        if other is S.Infinity and self.is_finite:
            return S.true
        return Le(self, other, evaluate=False)