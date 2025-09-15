from __future__ import print_function, division
from sympy.core import S, Symbol, Rational, Integer, Add, Dummy
from sympy.core.compatibility import as_int, SYMPY_INTS, range
from sympy.core.cache import cacheit
from sympy.core.function import Function, expand_mul
from sympy.core.numbers import E, pi
from sympy.core.relational import LessThan, StrictGreaterThan
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.trigonometric import sin, cos, cot
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.memoization import recurrence_memo
from mpmath import bernfrac, workprec
from mpmath.libmp import ifib as _ifib
from sympy.functions.combinatorial.factorials import factorial
from sympy.core.mul import prod
from collections import defaultdict
from sympy.functions.combinatorial.factorials import binomial
from sympy.core.mul import prod
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy import Sum
from sympy import zeta
from sympy.functions.special.gamma_functions import polygamma
from sympy.functions.special.gamma_functions import polygamma
from sympy.functions.special.gamma_functions import polygamma
from sympy import Sum
from sympy import Sum
from sympy import polygamma
from sympy import polygamma
from sympy import Sum
from sympy import gamma
from sympy import polygamma, log
from sympy import gamma
from sympy import hyper
from sympy import Product
from sympy import gamma
from mpmath import mp
from sympy import Expr
from mpmath import mp
from sympy import Expr
from mpmath import mp
from sympy.core.evalf import pure_complex
from mpmath import mp
from sympy import Expr

_sym = Symbol('x')
_symbols = Function('x')
_N = -1
_ITEMS = -2
_M = slice(None, _ITEMS)

class bell(Function):
    @classmethod
    def eval(cls, n, k_sym=None, symbols=None):
        if n is S.Infinity:
            if k_sym is None:
                return S.Infinity
            else:
                raise ValueError("Bell polynomial is not defined")

        if n.is_negative or n.is_integer is False:
            raise ValueError("a non-negative integer expected")

        if n.is_Integer and n.is_nonnegative:
            if k_sym is None:
                return Integer(cls._bell(int(n)))
            elif symbols is None:
                return cls._bell_poly(int(n)).subs(_sym, k_sym)
            else:
                r = cls._bell_incomplete_poly(int(n), int(k_sym), symbols)
                return r