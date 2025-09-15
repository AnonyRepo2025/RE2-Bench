from functools import reduce
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And
from sympy.sets.sets import (Set, Interval, Union, FiniteSet,
    ProductSet)
from sympy.utilities.misc import filldedent
from sympy.utilities.iterables import cartes
from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
from sympy.core.numbers import igcd, Rational
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy.solvers.solveset import _solveset_multi
from sympy.sets.setexpr import SetExpr
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.integers import ceiling
from sympy.functions import arg, Abs
from sympy.core.containers import Tuple
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.core.mod import Mod

converter[range] = lambda r: Range(r.start, r.stop, r.step)

class Range(Set):
    is_iterable = True
    start = property(lambda self: self.args[0])
    stop = property(lambda self: self.args[1])
    step = property(lambda self: self.args[2])
    @property
    def size(self):
        if not self:
            return S.Zero
        dif = self.stop - self.start
        n = abs(dif // self.step)
        if not n.is_Integer:
            if n.is_infinite:
                return S.Infinity
            raise ValueError('invalid method for symbolic range')
        return n
    def __bool__(self):
        return self.start != self.stop