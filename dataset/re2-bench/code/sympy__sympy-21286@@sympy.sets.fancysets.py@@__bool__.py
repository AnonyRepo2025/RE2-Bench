from functools import reduce
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
from sympy.core.numbers import oo
from sympy.core.relational import Eq, is_eq
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And, Or
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
from sympy.core.mod import Mod
from sympy.functions import arg, Abs
from sympy.core.containers import Tuple
from sympy.functions.elementary.trigonometric import sin, cos
from sympy import floor

converter[range] = lambda r: Range(r.start, r.stop, r.step)

class Range(Set):
    is_iterable = True
    start = property(lambda self: self.args[0])
    stop = property(lambda self: self.args[1])
    step = property(lambda self: self.args[2])
    def __bool__(self):
        # this only distinguishes between definite null range
        # and non-null/unknown null; getting True doesn't mean
        # that it actually is not null
        b = is_eq(self.start, self.stop)
        if b is None:
            raise ValueError('cannot tell if Range is null or not')
        return not bool(b)