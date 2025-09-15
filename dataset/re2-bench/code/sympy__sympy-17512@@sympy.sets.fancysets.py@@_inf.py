from __future__ import print_function, division
from sympy.core.basic import Basic
from sympy.core.compatibility import as_int, with_metaclass, range, PY3
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import oo, Integer
from sympy.core.logic import fuzzy_or
from sympy.core.relational import Eq
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And
from sympy.sets.sets import (Set, Interval, Union, FiniteSet,
    ProductSet, Intersection)
from sympy.utilities.misc import filldedent
from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
from sympy.core.numbers import igcd, Rational
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy.matrices import Matrix
from sympy.solvers.solveset import solveset, linsolve
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence, iterable, cartes
from sympy.sets.setexpr import SetExpr
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.integers import floor
from sympy import sin, cos
from sympy.functions import arg, Abs
from sympy.core.containers import Tuple



class Range(Set):
    is_iterable = True
    start = property(lambda self: self.args[0])
    stop = property(lambda self: self.args[1])
    step = property(lambda self: self.args[2])
    __bool__ = __nonzero__
    def __nonzero__(self):
        return self.start != self.stop
    @property
    def _inf(self):
        if not self:
            raise NotImplementedError
        if self.has(Symbol):
            if self.step.is_positive:
                return self[0]
            elif self.step.is_negative:
                return self[-1]
            _ = self.size  # validate
        if self.step > 0:
            return self.start
        else:
            return self.stop - self.step