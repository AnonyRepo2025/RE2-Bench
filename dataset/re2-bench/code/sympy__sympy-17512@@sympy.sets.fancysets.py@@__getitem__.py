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
    def __getitem__(self, i):
        from sympy.functions.elementary.integers import ceiling
        ooslice = "cannot slice from the end with an infinite value"
        zerostep = "slice step cannot be zero"
        # if we had to take every other element in the following
        # oo, ..., 6, 4, 2, 0
        # we might get oo, ..., 4, 0 or oo, ..., 6, 2
        ambiguous = "cannot unambiguously re-stride from the end " + \
            "with an infinite value"
        if isinstance(i, slice):
            if self.size.is_finite:  # validates, too
                start, stop, step = i.indices(self.size)
                n = ceiling((stop - start)/step)
                if n <= 0:
                    return Range(0)
                canonical_stop = start + n*step
                end = canonical_stop - step
                ss = step*self.step
                return Range(self[start], self[end] + ss, ss)
            else:  # infinite Range
                start = i.start
                stop = i.stop
                if i.step == 0:
                    raise ValueError(zerostep)
                step = i.step or 1
                ss = step*self.step
                #---------------------
                # handle infinite on right
                #   e.g. Range(0, oo) or Range(0, -oo, -1)
                # --------------------
                if self.stop.is_infinite:
                    # start and stop are not interdependent --
                    # they only depend on step --so we use the
                    # equivalent reversed values
                    return self.reversed[
                        stop if stop is None else -stop + 1:
                        start if start is None else -start:
                        step].reversed
                #---------------------
                # handle infinite on the left
                #   e.g. Range(oo, 0, -1) or Range(-oo, 0)
                # --------------------
                # consider combinations of
                # start/stop {== None, < 0, == 0, > 0} and
                # step {< 0, > 0}
                if start is None:
                    if stop is None:
                        if step < 0:
                            return Range(self[-1], self.start, ss)
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step < 0:
                            return Range(self[-1], self[stop], ss)
                        else:  # > 0
                            return Range(self.start, self[stop], ss)
                    elif stop == 0:
                        if step > 0:
                            return Range(0)
                        else:  # < 0
                            raise ValueError(ooslice)
                    elif stop == 1:
                        if step > 0:
                            raise ValueError(ooslice)  # infinite singleton
                        else:  # < 0
                            raise ValueError(ooslice)
                    else:  # > 1
                        raise ValueError(ooslice)
                elif start < 0:
                    if stop is None:
                        if step < 0:
                            return Range(self[start], self.start, ss)
                        else:  # > 0
                            return Range(self[start], self.stop, ss)
                    elif stop < 0:
                        return Range(self[start], self[stop], ss)
                    elif stop == 0:
                        if step < 0:
                            raise ValueError(ooslice)
                        else:  # > 0
                            return Range(0)
                    elif stop > 0:
                        raise ValueError(ooslice)
                elif start == 0:
                    if stop is None:
                        if step < 0:
                            raise ValueError(ooslice)  # infinite singleton
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step > 1:
                            raise ValueError(ambiguous)
                        elif step == 1:
                            return Range(self.start, self[stop], ss)
                        else:  # < 0
                            return Range(0)
                    else:  # >= 0
                        raise ValueError(ooslice)
                elif start > 0:
                    raise ValueError(ooslice)
        else:
            if not self:
                raise IndexError('Range index out of range')
            if i == 0:
                if self.start.is_infinite:
                    raise ValueError(ooslice)
                if self.has(Symbol):
                    if (self.stop > self.start) == self.step.is_positive and self.step.is_positive is not None:
                        pass
                    else:
                        _ = self.size  # validate
                return self.start
            if i == -1:
                if self.stop.is_infinite:
                    raise ValueError(ooslice)
                n = self.stop - self.step
                if n.is_Integer or (
                        n.is_integer and (
                            (n - self.start).is_nonnegative ==
                            self.step.is_positive)):
                    return n
            _ = self.size  # validate
            rv = (self.stop if i < 0 else self.start) + i*self.step
            if rv.is_infinite:
                raise ValueError(ooslice)
            if rv < self.inf or rv > self.sup:
                raise IndexError("Range index out of range")
            return rv