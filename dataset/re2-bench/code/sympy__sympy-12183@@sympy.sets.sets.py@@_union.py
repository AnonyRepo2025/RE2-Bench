from __future__ import print_function, division
from itertools import product
from sympy.core.sympify import (_sympify, sympify, converter,
    SympifyError)
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.singleton import Singleton, S
from sympy.core.evalf import EvalfMixin
from sympy.core.numbers import Float
from sympy.core.compatibility import (iterable, with_metaclass,
    ordered, range, PY3)
from sympy.core.evaluate import global_evaluate
from sympy.core.function import FunctionClass
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Dummy
from sympy.sets.contains import Contains
from sympy.utilities.misc import func_name, filldedent
from mpmath import mpi, mpf
from sympy.logic.boolalg import And, Or, Not, true, false
from sympy.utilities import subsets
from sympy.core import Lambda
from sympy.sets.fancysets import ImageSet
from sympy.geometry.util import _uniquely_named_symbol
from sympy.sets.fancysets import ImageSet
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.solvers.solveset import solveset
from sympy.core.function import diff, Lambda
from sympy.series import limit
from sympy.calculus.singularities import singularities
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.elementary.miscellaneous import Max
import itertools
from sympy.core.logic import fuzzy_and, fuzzy_bool
from sympy.core.compatibility import zip_longest
from sympy.utilities.iterables import sift
from sympy.simplify.simplify import clear_coefficients
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.elementary.miscellaneous import Max
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import Min, Max

converter[set] = lambda x: FiniteSet(*x)
converter[frozenset] = lambda x: FiniteSet(*x)

class Interval(Set, EvalfMixin):
    is_Interval = True
    _inf = left = start
    _sup = right = end
    @property
    def start(self):
        """
        The left end point of 'self'.

        This property takes the same value as the 'inf' property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).start
        0

        """
        return self._args[0]
    @property
    def end(self):
        """
        The right end point of 'self'.

        This property takes the same value as the 'sup' property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).end
        1

        """
        return self._args[1]
    @property
    def left_open(self):
        """
        True if 'self' is left-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, left_open=True).left_open
        True
        >>> Interval(0, 1, left_open=False).left_open
        False

        """
        return self._args[2]
    @property
    def right_open(self):
        """
        True if 'self' is right-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, right_open=True).right_open
        True
        >>> Interval(0, 1, right_open=False).right_open
        False

        """
        return self._args[3]
    def _union(self, other):
        """
        This function should only be used internally

        See Set._union for docstring
        """
        if other.is_UniversalSet:
            return S.UniversalSet
        if other.is_Interval and self._is_comparable(other):
            from sympy.functions.elementary.miscellaneous import Min, Max
            # Non-overlapping intervals
            end = Min(self.end, other.end)
            start = Max(self.start, other.start)
            if (end < start or
               (end == start and (end not in self and end not in other))):
                return None
            else:
                start = Min(self.start, other.start)
                end = Max(self.end, other.end)

                left_open = ((self.start != start or self.left_open) and
                             (other.start != start or other.left_open))
                right_open = ((self.end != end or self.right_open) and
                              (other.end != end or other.right_open))

                return Interval(start, end, left_open, right_open)

        # If I have open end points and these endpoints are contained in other.
        # But only in case, when endpoints are finite. Because
        # interval does not contain oo or -oo.
        open_left_in_other_and_finite = (self.left_open and
                                         sympify(other.contains(self.start)) is S.true and
                                         self.start.is_finite)
        open_right_in_other_and_finite = (self.right_open and
                                          sympify(other.contains(self.end)) is S.true and
                                          self.end.is_finite)
        if open_left_in_other_and_finite or open_right_in_other_and_finite:
            # Fill in my end points and return
            open_left = self.left_open and self.start not in other
            open_right = self.right_open and self.end not in other
            new_self = Interval(self.start, self.end, open_left, open_right)
            return set((new_self, other))

        return None
    def _contains(self, other):
        if not isinstance(other, Expr) or (
                other is S.Infinity or
                other is S.NegativeInfinity or
                other is S.NaN or
                other is S.ComplexInfinity) or other.is_real is False:
            return false

        if self.start is S.NegativeInfinity and self.end is S.Infinity:
            if not other.is_real is None:
                return other.is_real

        if self.left_open:
            expr = other > self.start
        else:
            expr = other >= self.start

        if self.right_open:
            expr = And(expr, other < self.end)
        else:
            expr = And(expr, other <= self.end)

        return _sympify(expr)