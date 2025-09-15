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
    def __new__(cls, start, end, left_open=False, right_open=False):

        start = _sympify(start)
        end = _sympify(end)
        left_open = _sympify(left_open)
        right_open = _sympify(right_open)

        if not all(isinstance(a, (type(true), type(false)))
            for a in [left_open, right_open]):
            raise NotImplementedError(
                "left_open and right_open can have only true/false values, "
                "got %s and %s" % (left_open, right_open))

        inftys = [S.Infinity, S.NegativeInfinity]
        # Only allow real intervals (use symbols with 'is_real=True').
        if not all(i.is_real is not False or i in inftys for i in (start, end)):
            raise ValueError("Non-real intervals are not supported")

        # evaluate if possible
        if (end < start) == True:
            return S.EmptySet
        elif (end - start).is_negative:
            return S.EmptySet

        if end == start and (left_open or right_open):
            return S.EmptySet
        if end == start and not (left_open or right_open):
            if start == S.Infinity or start == S.NegativeInfinity:
                return S.EmptySet
            return FiniteSet(end)

        # Make sure infinite interval end points are open.
        if start == S.NegativeInfinity:
            left_open = true
        if end == S.Infinity:
            right_open = true

        return Basic.__new__(cls, start, end, left_open, right_open)