from __future__ import print_function, division
from itertools import product
import inspect
from sympy.core.basic import Basic
from sympy.core.compatibility import (iterable, with_metaclass,
    ordered, range, PY3)
from sympy.core.cache import cacheit
from sympy.core.evalf import EvalfMixin
from sympy.core.evaluate import global_evaluate
from sympy.core.expr import Expr
from sympy.core.function import FunctionClass
from sympy.core.logic import fuzzy_bool
from sympy.core.mul import Mul
from sympy.core.numbers import Float
from sympy.core.operations import LatticeOp
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Symbol, Dummy, _uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And, Or, Not, true, false
from sympy.sets.contains import Contains
from sympy.utilities import subsets
from sympy.utilities.iterables import sift
from sympy.utilities.misc import func_name, filldedent
from mpmath import mpi, mpf
from sympy.core import Lambda
from sympy.sets.fancysets import ImageSet
from sympy.sets.setexpr import set_function
from sympy import exp, log
from sympy.sets.handlers.union import union_sets
from sympy.sets.handlers.intersection import intersection_sets
from sympy.sets import ImageSet
from sympy import symbols,Lambda
from sympy.sets.handlers.add import _set_add
from sympy.sets.handlers.add import _set_sub
from sympy.sets.handlers.mul import _set_mul
from sympy.sets.handlers.mul import _set_div
from sympy.sets.handlers.power import _set_pow
from sympy.sets.handlers.functions import _set_function
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.elementary.miscellaneous import Max
import itertools
from sympy.core.logic import fuzzy_and, fuzzy_bool
from sympy.core.compatibility import zip_longest
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.elementary.miscellaneous import Max
from sympy.core.relational import Eq
import sys
from sympy.utilities.iterables import sift

converter[set] = lambda x: FiniteSet(*x)
converter[frozenset] = lambda x: FiniteSet(*x)

class Set(Basic):
    is_number = False
    is_iterable = False
    is_interval = False
    is_FiniteSet = False
    is_Interval = False
    is_ProductSet = False
    is_Union = False
    is_Intersection = None
    is_EmptySet = None
    is_UniversalSet = None
    is_Complement = None
    is_ComplexRegion = False
    def is_subset(self, other):
        """
        Returns True if 'self' is a subset of 'other'.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_subset(Interval(0, 1, left_open=True))
        False

        """
        if isinstance(other, Set):
            # XXX issue 16873
            # self might be an unevaluated form of self
            # so the equality test will fail
            return self.intersect(other) == self
        else:
            raise ValueError("Unknown argument '%s'" % other)