from __future__ import print_function, division
from collections import defaultdict
import inspect
from sympy.core.basic import Basic
from sympy.core.compatibility import (iterable, with_metaclass,
    ordered, range, PY3, reduce)
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import deprecated
from sympy.core.evalf import EvalfMixin
from sympy.core.evaluate import global_evaluate
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_bool, fuzzy_or, fuzzy_and, fuzzy_not
from sympy.core.numbers import Float
from sympy.core.operations import LatticeOp
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Symbol, Dummy, _uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.sets.contains import Contains
from sympy.utilities import subsets
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.iterables import iproduct, sift, roundrobin
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
from sympy.sets.handlers.issubset import is_subset_sets
from .powerset import PowerSet
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.elementary.miscellaneous import Max
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.elementary.miscellaneous import Max
from sympy.core.relational import Eq
from .powerset import PowerSet
import sys
from sympy.utilities.iterables import sift

tfn = defaultdict(lambda: None, {
    True: S.true,
    S.true: S.true,
    False: S.false,
    S.false: S.false})
converter[set] = lambda x: FiniteSet(*x)
converter[frozenset] = lambda x: FiniteSet(*x)

class FiniteSet(Set, EvalfMixin):
    is_FiniteSet = True
    is_iterable = True
    is_empty = False
    is_finite_set = True
    def _contains(self, other):
        """
        Tests whether an element, other, is in the set.

        The actual test is for mathematical equality (as opposed to
        syntactical equality). In the worst case all elements of the
        set must be checked.

        Examples
        ========

        >>> from sympy import FiniteSet
        >>> 1 in FiniteSet(1, 2)
        True
        >>> 5 in FiniteSet(1, 2)
        False

        """
        if other in self._args_set:
            return True
        else:
            # evaluate=True is needed to override evaluate=False context;
            # we need Eq to do the evaluation
            return fuzzy_or(fuzzy_bool(Eq(e, other, evaluate=True))
                for e in self.args)