from __future__ import print_function, division
from itertools import product
from collections import defaultdict
import inspect
from sympy.core.basic import Basic
from sympy.core.compatibility import (iterable, with_metaclass,
    ordered, range, PY3, is_sequence)
from sympy.core.cache import cacheit
from sympy.core.evalf import EvalfMixin
from sympy.core.evaluate import global_evaluate
from sympy.core.expr import Expr
from sympy.core.function import FunctionClass
from sympy.core.logic import fuzzy_bool, fuzzy_or
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

tfn = defaultdict(lambda: None, {
    True: S.true,
    S.true: S.true,
    False: S.false,
    S.false: S.false})
converter[set] = lambda x: FiniteSet(*x)
converter[frozenset] = lambda x: FiniteSet(*x)

class Intersection(Set, LatticeOp):
    is_Intersection = True
    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_evaluate[0])

        # flatten inputs to merge intersections and iterables
        args = list(ordered(set(_sympify(args))))

        # Reduce sets using known rules
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_intersection(args)

        args = list(ordered(args, Set._infimum_key))

        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj