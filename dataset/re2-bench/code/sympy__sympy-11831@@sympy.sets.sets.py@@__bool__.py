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

class ProductSet(Set):
    is_ProductSet = True
    __nonzero__ = __bool__
    def __bool__(self):
        return all([bool(s) for s in self.args])