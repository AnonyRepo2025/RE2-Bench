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
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import Symbol, Dummy, _uniquely_named_symbol
from sympy.sets.contains import Contains
from sympy.utilities.iterables import sift
from sympy.utilities.misc import func_name, filldedent
from mpmath import mpi, mpf
from sympy.logic.boolalg import And, Or, Not, true, false
from sympy.utilities import subsets
from sympy.core import Lambda
from sympy.sets.fancysets import ImageSet
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
from sympy.simplify.simplify import clear_coefficients
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.elementary.miscellaneous import Max
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import Min, Max
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
    @staticmethod
    def _infimum_key(expr):
        """
        Return infimum (if possible) else S.Infinity.
        """
        try:
            infimum = expr.inf
            assert infimum.is_comparable
        except (NotImplementedError,
                AttributeError, AssertionError, ValueError):
            infimum = S.Infinity
        return infimum
    def _complement(self, other):
        # this behaves as other - self
        if isinstance(other, ProductSet):
            # For each set consider it or it's complement
            # We need at least one of the sets to be complemented
            # Consider all 2^n combinations.
            # We can conveniently represent these options easily using a
            # ProductSet

            # XXX: this doesn't work if the dimensions of the sets isn't same.
            # A - B is essentially same as A if B has a different
            # dimensionality than A
            switch_sets = ProductSet(FiniteSet(o, o - s) for s, o in
                                     zip(self.sets, other.sets))
            product_sets = (ProductSet(*set) for set in switch_sets)
            # Union of all combinations but this one
            return Union(p for p in product_sets if p != other)

        elif isinstance(other, Interval):
            if isinstance(self, Interval) or isinstance(self, FiniteSet):
                return Intersection(other, self.complement(S.Reals))

        elif isinstance(other, Union):
            return Union(o - self for o in other.args)

        elif isinstance(other, Complement):
            return Complement(other.args[0], Union(other.args[1], self), evaluate=False)

        elif isinstance(other, EmptySet):
            return S.EmptySet

        elif isinstance(other, FiniteSet):
            from sympy.utilities.iterables import sift

            def ternary_sift(el):
                contains = self.contains(el)
                return contains if contains in [True, False] else None

            sifted = sift(other, ternary_sift)
            # ignore those that are contained in self
            return Union(FiniteSet(*(sifted[False])),
                Complement(FiniteSet(*(sifted[None])), self, evaluate=False)
                if sifted[None] else S.EmptySet)
    def contains(self, other):
        """
        Returns True if 'other' is contained in 'self' as an element.

        As a shortcut it is possible to use the 'in' operator:

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).contains(0.5)
        True
        >>> 0.5 in Interval(0, 1)
        True

        """
        other = sympify(other, strict=True)
        ret = sympify(self._contains(other))
        if ret is None:
            ret = Contains(other, self, evaluate=False)
        return ret
    def __contains__(self, other):
        symb = sympify(self.contains(other))
        if not (symb is S.true or symb is S.false):
            raise TypeError('contains did not evaluate to a bool: %r' % symb)
        return bool(symb)