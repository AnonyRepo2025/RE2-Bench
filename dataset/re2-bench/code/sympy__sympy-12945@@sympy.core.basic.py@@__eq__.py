from __future__ import print_function, division
from collections import Mapping, defaultdict
from .assumptions import BasicMeta, ManagedProperties
from .cache import cacheit
from .sympify import _sympify, sympify, SympifyError
from .compatibility import (iterable, Iterator, ordered,
    string_types, with_metaclass, zip_longest, range)
from .singleton import S
from inspect import getmro
from .function import AppliedUndef, UndefinedFunction as UndefFunc
from sympy import Derivative, Function, Symbol
from sympy.series.order import Order
from sympy import Pow
from sympy.printing import sstr
from sympy.printing import sstr
from sympy import Symbol
from sympy import Symbol
from sympy.simplify import hypersimp
from sympy.polys import Poly, PolynomialError
from sympy.core.containers import Dict
from sympy.utilities import default_sort_key
from sympy import Dummy, Symbol
from sympy.core.function import UndefinedFunction, Function
from sympy.core.symbol import Dummy
from sympy.simplify.simplify import bottom_up
from sympy import count_ops
from sympy.core.symbol import Wild
from sympy.utilities.misc import filldedent



class Basic(with_metaclass(ManagedProperties)
):
    __slots__ = ['_mhash',              # hash value
                 '_args',               # arguments
                 '_assumptions'
                ]
    is_number = False
    is_Atom = False
    is_Symbol = False
    is_symbol = False
    is_Indexed = False
    is_Dummy = False
    is_Wild = False
    is_Function = False
    is_Add = False
    is_Mul = False
    is_Pow = False
    is_Number = False
    is_Float = False
    is_Rational = False
    is_Integer = False
    is_NumberSymbol = False
    is_Order = False
    is_Derivative = False
    is_Piecewise = False
    is_Poly = False
    is_AlgebraicNumber = False
    is_Relational = False
    is_Equality = False
    is_Boolean = False
    is_Not = False
    is_Matrix = False
    is_Vector = False
    is_Point = False
    _constructor_postprocessor_mapping = {}
    def __hash__(self):
        # hash cannot be cached using cache_it because infinite recurrence
        # occurs as hash is needed for setting cache dictionary keys
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h
    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.

        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args
    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        their symbolic trees.

        This is the same as a.compare(b) == 0 but faster.

        Notes
        =====

        If a class that overrides __eq__() needs to retain the
        implementation of __hash__() from a parent class, the
        interpreter must be told this explicitly by setting __hash__ =
        <ParentClass>.__hash__. Otherwise the inheritance of __hash__()
        will be blocked, just as if __hash__ had been explicitly set to
        None.

        References
        ==========

        from http://docs.python.org/dev/reference/datamodel.html#object.__hash__
        """
        from sympy import Pow
        if self is other:
            return True

        if type(self) is not type(other):
            # issue 6100 a**1.0 == a like a**2.0 == a**2
            if isinstance(self, Pow) and self.exp == 1:
                return self.base == other
            if isinstance(other, Pow) and other.exp == 1:
                return self == other.base
            try:
                other = _sympify(other)
            except SympifyError:
                return False    # sympy != other

            if type(self) != type(other):
                return False

        return self._hashable_content() == other._hashable_content()