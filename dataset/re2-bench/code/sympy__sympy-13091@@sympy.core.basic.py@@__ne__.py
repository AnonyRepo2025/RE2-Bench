from __future__ import print_function, division
from collections import Mapping, defaultdict
from itertools import chain
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
    def __ne__(self, other):
        """a != b  -> Compare two symbolic trees and see whether they are different

           this is the same as:

             a.compare(b) != 0

           but faster
        """
        return not self == other