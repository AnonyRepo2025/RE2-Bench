from __future__ import print_function, division
import collections
from sympy.core.add import Add
from sympy.core.basic import Basic, Atom
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.function import count_ops
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.compatibility import is_sequence, default_sort_key, range, \
    NotIterable
from sympy.simplify import simplify as _simplify, signsimp, nsimplify
from sympy.utilities.iterables import flatten
from sympy.functions import Abs
from sympy.core.compatibility import reduce, as_int, string_types
from sympy.assumptions.refine import refine
from sympy.core.decorators import call_highest_priority
from types import FunctionType
from sympy.matrices import MutableMatrix
from sympy.functions.elementary.complexes import re, im
from sympy.combinatorics import Permutation
from sympy.simplify import trigsimp
import numpy



class MatrixArithmetic(MatrixRequired):
    _op_priority = 10.01
    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        other = _matrixify(other)
        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
            return NotImplemented

        return self.__mul__(other)