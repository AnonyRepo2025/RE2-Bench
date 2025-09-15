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



class MatrixShaping(MatrixRequired):
    def _eval_col_insert(self, pos, other):
        cols = self.cols

        def entry(i, j):
            if j < pos:
                return self[i, j]
            elif pos <= j < pos + other.cols:
                return other[i, j - pos]
            return self[i, j - other.cols]

        return self._new(self.rows, self.cols + other.cols,
                         lambda i, j: entry(i, j))