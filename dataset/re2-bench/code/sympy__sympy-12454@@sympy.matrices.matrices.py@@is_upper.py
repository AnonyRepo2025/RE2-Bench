from __future__ import print_function, division
import collections
from sympy.core.add import Add
from sympy.core.basic import Basic, Atom
from sympy.core.expr import Expr
from sympy.core.function import count_ops
from sympy.core.logic import fuzzy_and
from sympy.core.power import Pow
from sympy.core.symbol import Symbol, Dummy, symbols
from sympy.core.numbers import Integer, ilcm, Float
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.compatibility import is_sequence, default_sort_key, range, \
    NotIterable
from sympy.polys import PurePoly, roots, cancel, gcd
from sympy.simplify import simplify as _simplify, signsimp, nsimplify
from sympy.utilities.iterables import flatten, numbered_symbols
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions import exp, factorial
from sympy.printing import sstr
from sympy.core.compatibility import reduce, as_int, string_types
from sympy.assumptions.refine import refine
from sympy.core.decorators import call_highest_priority
from sympy.core.decorators import deprecated
from sympy.utilities.exceptions import SymPyDeprecationWarning
from types import FunctionType
from sympy.matrices import MutableMatrix
from sympy.matrices import MutableMatrix
from sympy.functions.elementary.complexes import re, im
from .dense import matrix2numpy
from sympy.matrices import diag, MutableMatrix
from sympy import binomial
from sympy.matrices import eye
from sympy.matrices.sparse import SparseMatrix
from .dense import Matrix
from sympy.matrices import zeros
from sympy.physics.matrices import mgamma
from sympy.matrices import diag
from .dense import Matrix
from sympy import LeviCivita
from sympy.matrices import zeros
from sympy.matrices import eye
from sympy.matrices import diag
from sympy.matrices import Matrix, zeros
from sympy.ntheory import totient
from .dense import Matrix
from sympy.matrices import MutableMatrix
from sympy.matrices import MutableMatrix
from sympy.matrices import diag
from sympy.matrices import SparseMatrix
from sympy.matrices import zeros
from sympy.matrices import eye
from sympy.matrices import zeros
import numpy
from sympy.printing.str import StrPrinter
from sympy import eye
from sympy.matrices import MutableMatrix



class MatrixProperties(MatrixRequired):
    @property
    def is_upper(self):
        """Check if matrix is an upper triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_upper
        True

        >>> m = Matrix(4, 3, [5, 1, 9, 0, 4 , 6, 0, 0, 5, 0, 0, 0])
        >>> m
        Matrix([
        [5, 1, 9],
        [0, 4, 6],
        [0, 0, 5],
        [0, 0, 0]])
        >>> m.is_upper
        True

        >>> m = Matrix(2, 3, [4, 2, 5, 6, 1, 1])
        >>> m
        Matrix([
        [4, 2, 5],
        [6, 1, 1]])
        >>> m.is_upper
        False

        See Also
        ========

        is_lower
        is_diagonal
        is_upper_hessenberg
        """
        return all(self[i, j].is_zero
                   for i in range(1, self.rows)
                   for j in range(min(i, self.cols)))