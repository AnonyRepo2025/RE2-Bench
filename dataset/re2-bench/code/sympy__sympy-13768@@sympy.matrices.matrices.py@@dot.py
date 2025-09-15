from __future__ import print_function, division
import collections
from mpmath.libmp.libmpf import prec_to_dps
from sympy.assumptions.refine import refine
from sympy.core.add import Add
from sympy.core.basic import Basic, Atom
from sympy.core.expr import Expr
from sympy.core.function import expand_mul
from sympy.core.power import Pow
from sympy.core.symbol import (Symbol, Dummy, symbols,
    _uniquely_named_symbol)
from sympy.core.numbers import Integer, ilcm, Float
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions import Abs, exp, factorial
from sympy.polys import PurePoly, roots, cancel, gcd
from sympy.printing import sstr
from sympy.simplify import simplify as _simplify, signsimp, nsimplify
from sympy.core.compatibility import reduce, as_int, string_types
from sympy.utilities.iterables import flatten, numbered_symbols
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import (is_sequence, default_sort_key, range,
    NotIterable)
from sympy.utilities.exceptions import SymPyDeprecationWarning
from types import FunctionType
from .common import (a2idx, classof, MatrixError, ShapeError,
        NonSquareMatrixError, MatrixCommon)
from sympy.matrices import eye
from sympy import Derivative
from sympy import derive_by_array
from .dense import Matrix
from sympy.matrices import zeros
from .dense import matrix2numpy
from sympy.matrices import diag, MutableMatrix
from sympy import binomial
from sympy.matrices.sparse import SparseMatrix
from .dense import Matrix
from sympy.physics.matrices import mgamma
from .dense import Matrix
from sympy import LeviCivita
from sympy.matrices import zeros
from sympy.matrices import diag
from sympy import re
from sympy.matrices import Matrix, zeros
from sympy.ntheory import totient
from .dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.matrices import eye
from sympy.matrices import zeros
import numpy
from sympy.printing.str import StrPrinter
from sympy import gcd
from sympy import eye



class MatrixBase(MatrixDeprecated, MatrixCalculus, MatrixEigen, MatrixCommon):
    __array_priority__ = 11
    is_Matrix = True
    _class_priority = 3
    _sympify = staticmethod(sympify)
    __hash__ = None  # Mutable
    def dot(self, b):
        """Return the dot product of two vectors of equal length. ``self`` must
        be a ``Matrix`` of size 1 x n or n x 1, and ``b`` must be either a
        matrix of size 1 x n, n x 1, or a list/tuple of length n. A scalar is returned.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> v = Matrix([1, 1, 1])
        >>> M.row(0).dot(v)
        6
        >>> M.col(0).dot(v)
        12
        >>> v = [3, 2, 1]
        >>> M.row(0).dot(v)
        10

        See Also
        ========

        cross
        multiply
        multiply_elementwise
        """
        from .dense import Matrix

        if not isinstance(b, MatrixBase):
            if is_sequence(b):
                if len(b) != self.cols and len(b) != self.rows:
                    raise ShapeError(
                        "Dimensions incorrect for dot product: %s, %s" % (
                            self.shape, len(b)))
                return self.dot(Matrix(b))
            else:
                raise TypeError(
                    "`b` must be an ordered iterable or Matrix, not %s." %
                    type(b))

        mat = self
        if (1 not in mat.shape) or (1 not in b.shape) :
            SymPyDeprecationWarning(
                feature="Dot product of non row/column vectors",
                issue=13815,
                deprecated_since_version="1.2").warn()
            return mat._legacy_array_dot(b)
        if len(mat) != len(b):
            raise ShapeError("Dimensions incorrect for dot product: %s, %s" % (self.shape, b.shape))
        n = len(mat)
        if mat.shape != (1, n):
            mat = mat.reshape(1, n)
        if b.shape != (n, 1):
            b = b.reshape(n, 1)
        # Now ``mat`` is a row vector and ``b`` is a column vector.
        return (mat * b)[0]