from __future__ import print_function, division
import copy
from collections import defaultdict
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.compatibility import is_sequence, as_int, range
from sympy.core.logic import fuzzy_and
from sympy.core.singleton import S
from sympy.functions import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.iterables import uniq
from .matrices import MatrixBase, ShapeError, a2idx
from .dense import Matrix
import collections
from .immutable import ImmutableSparseMatrix
from sympy.core.numbers import nan, oo
from sympy.core.numbers import nan, oo
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.matexpr import MatrixElement



class MutableSparseMatrix(SparseMatrix, MatrixBase):
    __hash__ = None
    @classmethod
    def _new(cls, *args, **kwargs):
        return cls(*args)
    def row_join(self, other):
        """Returns B appended after A (column-wise augmenting)::

            [A B]

        Examples
        ========

        >>> from sympy import SparseMatrix, Matrix
        >>> A = SparseMatrix(((1, 0, 1), (0, 1, 0), (1, 1, 0)))
        >>> A
        Matrix([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]])
        >>> B = SparseMatrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        >>> B
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        >>> C = A.row_join(B); C
        Matrix([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 1]])
        >>> C == A.row_join(Matrix(B))
        True

        Joining at row ends is the same as appending columns at the end
        of the matrix:

        >>> C == A.col_insert(A.cols, B)
        True
        """
        # A null matrix can always be stacked (see  #10770)
        if self.cols == 0 and self.rows != other.rows:
            return self._new(other.rows, 0, []).row_join(other)

        A, B = self, other
        if not A.rows == B.rows:
            raise ShapeError()
        A = A.copy()
        if not isinstance(B, SparseMatrix):
            k = 0
            b = B._mat
            for i in range(B.rows):
                for j in range(B.cols):
                    v = b[k]
                    if v:
                        A._smat[(i, j + A.cols)] = v
                    k += 1
        else:
            for (i, j), v in B._smat.items():
                A._smat[(i, j + A.cols)] = v
        A.cols += B.cols
        return A