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
    def col_join(self, other):
        """Returns B augmented beneath A (row-wise joining)::

            [A]
            [B]

        Examples
        ========

        >>> from sympy import SparseMatrix, Matrix, ones
        >>> A = SparseMatrix(ones(3))
        >>> A
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
        >>> B = SparseMatrix.eye(3)
        >>> B
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        >>> C = A.col_join(B); C
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        >>> C == A.col_join(Matrix(B))
        True

        Joining along columns is the same as appending rows at the end
        of the matrix:

        >>> C == A.row_insert(A.rows, Matrix(B))
        True
        """
        # A null matrix can always be stacked (see  #10770)
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        A, B = self, other
        if not A.cols == B.cols:
            raise ShapeError()
        A = A.copy()
        if not isinstance(B, SparseMatrix):
            k = 0
            b = B._mat
            for i in range(B.rows):
                for j in range(B.cols):
                    v = b[k]
                    if v:
                        A._smat[(i + A.rows, j)] = v
                    k += 1
        else:
            for (i, j), v in B._smat.items():
                A._smat[i + A.rows, j] = v
        A.rows += B.rows
        return A