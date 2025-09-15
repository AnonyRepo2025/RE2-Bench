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
    """Provides basic matrix shaping and extracting of submatrices"""

    def _eval_col_del(self, col):
        def entry(i, j):
            return self[i, j] if j < col else self[i, j + 1]
        return self._new(self.rows, self.cols - 1, entry)

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

    def _eval_col_join(self, other):
        rows = self.rows

        def entry(i, j):
            if i < rows:
                return self[i, j]
            return other[i - rows, j]

        return classof(self, other)._new(self.rows + other.rows, self.cols,
                                         lambda i, j: entry(i, j))

    def _eval_extract(self, rowsList, colsList):
        mat = list(self)
        cols = self.cols
        indices = (i * cols + j for i in rowsList for j in colsList)
        return self._new(len(rowsList), len(colsList),
                         list(mat[i] for i in indices))

    def _eval_get_diag_blocks(self):
        sub_blocks = []

        def recurse_sub_blocks(M):
            i = 1
            while i <= M.shape[0]:
                if i == 1:
                    to_the_right = M[0, i:]
                    to_the_bottom = M[i:, 0]
                else:
                    to_the_right = M[:i, i:]
                    to_the_bottom = M[i:, :i]
                if any(to_the_right) or any(to_the_bottom):
                    i += 1
                    continue
                else:
                    sub_blocks.append(M[:i, :i])
                    if M.shape == M[:i, :i].shape:
                        return
                    else:
                        recurse_sub_blocks(M[i:, i:])
                        return

        recurse_sub_blocks(self)
        return sub_blocks

    def _eval_row_del(self, row):
        def entry(i, j):
            return self[i, j] if i < row else self[i + 1, j]
        return self._new(self.rows - 1, self.cols, entry)

    def _eval_row_insert(self, pos, other):
        entries = list(self)
        insert_pos = pos * self.cols
        entries[insert_pos:insert_pos] = list(other)
        return self._new(self.rows + other.rows, self.cols, entries)

    def _eval_row_join(self, other):
        cols = self.cols

        def entry(i, j):
            if j < cols:
                return self[i, j]
            return other[i, j - cols]

        return classof(self, other)._new(self.rows, self.cols + other.cols,
                                         lambda i, j: entry(i, j))

    def _eval_tolist(self):
        return [list(self[i,:]) for i in range(self.rows)]

    def _eval_vec(self):
        rows = self.rows

        def entry(n, _):
            # we want to read off the columns first
            j = n // rows
            i = n - j * rows
            return self[i, j]

        return self._new(len(self), 1, entry)

    def col_del(self, col):
        """Delete the specified column."""
        if col < 0:
            col += self.cols
        if not 0 <= col < self.cols:
            raise ValueError("Column {} out of range.".format(col))
        return self._eval_col_del(col)

    def col_insert(self, pos, other):
        """Insert one or more columns at the given column position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.col_insert(1, V)
        Matrix([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])

        See Also
        ========

        col
        row_insert
        """
        # Allows you to build a matrix even if it is null matrix
        if not self:
            return type(self)(other)

        if pos < 0:
            pos = self.cols + pos
        if pos < 0:
            pos = 0
        elif pos > self.cols:
            pos = self.cols

        if self.rows != other.rows:
            raise ShapeError(
                "self and other must have the same number of rows.")

        return self._eval_col_insert(pos, other)

    def col_join(self, other):
        """Concatenates two matrices along self's last and other's first row.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.col_join(V)
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]])

        See Also
        ========

        col
        row_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        if self.cols != other.cols:
            raise ShapeError(
                "`self` and `other` must have the same number of columns.")
        return self._eval_col_join(other)

    def col(self, j):
        """Elementary column selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).col(0)
        Matrix([
        [1],
        [0]])

        See Also
        ========

        row
        col_op
        col_swap
        col_del
        col_join
        col_insert
        """
        return self[:, j]

    def extract(self, rowsList, colsList):
        """Return a submatrix by specifying a list of rows and columns.
        Negative indices can be given. All indices must be in the range
        -n <= i < n where n is the number of rows or columns.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(4, 3, range(12))
        >>> m
        Matrix([
        [0,  1,  2],
        [3,  4,  5],
        [6,  7,  8],
        [9, 10, 11]])
        >>> m.extract([0, 1, 3], [0, 1])
        Matrix([
        [0,  1],
        [3,  4],
        [9, 10]])

        Rows or columns can be repeated:

        >>> m.extract([0, 0, 1], [-1])
        Matrix([
        [2],
        [2],
        [5]])

        Every other row can be taken by using range to provide the indices:

        >>> m.extract(range(0, m.rows, 2), [-1])
        Matrix([
        [2],
        [8]])

        RowsList or colsList can also be a list of booleans, in which case
        the rows or columns corresponding to the True values will be selected:

        >>> m.extract([0, 1, 2, 3], [True, False, True])
        Matrix([
        [0,  2],
        [3,  5],
        [6,  8],
        [9, 11]])
        """

        if not is_sequence(rowsList) or not is_sequence(colsList):
            raise TypeError("rowsList and colsList must be iterable")
        # ensure rowsList and colsList are lists of integers
        if rowsList and all(isinstance(i, bool) for i in rowsList):
            rowsList = [index for index, item in enumerate(rowsList) if item]
        if colsList and all(isinstance(i, bool) for i in colsList):
            colsList = [index for index, item in enumerate(colsList) if item]

        # ensure everything is in range
        rowsList = [a2idx(k, self.rows) for k in rowsList]
        colsList = [a2idx(k, self.cols) for k in colsList]

        return self._eval_extract(rowsList, colsList)

    def get_diag_blocks(self):
        """Obtains the square sub-matrices on the main diagonal of a square matrix.

        Useful for inverting symbolic matrices or solving systems of
        linear equations which may be decoupled by having a block diagonal
        structure.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y, z
        >>> A = Matrix([[1, 3, 0, 0], [y, z*z, 0, 0], [0, 0, x, 0], [0, 0, 0, 0]])
        >>> a1, a2, a3 = A.get_diag_blocks()
        >>> a1
        Matrix([
        [1,    3],
        [y, z**2]])
        >>> a2
        Matrix([[x]])
        >>> a3
        Matrix([[0]])

        """
        return self._eval_get_diag_blocks()

    @classmethod
    def hstack(cls, *args):
        """Return a matrix formed by joining args horizontally (i.e.
        by repeated application of row_join).

        Examples
        ========

        >>> from sympy.matrices import Matrix, eye
        >>> Matrix.hstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.row_join, args)

    def reshape(self, rows, cols):
        """Reshape the matrix. Total number of elements must remain the same.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 3, lambda i, j: 1)
        >>> m
        Matrix([
        [1, 1, 1],
        [1, 1, 1]])
        >>> m.reshape(1, 6)
        Matrix([[1, 1, 1, 1, 1, 1]])
        >>> m.reshape(3, 2)
        Matrix([
        [1, 1],
        [1, 1],
        [1, 1]])

        """
        if self.rows * self.cols != rows * cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        return self._new(rows, cols, lambda i, j: self[i * cols + j])

    def row_del(self, row):
        """Delete the specified row."""
        if row < 0:
            row += self.rows
        if not 0 <= row < self.rows:
            raise ValueError("Row {} out of range.".format(row))

        return self._eval_row_del(row)

    def row_insert(self, pos, other):
        """Insert one or more rows at the given row position.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.row_insert(1, V)
        Matrix([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]])

        See Also
        ========

        row
        col_insert
        """
        from sympy.matrices import MutableMatrix
        # Allows you to build a matrix even if it is null matrix
        if not self:
            return self._new(other)

        if pos < 0:
            pos = self.rows + pos
        if pos < 0:
            pos = 0
        elif pos > self.rows:
            pos = self.rows

        if self.cols != other.cols:
            raise ShapeError(
                "`self` and `other` must have the same number of columns.")

        return self._eval_row_insert(pos, other)

    def row_join(self, other):
        """Concatenates two matrices along self's last and rhs's first column

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.row_join(V)
        Matrix([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]])

        See Also
        ========

        row
        col_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.cols == 0 and self.rows != other.rows:
            return self._new(other.rows, 0, []).row_join(other)

        if self.rows != other.rows:
            raise ShapeError(
                "`self` and `rhs` must have the same number of rows.")
        return self._eval_row_join(other)

    def row(self, i):
        """Elementary row selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).row(0)
        Matrix([[1, 0]])

        See Also
        ========

        col
        row_op
        row_swap
        row_del
        row_join
        row_insert
        """
        return self[i, :]

    @property
    def shape(self):
        """The shape (dimensions) of the matrix as the 2-tuple (rows, cols).

        Examples
        ========

        >>> from sympy.matrices import zeros
        >>> M = zeros(2, 3)
        >>> M.shape
        (2, 3)
        >>> M.rows
        2
        >>> M.cols
        3
        """
        return (self.rows, self.cols)

    def tolist(self):
        """Return the Matrix as a nested Python list.

        Examples
        ========

        >>> from sympy import Matrix, ones
        >>> m = Matrix(3, 3, range(9))
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
        >>> m.tolist()
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> ones(3, 0).tolist()
        [[], [], []]

        When there are no rows then it will not be possible to tell how
        many columns were in the original matrix:

        >>> ones(0, 3).tolist()
        []

        """
        if not self.rows:
            return []
        if not self.cols:
            return [[] for i in range(self.rows)]
        return self._eval_tolist()

    def vec(self):
        """Return the Matrix converted into a one column matrix by stacking columns

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 3], [2, 4]])
        >>> m
        Matrix([
        [1, 3],
        [2, 4]])
        >>> m.vec()
        Matrix([
        [1],
        [2],
        [3],
        [4]])

        See Also
        ========

        vech
        """
        return self._eval_vec()

    @classmethod
    def vstack(cls, *args):
        """Return a matrix formed by joining args vertically (i.e.
        by repeated application of col_join).

        Examples
        ========

        >>> from sympy.matrices import Matrix, eye
        >>> Matrix.vstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.col_join, args)
