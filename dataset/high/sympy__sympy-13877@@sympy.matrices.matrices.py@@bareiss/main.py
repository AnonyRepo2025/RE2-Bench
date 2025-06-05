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
from types import FunctionType
from .common import (a2idx, classof, MatrixError, ShapeError,
        NonSquareMatrixError, MatrixCommon)
from sympy.matrices import eye
from sympy import Derivative
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



class MatrixDeterminant(MatrixCommon):

    def _eval_det_bareiss(self):
        def bareiss(mat, cumm=1):
            if mat.rows == 0:
                return S.One
            elif mat.rows == 1:
                return mat[0, 0]
            pivot_pos, pivot_val, _, _ = _find_reasonable_pivot(mat[:, 0],
                                         iszerofunc=_is_zero_after_expand_mul)
            if pivot_pos == None:
                return S.Zero
            sign = (-1) ** (pivot_pos % 2)
            rows = list(i for i in range(mat.rows) if i != pivot_pos)
            cols = list(range(mat.cols))
            tmp_mat = mat.extract(rows, cols)

            def entry(i, j):
                ret = (pivot_val*tmp_mat[i, j + 1] - mat[pivot_pos, j + 1]*tmp_mat[i, 0]) / cumm
                if not ret.is_Atom:
                    cancel(ret)
                return ret

            return sign*bareiss(self._new(mat.rows - 1, mat.cols - 1, entry), pivot_val)

        return cancel(bareiss(self))

