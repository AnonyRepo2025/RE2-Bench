from __future__ import print_function, division
from functools import wraps, reduce
import collections
from sympy.core import S, Symbol, Tuple, Integer, Basic, Expr, Eq
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import range, SYMPY_INTS, default_sort_key
from sympy.core.sympify import SympifyError, sympify
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices import ShapeError
from sympy.simplify import simplify
from sympy.utilities.misc import filldedent
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from sympy import Derivative
from sympy.matrices.expressions.adjoint import Adjoint
from sympy.matrices.expressions.transpose import Transpose
from sympy import I
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.adjoint import Adjoint
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.immutable import ImmutableDenseMatrix
from numpy import empty
from sympy import Sum, Mul, Add, MatMul, transpose, trace
from sympy.strategies.traverse import bottom_up
from .applyfunc import ElementwiseApplyFunction
from sympy import MatrixBase
from sympy import Sum, symbols, Dummy
from sympy.matrices.expressions.slice import MatrixSlice
from sympy import MatrixBase
from sympy.matrices.expressions.slice import MatrixSlice



class MatrixExpr(Expr):
    _iterable = False
    _op_priority = 11.0
    is_Matrix = True
    is_MatrixExpr = True
    is_Identity = None
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False
    is_commutative = False
    is_number = False
    is_symbol = False
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    T = property(transpose, None, None, 'Matrix transposition.')
    inv = inverse
    def __new__(cls, *args, **kwargs):
        args = map(sympify, args)
        return Basic.__new__(cls, *args, **kwargs)
    def __neg__(self):
        return MatMul(S.NegativeOne, self).doit()
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return MatAdd(self, -other, check=True).doit()
    @property
    def rows(self):
        return self.shape[0]
    @property
    def cols(self):
        return self.shape[1]
    def _eval_Eq(self, other):
        if not isinstance(other, MatrixExpr):
            return False
        if self.shape != other.shape:
            return False
        if (self - other).is_ZeroMatrix:
            return True
        return Eq(self, other, evaluate=False)