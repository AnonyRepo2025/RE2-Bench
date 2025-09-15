from typing import Tuple as tTuple
from sympy.core.logic import FuzzyBool
from functools import wraps, reduce
import collections
from sympy.core import S, Symbol, Integer, Basic, Expr, Mul, Add
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import SYMPY_INTS, default_sort_key
from sympy.core.symbol import Str
from sympy.core.sympify import SympifyError, _sympify
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.simplify import simplify
from sympy.matrices.matrices import MatrixKind
from sympy.utilities.misc import filldedent
from sympy.multipledispatch import dispatch
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.tensor.array.expressions.conv_array_to_matrix import convert_array_to_matrix
from sympy import ImmutableDenseMatrix
from sympy.matrices.expressions.adjoint import Adjoint
from sympy.matrices.expressions.transpose import Transpose
from sympy import I
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.adjoint import Adjoint
from sympy.core.assumptions import check_assumptions
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.immutable import ImmutableDenseMatrix
from numpy import empty
from sympy import Sum, Mul, Add, MatMul, transpose, trace
from sympy.strategies.traverse import bottom_up
from .applyfunc import ElementwiseApplyFunction
from sympy import MatrixBase
from sympy import Sum, symbols, Dummy
from sympy.core.expr import ExprBuilder
from sympy.core.expr import ExprBuilder
from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
from ...tensor.array.expressions.array_expressions import ArrayContraction
from sympy.matrices.expressions.slice import MatrixSlice
from sympy import MatrixBase
from sympy.matrices.expressions.slice import MatrixSlice

Basic._constructor_postprocessor_mapping[MatrixExpr] = {
    "Mul": [get_postprocessor(Mul)],
    "Add": [get_postprocessor(Add)],
}

def _matrix_derivative(expr, x):
    from sympy.tensor.array.array_derivatives import ArrayDerivative
    lines = expr._eval_derivative_matrix_lines(x)

    parts = [i.build() for i in lines]

    from sympy.tensor.array.expressions.conv_array_to_matrix import convert_array_to_matrix

    parts = [[convert_array_to_matrix(j) for j in i] for i in parts]

    def _get_shape(elem):
        if isinstance(elem, MatrixExpr):
            return elem.shape
        return 1, 1

    def get_rank(parts):
        return sum([j not in (1, None) for i in parts for j in _get_shape(i)])

    ranks = [get_rank(i) for i in parts]
    rank = ranks[0]

    def contract_one_dims(parts):
        if len(parts) == 1:
            return parts[0]
        else:
            p1, p2 = parts[:2]
            if p2.is_Matrix:
                p2 = p2.T
            if p1 == Identity(1):
                pbase = p2
            elif p2 == Identity(1):
                pbase = p1
            else:
                pbase = p1*p2
            if len(parts) == 2:
                return pbase
            else:  # len(parts) > 2
                if pbase.is_Matrix:
                    raise ValueError("")
                return pbase*Mul.fromiter(parts[2:])

    if rank <= 2:
        return Add.fromiter([contract_one_dims(i) for i in parts])

    return ArrayDerivative(expr, x)
