from __future__ import print_function, division
from sympy.matrices.common import NonSquareMatrixError
from .matexpr import MatrixExpr, Identity, ZeroMatrix
from sympy.core import S
from sympy.core.sympify import _sympify
from sympy.matrices import MatrixBase
from sympy.matrices.common import NonInvertibleMatrixError
from .permutation import PermutationMatrix
from sympy.matrices.expressions import MatMul
from sympy.matrices.expressions import Inverse
from sympy import Pow
from sympy.core.expr import ExprBuilder
from sympy.codegen.array_utils import CodegenArrayContraction, CodegenArrayTensorProduct
from .matmul import MatMul
from .inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixElement



class MatPow(MatrixExpr):
    def _entry(self, i, j, **kwargs):
        from sympy.matrices.expressions import MatMul
        A = self.doit()
        if isinstance(A, MatPow):
            # We still have a MatPow, make an explicit MatMul out of it.
            if not A.base.is_square:
                raise NonSquareMatrixError("Power of non-square matrix %s" % A.base)
            elif A.exp.is_Integer and A.exp.is_positive:
                A = MatMul(*[A.base for k in range(A.exp)])
            #elif A.exp.is_Integer and self.exp.is_negative:
            # Note: possible future improvement: in principle we can take
            # positive powers of the inverse, but carefully avoid recursion,
            # perhaps by adding `_entry` to Inverse (as it is our subclass).
            # T = A.base.as_explicit().inverse()
            # A = MatMul(*[T for k in range(-A.exp)])
            else:
                # Leave the expression unevaluated:
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)
        return A._entry(i, j)