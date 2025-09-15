from __future__ import print_function, division
from sympy import Number
from sympy.core import Mul, Basic, sympify, Add
from sympy.core.compatibility import range
from sympy.functions import adjoint
from sympy.matrices.expressions.transpose import transpose
from sympy.strategies import (rm_id, unpack, typed, flatten, exhaust,
        do_one, new)
from sympy.matrices.expressions.matexpr import (MatrixExpr, ShapeError,
        Identity, ZeroMatrix)
from sympy.matrices.matrices import MatrixBase
from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
from sympy.core.symbol import Dummy
from sympy.concrete.summations import Sum
from sympy.matrices import ImmutableMatrix
from sympy.matrices.expressions.determinant import Determinant
from .trace import trace
from sympy.matrices.expressions.inverse import Inverse

rules = (any_zeros, remove_ids, xxinv, unpack, rm_id(lambda x: x == 1),
         merge_explicit, factor_in_front, flatten)
canonicalize = exhaust(typed({MatMul: do_one(*rules)}))
handlers_dict['MatMul'] = refine_MatMul

class MatMul(MatrixExpr):
    is_MatMul = True
    def as_coeff_matrices(self):
        scalars = [x for x in self.args if not x.is_Matrix]
        matrices = [x for x in self.args if x.is_Matrix]
        coeff = Mul(*scalars)

        return coeff, matrices
    def args_cnc(self, **kwargs):
        coeff, matrices = self.as_coeff_matrices()
        coeff_c, coeff_nc = coeff.args_cnc(**kwargs)
        if coeff_c == [1]:
            coeff_c = []
        elif coeff_c == set([1]):
            coeff_c = set()

        return coeff_c, coeff_nc + matrices