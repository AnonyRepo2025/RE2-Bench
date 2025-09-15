from __future__ import print_function, division
from sympy import Derivative, Expr, Integer, oo, Mul, expand, Add
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.matrices import eye
from sympy.physics.quantum.state import KetBase, BraBase
from sympy.physics.quantum.state import Wavefunction

__all__ = [
    'Operator',
    'HermitianOperator',
    'UnitaryOperator',
    'IdentityOperator',
    'OuterProduct',
    'DifferentialOperator'
]

class IdentityOperator(Operator):
    def __mul__(self, other):

        if isinstance(other, (Operator, Dagger)):
            return other

        return Mul(self, other)