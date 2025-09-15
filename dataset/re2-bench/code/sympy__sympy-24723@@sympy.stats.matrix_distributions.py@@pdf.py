from math import prod
from sympy.core.basic import Basic
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import multigamma
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,
                            MatrixSymbol, MatrixBase, Transpose, MatrixSet,
                            matrix2numpy)
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,
                            _symbol_converter, MatrixDomain, Distribution)
from sympy.external import import_module
from scipy import stats as scipy_stats
import numpy
import numpy
import logging
from sympy.matrices.dense import eye
import pymc
import pymc3 as pymc

_get_sample_class_matrixrv = {
    'scipy': SampleMatrixScipy,
    'pymc3': SampleMatrixPymc,
    'pymc': SampleMatrixPymc,
    'numpy': SampleMatrixNumpy
}

class MatrixNormalDistribution(MatrixDistribution):
    _argnames = ('location_matrix', 'scale_matrix_1', 'scale_matrix_2')
    def pdf(self, x):
        M, U, V = self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        n, p = M.shape
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        term1 = Inverse(V)*Transpose(x - M)*Inverse(U)*(x - M)
        num = exp(-Trace(term1)/S(2))
        den = (2*pi)**(S(n*p)/2) * Determinant(U)**(S(p)/2) * Determinant(V)**(S(n)/2)
        return num/den