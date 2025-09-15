from __future__ import print_function, division
from sympy import (log, sqrt, pi, S, Dummy, Interval, sympify, gamma,
                   Piecewise, And, Eq, binomial, factorial, Sum, floor, Abs,
                   Lambda, Basic, lowergamma, erf, erfi,  erfinv, I, hyper,
                   uppergamma, sinh, atan, Ne, expint)
from sympy import beta as beta_fn
from sympy import cos, sin, tan, atan, exp, besseli, besselj, besselk
from sympy.external import import_module
from sympy.matrices import MatrixBase
from sympy.stats.crv import (SingleContinuousPSpace, SingleContinuousDistribution,
        ContinuousDistributionHandmade)
from sympy.stats.joint_rv import JointPSpace, CompoundDistribution
from sympy.stats.joint_rv_types import multivariate_rv
from sympy.stats.rv import _value_check, RandomSymbol
import random
from sympy import asin
from sympy.stats.joint_rv_types import MultivariateLaplaceDistribution
from sympy.stats.joint_rv_types import MultivariateNormalDistribution
from sympy import Max, Min
from scipy.stats import invgamma

oo = S.Infinity
__all__ = ['ContinuousRV',
'Arcsin',
'Benini',
'Beta',
'BetaNoncentral',
'BetaPrime',
'Cauchy',
'Chi',
'ChiNoncentral',
'ChiSquared',
'Dagum',
'Erlang',
'Exponential',
'FDistribution',
'FisherZ',
'Frechet',
'Gamma',
'GammaInverse',
'Gompertz',
'Gumbel',
'Kumaraswamy',
'Laplace',
'Logistic',
'LogNormal',
'Maxwell',
'Nakagami',
'Normal',
'Pareto',
'QuadraticU',
'RaisedCosine',
'Rayleigh',
'StudentT',
'ShiftedGompertz',
'Trapezoidal',
'Triangular',
'Uniform',
'UniformSum',
'VonMises',
'Weibull',
'WignerSemicircle'
]

class TriangularDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c')
    def _moment_generating_function(self, t):
        a, b, c = self.a, self.b, self.c
        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)) / (
        (b - a) * (c - a) * (b - c) * t ** 2)