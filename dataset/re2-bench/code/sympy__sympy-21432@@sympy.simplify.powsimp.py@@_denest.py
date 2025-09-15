from collections import defaultdict
from sympy.core.function import expand_log, count_ops
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.compatibility import ordered, default_sort_key, reduce
from sympy.core.numbers import Integer, Rational
from sympy.core.mul import prod, _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.simplify import posify
from sympy.simplify.simplify import logcombine

_y = Dummy('y')

        def _denest(b, e):
            if not isinstance(b, (Pow, exp)):
                return b.is_positive, Pow(b, e, evaluate=False)
            return _denest(b.base, b.exp*e)
