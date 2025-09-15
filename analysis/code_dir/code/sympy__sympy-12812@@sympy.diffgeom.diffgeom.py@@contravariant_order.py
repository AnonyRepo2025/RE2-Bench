from __future__ import print_function, division
from itertools import permutations
from sympy.matrices import Matrix
from sympy.core import Basic, Expr, Dummy, Function, sympify, diff, Pow, Mul, Add, symbols, Tuple
from sympy.core.compatibility import range
from sympy.core.numbers import Zero
from sympy.solvers import solve
from sympy.functions import factorial
from sympy.simplify import simplify
from sympy.core.compatibility import reduce
from sympy.combinatorics import Permutation
from sympy.tensor.array import ImmutableDenseNDimArray

def contravariant_order(expr, _strict=False):
    if isinstance(expr, Add):
        orders = [contravariant_order(e) for e in expr.args]
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing contravariant fields of varying order.')
        return orders[0]
    elif isinstance(expr, Mul):
        orders = [contravariant_order(e) for e in expr.args]
        not_zero = [o for o in orders if o != 0]
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between vectors.')
        return 0 if not not_zero else not_zero[0]
    elif isinstance(expr, Pow):
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError('Misformed expression containing a power of a vector.')
        return 0
    elif isinstance(expr, BaseVectorField):
        return 1
    elif isinstance(expr, TensorProduct):
        return sum((contravariant_order(a) for a in expr.args))
    elif not _strict or expr.atoms(BaseScalarField):
        return 0
    else:
        return -1