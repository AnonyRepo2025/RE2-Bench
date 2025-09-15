from __future__ import print_function, division
from functools import wraps, reduce
from operator import mul
from sympy.core import (
    S, Basic, Expr, I, Integer, Add, Mul, Dummy, Tuple
)
from sympy.core.basic import preorder_traversal
from sympy.core.compatibility import iterable, ordered
from sympy.core.decorators import _sympifyit
from sympy.core.function import Derivative
from sympy.core.mul import _keep_coeff
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import (
    OperationNotSupported, DomainError,
    CoercionFailed, UnificationFailed,
    GeneratorsNeeded, PolynomialError,
    MultivariatePolynomialError,
    ExactQuotientFailed,
    PolificationFailed,
    ComputationFailed,
    GeneratorsError,
)
from sympy.polys.polyutils import (
    basic_from_dict,
    _sort_gens,
    _unify_gens,
    _dict_reorder,
    _dict_from_expr,
    _parallel_dict_from_expr,
)
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, sift, public, filldedent
from sympy.utilities.exceptions import SymPyDeprecationWarning
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.relational import Equality
from sympy.simplify.simplify import simplify
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import factor_terms
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.rings import xring
from sympy.polys.dispersion import dispersionset
from sympy.polys.dispersion import dispersion
from sympy.functions.elementary.complexes import sign
from sympy.core.add import Add
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.simplify.simplify import bottom_up
from sympy.polys.rings import PolyRing
from sympy.polys.rings import xring
from sympy.polys.rings import xring
from sympy.core.numbers import ilcm
from sympy.core.exprtools import factor_nc



def _symbolic_factor_list(expr, opt, method):
    """Helper function for :func:`_symbolic_factor`. """
    coeff, factors = S.One, []

    args = [i._eval_factor() if hasattr(i, '_eval_factor') else i
        for i in Mul.make_args(expr)]
    for arg in args:
        if arg.is_Number:
            coeff *= arg
            continue
        elif arg.is_Pow:
            base, exp = arg.args
            if base.is_Number and exp.is_Number:
                coeff *= arg
                continue
            if base.is_Number:
                factors.append((base, exp))
                continue
        else:
            base, exp = arg, S.One

        try:
            poly, _ = _poly_from_expr(base, opt)
        except PolificationFailed as exc:
            factors.append((exc.expr, exp))
        else:
            func = getattr(poly, method + '_list')

            _coeff, _factors = func()
            if _coeff is not S.One:
                if exp.is_Integer:
                    coeff *= _coeff**exp
                elif _coeff.is_positive:
                    factors.append((_coeff, exp))
                else:
                    _factors.append((_coeff, S.One))

            if exp is S.One:
                factors.extend(_factors)
            elif exp.is_integer:
                factors.extend([(f, k*exp) for f, k in _factors])
            else:
                other = []

                for f, k in _factors:
                    if f.as_expr().is_positive:
                        factors.append((f, k*exp))
                    else:
                        other.append((f, k))

                factors.append((_factors_product(other), exp))
    if method == 'sqf':
        factors = [(reduce(mul, (f for f, _ in factors if _ == k)), k)
                   for k in set(i for _, i in factors)]

    return coeff, factors
