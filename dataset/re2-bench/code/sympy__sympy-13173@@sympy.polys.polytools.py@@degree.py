from __future__ import print_function, division
from sympy.core import (
    S, Basic, Expr, I, Integer, Add, Mul, Dummy, Tuple
)
from sympy.core.mul import _keep_coeff
from sympy.core.symbol import Symbol
from sympy.core.basic import preorder_traversal
from sympy.core.relational import Relational
from sympy.core.sympify import sympify
from sympy.core.decorators import _sympifyit
from sympy.core.function import Derivative
from sympy.logic.boolalg import BooleanAtom
from sympy.polys.polyclasses import DMP
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
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
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
from sympy.utilities import group, sift, public, filldedent
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.constructor import construct_domain
from sympy.polys import polyoptions as options
from sympy.core.compatibility import iterable, range, ordered
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.relational import Equality
from sympy.simplify.simplify import simplify
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import factor_terms
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.rings import xring
from sympy.polys.dispersion import dispersionset
from sympy.polys.dispersion import dispersion
from sympy.core.add import Add
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.polys.rings import PolyRing
from sympy.polys.rings import xring
from sympy.polys.rings import xring
from sympy.core.numbers import ilcm
from sympy.core.exprtools import factor_nc



def degree(f, gen=0):
    """
    Return the degree of ``f`` in the given variable.

    The degree of 0 is negative infinity.

    Examples
    ========

    >>> from sympy import degree
    >>> from sympy.abc import x, y

    >>> degree(x**2 + y*x + 1, gen=x)
    2
    >>> degree(x**2 + y*x + 1, gen=y)
    1
    >>> degree(0, x)
    -oo

    See also
    ========
    total_degree
    degree_list
    """

    f = sympify(f, strict=True)
    if f.is_Poly:
        p = f
        isNum = p.as_expr().is_Number
    else:
        isNum = f.is_Number
        if not isNum:
            p, _ = poly_from_expr(f)

    if isNum:
        return S.Zero if f else S.NegativeInfinity

    if not sympify(gen, strict=True).is_Number:
        if f.is_Poly and gen not in p.gens:
            # try recast without explicit gens
            p, _ = poly_from_expr(f.as_expr())
        if gen not in p.gens:
            return S.Zero
    elif not f.is_Poly and len(f.free_symbols) > 1:
        raise TypeError(filldedent('''
         A symbolic generator of interest is required for a multivariate
         expression like func = %s, e.g. degree(func, gen = %s) instead of
         degree(func, gen = %s).
        ''' % (f, next(ordered(f.free_symbols)), gen)))

    return Integer(p.degree(gen))
