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
from sympy.utilities import group, sift, public
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.constructor import construct_domain
from sympy.polys import polyoptions as options
from sympy.core.compatibility import iterable, range
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



class Poly(Expr):
    __slots__ = ['rep', 'gens']
    is_commutative = True
    is_Poly = True
    _eval_derivative = diff
    _eval_diff = diff
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __bool__ = __nonzero__
    @property
    def free_symbols(self):
        """
        Free symbols of a polynomial expression.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x**2 + 1).free_symbols
        {x}
        >>> Poly(x**2 + y).free_symbols
        {x, y}
        >>> Poly(x**2 + y, x).free_symbols
        {x, y}
        >>> Poly(x**2 + y, x, z).free_symbols
        {x, y}

        """
        symbols = set()
        gens = self.gens
        for i in range(len(gens)):
            for monom in self.monoms():
                if monom[i]:
                    symbols |= gens[i].free_symbols
                    break

        return symbols | self.free_symbols_in_domain