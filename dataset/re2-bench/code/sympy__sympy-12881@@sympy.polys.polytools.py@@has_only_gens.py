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
    def has_only_gens(f, *gens):
        """
        Return ``True`` if ``Poly(f, *gens)`` retains ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x*y + 1, x, y, z).has_only_gens(x, y)
        True
        >>> Poly(x*y + z, x, y, z).has_only_gens(x, y)
        False

        """
        indices = set()

        for gen in gens:
            try:
                index = f.gens.index(gen)
            except ValueError:
                raise GeneratorsError(
                    "%s doesn't have %s as generator" % (f, gen))
            else:
                indices.add(index)

        for monom in f.monoms():
            for i, elt in enumerate(monom):
                if i not in indices and elt:
                    return False

        return True
    def monoms(f, order=None):
        """
        Returns all non-zero monomials from ``f`` in lex order.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x*y**2 + x*y + 3*y, x, y).monoms()
        [(2, 0), (1, 2), (1, 1), (0, 1)]

        See Also
        ========
        all_monoms

        """
        return f.rep.monoms(order=order)