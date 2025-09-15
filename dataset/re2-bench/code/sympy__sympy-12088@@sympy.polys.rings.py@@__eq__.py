from __future__ import print_function, division
from operator import add, mul, lt, le, gt, ge
from types import GeneratorType
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.numbers import igcd, oo
from sympy.core.sympify import CantSympify, sympify
from sympy.core.compatibility import is_sequence, reduce, string_types, range
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.compatibility import IPolys
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,
                                   _parallel_dict_from_expr)
from sympy.polys.polyerrors import (
    CoercionFailed, GeneratorsError,
    ExactQuotientFailed, MultivariatePolynomialError)
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.polyoptions import (Domain as DomainOpt,
                                     Order as OrderOpt, build_options)
from sympy.polys.densebasic import dmp_to_dict, dmp_from_dict
from sympy.polys.constructor import construct_domain
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from sympy.polys.fields import FracField

_ring_cache = {}

class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):
    _hash = None
    __floordiv__ = __div__ = __truediv__
    __rfloordiv__ = __rdiv__ = __rtruediv__
    rem_ground = trunc_ground
    def __eq__(p1, p2):
        """Equality test for polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = (x + y)**2 + (x - y)**2
        >>> p1 == 4*x*y
        False
        >>> p1 == 2*(x**2 + y**2)
        True

        """
        if not p2:
            return not p1
        elif isinstance(p2, PolyElement) and p2.ring == p1.ring:
            return dict.__eq__(p1, p2)
        elif len(p1) > 1:
            return False
        else:
            return p1.get(p1.ring.zero_monom) == p2