from __future__ import print_function, division
from operator import add, mul, lt, le, gt, ge
from sympy.core.compatibility import is_sequence, reduce, string_types
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import CantSympify, sympify
from sympy.polys.rings import PolyElement
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import _parallel_dict_from_expr
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.constructor import construct_domain
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from sympy.polys.rings import PolyRing
from sympy.polys.rings import PolyRing

_field_cache = {}

class FracElement(DomainElement, DefaultPrinting, CantSympify):
    _hash = None
    __bool__ = __nonzero__
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    def __eq__(f, g):
        if isinstance(g, FracElement) and f.field == g.field:
            return f.numer == g.numer and f.denom == g.denom
        else:
            return f.numer == g and f.denom == f.field.ring.one