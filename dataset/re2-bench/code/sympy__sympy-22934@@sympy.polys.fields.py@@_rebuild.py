from typing import Any, Dict as tDict
from functools import reduce
from operator import add, mul, lt, le, gt, ge
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Exp1
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import CantSympify, sympify
from sympy.functions.elementary.exponential import ExpBase
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.constructor import construct_domain
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import _parallel_dict_from_expr
from sympy.polys.rings import PolyElement
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
from sympy.polys.rings import PolyRing
from sympy.polys.rings import PolyRing

_field_cache = {}  # type: tDict[Any, Any]

class FracField(DefaultPrinting):
    """Multivariate distributed rational function field. """

    def __new__(cls, symbols, domain, order=lex):
        from sympy.polys.rings import PolyRing
        ring = PolyRing(symbols, domain, order)
        symbols = ring.symbols
        ngens = ring.ngens
        domain = ring.domain
        order = ring.order

        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)
        obj = _field_cache.get(_hash_tuple)

        if obj is None:
            obj = object.__new__(cls)
            obj._hash_tuple = _hash_tuple
            obj._hash = hash(_hash_tuple)
            obj.ring = ring
            obj.dtype = type("FracElement", (FracElement,), {"field": obj})
            obj.symbols = symbols
            obj.ngens = ngens
            obj.domain = domain
            obj.order = order

            obj.zero = obj.dtype(ring.zero)
            obj.one = obj.dtype(ring.one)

            obj.gens = obj._gens()

            for symbol, generator in zip(obj.symbols, obj.gens):
                if isinstance(symbol, Symbol):
                    name = symbol.name

                    if not hasattr(obj, name):
                        setattr(obj, name, generator)

            _field_cache[_hash_tuple] = obj

        return obj

    def _gens(self):
        """Return a list of polynomial generators. """
        return tuple([ self.dtype(gen) for gen in self.ring.gens ])

    def __getnewargs__(self):
        return (self.symbols, self.domain, self.order)

    def __hash__(self):
        return self._hash

    def index(self, gen):
        if isinstance(gen, self.dtype):
            return self.ring.index(gen.to_poly())
        else:
            raise ValueError("expected a %s, got %s instead" % (self.dtype,gen))

    def __eq__(self, other):
        return isinstance(other, FracField) and \
            (self.symbols, self.ngens, self.domain, self.order) == \
            (other.symbols, other.ngens, other.domain, other.order)

    def __ne__(self, other):
        return not self == other

    def raw_new(self, numer, denom=None):
        return self.dtype(numer, denom)
    def new(self, numer, denom=None):
        if denom is None: denom = self.ring.one
        numer, denom = numer.cancel(denom)
        return self.raw_new(numer, denom)

    def domain_new(self, element):
        return self.domain.convert(element)

    def ground_new(self, element):
        try:
            return self.new(self.ring.ground_new(element))
        except CoercionFailed:
            domain = self.domain

            if not domain.is_Field and domain.has_assoc_Field:
                ring = self.ring
                ground_field = domain.get_field()
                element = ground_field.convert(element)
                numer = ring.ground_new(ground_field.numer(element))
                denom = ring.ground_new(ground_field.denom(element))
                return self.raw_new(numer, denom)
            else:
                raise

    def field_new(self, element):
        if isinstance(element, FracElement):
            if self == element.field:
                return element

            if isinstance(self.domain, FractionField) and \
                self.domain.field == element.field:
                return self.ground_new(element)
            elif isinstance(self.domain, PolynomialRing) and \
                self.domain.ring.to_field() == element.field:
                return self.ground_new(element)
            else:
                raise NotImplementedError("conversion")
        elif isinstance(element, PolyElement):
            denom, numer = element.clear_denoms()

            if isinstance(self.domain, PolynomialRing) and \
                numer.ring == self.domain.ring:
                numer = self.ring.ground_new(numer)
            elif isinstance(self.domain, FractionField) and \
                numer.ring == self.domain.field.to_ring():
                numer = self.ring.ground_new(numer)
            else:
                numer = numer.set_ring(self.ring)

            denom = self.ring.ground_new(denom)
            return self.raw_new(numer, denom)
        elif isinstance(element, tuple) and len(element) == 2:
            numer, denom = list(map(self.ring.ring_new, element))
            return self.new(numer, denom)
        elif isinstance(element, str):
            raise NotImplementedError("parsing")
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)

    __call__ = field_new

    def _rebuild_expr(self, expr, mapping):
        domain = self.domain
        powers = tuple((gen, gen.as_base_exp()) for gen in mapping.keys()
            if gen.is_Pow or isinstance(gen, ExpBase))

        def _rebuild(expr):
            generator = mapping.get(expr)

            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            elif expr.is_Pow or isinstance(expr, (ExpBase, Exp1)):
                b, e = expr.as_base_exp()
                # look for bg**eg whose integer power may be b**e
                for gen, (bg, eg) in powers:
                    if bg == b and Mod(e, eg) == 0:
                        return mapping.get(gen)**int(e/eg)
                if e.is_Integer and e is not S.One:
                    return _rebuild(b)**int(e)
            elif mapping.get(1/expr) is not None:
                return 1/mapping.get(1/expr)

            try:
                return domain.convert(expr)
            except CoercionFailed:
                if not domain.is_Field and domain.has_assoc_Field:
                    return domain.get_field().convert(expr)
                else:
                    raise

        return _rebuild(sympify(expr))

    def from_expr(self, expr):
        mapping = dict(list(zip(self.symbols, self.gens)))

        try:
            frac = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            raise ValueError("expected an expression convertible to a rational function in %s, got %s" % (self, expr))
        else:
            return self.field_new(frac)

    def to_domain(self):
        return FractionField(self)

    def to_ring(self):
        from sympy.polys.rings import PolyRing
        return PolyRing(self.symbols, self.domain, self.order)
