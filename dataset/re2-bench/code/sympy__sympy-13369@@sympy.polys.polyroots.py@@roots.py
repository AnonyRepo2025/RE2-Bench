from __future__ import print_function, division
import math
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core import S, I, pi
from sympy.core.compatibility import ordered
from sympy.core.mul import expand_2arg, Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.core.numbers import Rational, igcd, comp
from sympy.core.exprtools import factor_terms
from sympy.core.logic import fuzzy_not
from sympy.ntheory import divisors, isprime, nextprime
from sympy.functions import exp, sqrt, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
    DomainError)
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.rationaltools import together
from sympy.simplify import simplify, powsimp
from sympy.utilities import public
from sympy.core.compatibility import reduce, range
from sympy.solvers.solvers import solve as _solve
from sympy.polys.polytools import to_rational_coeffs



def roots(f, *gens, **flags):
    """
    Computes symbolic roots of a univariate polynomial.

    Given a univariate polynomial f with symbolic coefficients (or
    a list of the polynomial's coefficients), returns a dictionary
    with its roots and their multiplicities.

    Only roots expressible via radicals will be returned.  To get
    a complete set of roots use RootOf class or numerical methods
    instead. By default cubic and quartic formulas are used in
    the algorithm. To disable them because of unreadable output
    set ``cubics=False`` or ``quartics=False`` respectively. If cubic
    roots are real but are expressed in terms of complex numbers
    (casus irreducibilis [1]) the ``trig`` flag can be set to True to
    have the solutions returned in terms of cosine and inverse cosine
    functions.

    To get roots from a specific domain set the ``filter`` flag with
    one of the following specifiers: Z, Q, R, I, C. By default all
    roots are returned (this is equivalent to setting ``filter='C'``).

    By default a dictionary is returned giving a compact result in
    case of multiple roots.  However to get a list containing all
    those roots set the ``multiple`` flag to True; the list will
    have identical roots appearing next to each other in the result.
    (For a given Poly, the all_roots method will give the roots in
    sorted numerical order.)

    Examples
    ========

    >>> from sympy import Poly, roots
    >>> from sympy.abc import x, y

    >>> roots(x**2 - 1, x)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-1, x)
    >>> roots(p)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-y, x, y)

    >>> roots(Poly(p, x))
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots(x**2 - y, x)
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots([1, 0, -1])
    {-1: 1, 1: 1}


    References
    ==========

    1. http://en.wikipedia.org/wiki/Cubic_function#Trigonometric_.28and_hyperbolic.29_method

    """
    from sympy.polys.polytools import to_rational_coeffs
    flags = dict(flags)

    auto = flags.pop('auto', True)
    cubics = flags.pop('cubics', True)
    trig = flags.pop('trig', False)
    quartics = flags.pop('quartics', True)
    quintics = flags.pop('quintics', False)
    multiple = flags.pop('multiple', False)
    filter = flags.pop('filter', None)
    predicate = flags.pop('predicate', None)

    if isinstance(f, list):
        if gens:
            raise ValueError('redundant generators given')

        x = Dummy('x')

        poly, i = {}, len(f) - 1

        for coeff in f:
            poly[i], i = sympify(coeff), i - 1

        f = Poly(poly, x, field=True)
    else:
        try:
            f = Poly(f, *gens, **flags)
            if f.length == 2 and f.degree() != 1:
                # check for foo**n factors in the constant
                n = f.degree()
                npow_bases = []
                expr = f.as_expr()
                con = expr.as_independent(*gens)[0]
                for p in Mul.make_args(con):
                    if p.is_Pow and not p.exp % n:
                        npow_bases.append(p.base**(p.exp/n))
                    else:
                        other.append(p)
                    if npow_bases:
                        b = Mul(*npow_bases)
                        B = Dummy()
                        d = roots(Poly(expr - con + B**n*Mul(*others), *gens,
                            **flags), *gens, **flags)
                        rv = {}
                        for k, v in d.items():
                            rv[k.subs(B, b)] = v
                        return rv

        except GeneratorsNeeded:
            if multiple:
                return []
            else:
                return {}

        if f.is_multivariate:
            raise PolynomialError('multivariate polynomials are not supported')

    def _update_dict(result, root, k):
        if root in result:
            result[root] += k
        else:
            result[root] = k

    def _try_decompose(f):
        """Find roots using functional decomposition. """
        factors, roots = f.decompose(), []

        for root in _try_heuristics(factors[0]):
            roots.append(root)

        for factor in factors[1:]:
            previous, roots = list(roots), []

            for root in previous:
                g = factor - Poly(root, f.gen)

                for root in _try_heuristics(g):
                    roots.append(root)

        return roots

    def _try_heuristics(f):
        """Find roots using formulas and some tricks. """
        if f.is_ground:
            return []
        if f.is_monomial:
            return [S(0)]*f.degree()

        if f.length() == 2:
            if f.degree() == 1:
                return list(map(cancel, roots_linear(f)))
            else:
                return roots_binomial(f)

        result = []

        for i in [-1, 1]:
            if not f.eval(i):
                f = f.quo(Poly(f.gen - i, f.gen))
                result.append(i)
                break

        n = f.degree()

        if n == 1:
            result += list(map(cancel, roots_linear(f)))
        elif n == 2:
            result += list(map(cancel, roots_quadratic(f)))
        elif f.is_cyclotomic:
            result += roots_cyclotomic(f)
        elif n == 3 and cubics:
            result += roots_cubic(f, trig=trig)
        elif n == 4 and quartics:
            result += roots_quartic(f)
        elif n == 5 and quintics:
            result += roots_quintic(f)

        return result

    (k,), f = f.terms_gcd()

    if not k:
        zeros = {}
    else:
        zeros = {S(0): k}

    coeff, f = preprocess_roots(f)

    if auto and f.get_domain().is_Ring:
        f = f.to_field()

    rescale_x = None
    translate_x = None

    result = {}

    if not f.is_ground:
        if not f.get_domain().is_Exact:
            for r in f.nroots():
                _update_dict(result, r, 1)
        elif f.degree() == 1:
            result[roots_linear(f)[0]] = 1
        elif f.length() == 2:
            roots_fun = roots_quadratic if f.degree() == 2 else roots_binomial
            for r in roots_fun(f):
                _update_dict(result, r, 1)
        else:
            _, factors = Poly(f.as_expr()).factor_list()
            if len(factors) == 1 and f.degree() == 2:
                for r in roots_quadratic(f):
                    _update_dict(result, r, 1)
            else:
                if len(factors) == 1 and factors[0][1] == 1:
                    if f.get_domain().is_EX:
                        res = to_rational_coeffs(f)
                        if res:
                            if res[0] is None:
                                translate_x, f = res[2:]
                            else:
                                rescale_x, f = res[1], res[-1]
                            result = roots(f)
                            if not result:
                                for root in _try_decompose(f):
                                    _update_dict(result, root, 1)
                        else:
                            for r in _try_heuristics(f):
                                _update_dict(result, r, 1)
                    else:
                        for root in _try_decompose(f):
                            _update_dict(result, root, 1)
                else:
                    for factor, k in factors:
                        for r in _try_heuristics(Poly(factor, f.gen, field=True)):
                            _update_dict(result, r, k)

    if coeff is not S.One:
        _result, result, = result, {}

        for root, k in _result.items():
            result[coeff*root] = k

    result.update(zeros)

    if filter not in [None, 'C']:
        handlers = {
            'Z': lambda r: r.is_Integer,
            'Q': lambda r: r.is_Rational,
            'R': lambda r: r.is_real,
            'I': lambda r: r.is_imaginary,
        }

        try:
            query = handlers[filter]
        except KeyError:
            raise ValueError("Invalid filter: %s" % filter)

        for zero in dict(result).keys():
            if not query(zero):
                del result[zero]

    if predicate is not None:
        for zero in dict(result).keys():
            if not predicate(zero):
                del result[zero]
    if rescale_x:
        result1 = {}
        for k, v in result.items():
            result1[k*rescale_x] = v
        result = result1
    if translate_x:
        result1 = {}
        for k, v in result.items():
            result1[k + translate_x] = v
        result = result1

    if not multiple:
        return result
    else:
        zeros = []

        for zero in ordered(result):
            zeros.extend([zero]*result[zero])

        return zeros
