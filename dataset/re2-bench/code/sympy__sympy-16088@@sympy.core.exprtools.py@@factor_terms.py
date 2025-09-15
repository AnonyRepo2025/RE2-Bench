from __future__ import print_function, division
from sympy.core.add import Add
from sympy.core.compatibility import iterable, is_sequence, SYMPY_INTS, range
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.power import Pow
from sympy.core.basic import Basic, preorder_traversal
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.core.numbers import Rational, Integer, Number, I
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.coreerrors import NonCommutativeExpression
from sympy.core.containers import Tuple, Dict
from sympy.utilities import default_sort_key
from sympy.utilities.iterables import (common_prefix, common_suffix,
        variations, ordered)
from collections import defaultdict
from sympy.simplify.simplify import powsimp
from sympy.polys import gcd, factor
from sympy.concrete.summations import Sum
from sympy.integrals.integrals import Integral
from sympy import Dummy
from sympy.polys.polytools import real_roots
from sympy.polys.polyroots import roots
from sympy.polys.polyerrors import PolynomialError

_eps = Dummy(positive=True)

def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):
    """Remove common factors from terms in all arguments without
    changing the underlying structure of the expr. No expansion or
    simplification (and no processing of non-commutatives) is performed.

    If radical=True then a radical common to all terms will be factored
    out of any Add sub-expressions of the expr.

    If clear=False (default) then coefficients will not be separated
    from a single Add if they can be distributed to leave one or more
    terms with integer coefficients.

    If fraction=True (default is False) then a common denominator will be
    constructed for the expression.

    If sign=True (default) then even if the only factor in common is a -1,
    it will be factored out of the expression.

    Examples
    ========

    >>> from sympy import factor_terms, Symbol
    >>> from sympy.abc import x, y
    >>> factor_terms(x + x*(2 + 4*y)**3)
    x*(8*(2*y + 1)**3 + 1)
    >>> A = Symbol('A', commutative=False)
    >>> factor_terms(x*A + x*A + x*y*A)
    x*(y*A + 2*A)

    When ``clear`` is False, a rational will only be factored out of an
    Add expression if all terms of the Add have coefficients that are
    fractions:

    >>> factor_terms(x/2 + 1, clear=False)
    x/2 + 1
    >>> factor_terms(x/2 + 1, clear=True)
    (x + 2)/2

    If a -1 is all that can be factored out, to *not* factor it out, the
    flag ``sign`` must be False:

    >>> factor_terms(-x - y)
    -(x + y)
    >>> factor_terms(-x - y, sign=False)
    -x - y
    >>> factor_terms(-2*x - 2*y, sign=False)
    -2*(x + y)

    See Also
    ========
    gcd_terms, sympy.polys.polytools.terms_gcd

    """
    def do(expr):
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral
        is_iterable = iterable(expr)

        if not isinstance(expr, Basic) or expr.is_Atom:
            if is_iterable:
                return type(expr)([do(i) for i in expr])
            return expr

        if expr.is_Pow or expr.is_Function or \
                is_iterable or not hasattr(expr, 'args_cnc'):
            args = expr.args
            newargs = tuple([do(i) for i in args])
            if newargs == args:
                return expr
            return expr.func(*newargs)

        if isinstance(expr, (Sum, Integral)):
            return _factor_sum_int(expr,
                radical=radical, clear=clear,
                fraction=fraction, sign=sign)

        cont, p = expr.as_content_primitive(radical=radical, clear=clear)
        if p.is_Add:
            list_args = [do(a) for a in Add.make_args(p)]
            # get a common negative (if there) which gcd_terms does not remove
            if all(a.as_coeff_Mul()[0].extract_multiplicatively(-1) is not None
                   for a in list_args):
                cont = -cont
                list_args = [-a for a in list_args]
            # watch out for exp(-(x+2)) which gcd_terms will change to exp(-x-2)
            special = {}
            for i, a in enumerate(list_args):
                b, e = a.as_base_exp()
                if e.is_Mul and e != Mul(*e.args):
                    list_args[i] = Dummy()
                    special[list_args[i]] = a
            # rebuild p not worrying about the order which gcd_terms will fix
            p = Add._from_args(list_args)
            p = gcd_terms(p,
                isprimitive=True,
                clear=clear,
                fraction=fraction).xreplace(special)
        elif p.args:
            p = p.func(
                *[do(a) for a in p.args])
        rv = _keep_coeff(cont, p, clear=clear, sign=sign)
        return rv
    expr = sympify(expr)
    return do(expr)
