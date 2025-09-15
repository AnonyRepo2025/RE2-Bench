from __future__ import print_function, division
from collections import defaultdict
from sympy import SYMPY_DEBUG
from sympy.core.evaluate import global_evaluate
from sympy.core.compatibility import iterable, ordered, default_sort_key
from sympy.core import expand_power_base, sympify, Add, S, Mul, Derivative, Pow, symbols, expand_mul
from sympy.core.numbers import Rational
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.mul import _keep_coeff, _unevaluated_Mul
from sympy.core.function import _mexpand
from sympy.core.add import _unevaluated_Add
from sympy.functions import exp, sqrt, log
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.simplify import signsimp
from sympy.simplify.simplify import nsimplify
from sympy.simplify.powsimp import powsimp, powdenest

expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand

def fraction(expr, exact=False):
    """Returns a pair with expression's numerator and denominator.
       If the given expression is not a fraction then this function
       will return the tuple (expr, 1).

       This function will not make any attempt to simplify nested
       fractions or to do any term rewriting at all.

       If only one of the numerator/denominator pair is needed then
       use numer(expr) or denom(expr) functions respectively.

       >>> from sympy import fraction, Rational, Symbol
       >>> from sympy.abc import x, y

       >>> fraction(x/y)
       (x, y)
       >>> fraction(x)
       (x, 1)

       >>> fraction(1/y**2)
       (1, y**2)

       >>> fraction(x*y/2)
       (x*y, 2)
       >>> fraction(Rational(1, 2))
       (1, 2)

       This function will also work fine with assumptions:

       >>> k = Symbol('k', negative=True)
       >>> fraction(x * y**k)
       (x, y**(-k))

       If we know nothing about sign of some exponent and 'exact'
       flag is unset, then structure this exponent's structure will
       be analyzed and pretty fraction will be returned:

       >>> from sympy import exp, Mul
       >>> fraction(2*x**(-y))
       (2, x**y)

       >>> fraction(exp(-x))
       (1, exp(x))

       >>> fraction(exp(-x), exact=True)
       (exp(-x), 1)

       The `exact` flag will also keep any unevaluated Muls from
       being evaluated:

       >>> u = Mul(2, x + 1, evaluate=False)
       >>> fraction(u)
       (2*x + 2, 1)
       >>> fraction(u, exact=True)
       (2*(x  + 1), 1)
    """
    expr = sympify(expr)

    numer, denom = [], []

    for term in Mul.make_args(expr):
        if term.is_commutative and (term.is_Pow or term.func is exp):
            b, ex = term.as_base_exp()
            if ex.is_negative:
                if ex is S.NegativeOne:
                    denom.append(b)
                elif exact:
                    if ex.is_constant():
                        denom.append(Pow(b, -ex))
                    else:
                        numer.append(term)
                else:
                    denom.append(Pow(b, -ex))
            elif ex.is_positive:
                numer.append(term)
            elif not exact and ex.is_Mul:
                n, d = term.as_numer_denom()
                numer.append(n)
                denom.append(d)
            else:
                numer.append(term)
        elif term.is_Rational:
            n, d = term.as_numer_denom()
            numer.append(n)
            denom.append(d)
        else:
            numer.append(term)
    if exact:
        return Mul(*numer, evaluate=False), Mul(*denom, evaluate=False)
    else:
        return Mul(*numer), Mul(*denom)
