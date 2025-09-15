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
    expr = sympify(expr)
    numer, denom = ([], [])
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
        return (Mul(*numer, evaluate=False), Mul(*denom, evaluate=False))
    else:
        return (Mul(*numer), Mul(*denom))