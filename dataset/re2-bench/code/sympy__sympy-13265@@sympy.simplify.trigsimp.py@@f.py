from __future__ import print_function, division
from collections import defaultdict
from sympy.core.cache import cacheit
from sympy.core import (sympify, Basic, S, Expr, expand_mul, factor_terms,
    Mul, Dummy, igcd, FunctionClass, Add, symbols, Wild, expand)
from sympy.core.compatibility import reduce, iterable
from sympy.core.numbers import I, Integer
from sympy.core.function import count_ops, _mexpand
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.polys import Poly
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.polys.domains import ZZ
from sympy.polys import factor, cancel, parallel_poly_from_expr
from sympy.utilities.misc import debug
from sympy.simplify.ratsimp import ratsimpmodprime
from sympy.simplify.fu import fu
from sympy.simplify.fu import hyper_as_trig, TR2i
from sympy.simplify.simplify import bottom_up
from sympy.simplify.fu import TR10i
from sympy.simplify.fu import hyper_as_trig
from sympy.simplify.simplify import bottom_up
from sympy.simplify.fu import (
        TR1, TR2, TR3, TR2i, TR10, L, TR10i,
        TR8, TR6, TR15, TR16, TR111, TR5, TRmorrie, TR11, TR14, TR22,
        TR12)
from sympy.core.compatibility import _nodes

_trigs = (TrigonometricFunction, HyperbolicFunction)
_trigpat = None
_idn = lambda x: x
_midn = lambda x: -x
_one = lambda x: S.One

def exptrigsimp(expr):
    """
    Simplifies exponential / trigonometric / hyperbolic functions.

    Examples
    ========

    >>> from sympy import exptrigsimp, exp, cosh, sinh
    >>> from sympy.abc import z

    >>> exptrigsimp(exp(z) + exp(-z))
    2*cosh(z)
    >>> exptrigsimp(cosh(z) - sinh(z))
    exp(-z)
    """
    from sympy.simplify.fu import hyper_as_trig, TR2i
    from sympy.simplify.simplify import bottom_up

    def exp_trig(e):
        # select the better of e, and e rewritten in terms of exp or trig
        # functions
        choices = [e]
        if e.has(*_trigs):
            choices.append(e.rewrite(exp))
        choices.append(e.rewrite(cos))
        return min(*choices, key=count_ops)
    newexpr = bottom_up(expr, exp_trig)

    def f(rv):
        if not rv.is_Mul:
            return rv
        rvd = rv.as_powers_dict()
        newd = rvd.copy()

        def signlog(expr, sign=1):
            if expr is S.Exp1:
                return sign, 1
            elif isinstance(expr, exp):
                return sign, expr.args[0]
            elif sign == 1:
                return signlog(-expr, sign=-1)
            else:
                return None, None

        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                # k == c*(1 + sign*E**x)
                c = k.args[0]
                sign, x = signlog(k.args[1]/c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x*m/2:
                    # sinh and cosh
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2*c*cosh(x/2)] += m
                    else:
                        newd[-2*c*sinh(x/2)] += m
                elif newd[1 - sign*S.Exp1**x] == -m:
                    # tanh
                    del newd[1 - sign*S.Exp1**x]
                    if sign == 1:
                        newd[-c/tanh(x/2)] += m
                    else:
                        newd[-c*tanh(x/2)] += m
                else:
                    newd[1 + sign*S.Exp1**x] += m
                    newd[c] += m

        return Mul(*[k**newd[k] for k in newd])
    newexpr = bottom_up(newexpr, f)

    # sin/cos and sinh/cosh ratios to tan and tanh, respectively
    if newexpr.has(HyperbolicFunction):
        e, f = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)

    # can we ever generate an I where there was none previously?
    if not (newexpr.has(I) and not expr.has(I)):
        expr = newexpr
    return expr
