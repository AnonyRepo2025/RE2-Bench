from __future__ import print_function, division
from collections import defaultdict
from sympy.core import Basic, S, Add, Mul, Pow, Symbol, sympify, expand_mul, expand_func, Function, Dummy, Expr, factor_terms, symbols, expand_power_exp
from sympy.core.compatibility import iterable, ordered, range, as_int
from sympy.core.numbers import Float, I, pi, Rational, Integer
from sympy.core.function import expand_log, count_ops, _mexpand, _coeff_isneg, nfloat
from sympy.core.rules import Transform
from sympy.core.evaluate import global_evaluate
from sympy.functions import gamma, exp, sqrt, log, exp_polar, piecewise_fold
from sympy.core.sympify import _sympify
from sympy.functions.elementary.exponential import ExpBase
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.complexes import unpolarify
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.combinatorial.factorials import CombinatorialFunction
from sympy.functions.special.bessel import besselj, besseli, besselk, jn, bessely
from sympy.utilities.iterables import has_variety
from sympy.simplify.radsimp import radsimp, fraction
from sympy.simplify.trigsimp import trigsimp, exptrigsimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.combsimp import combsimp
from sympy.polys import together, cancel, factor
import mpmath
from sympy.simplify.hyperexpand import hyperexpand
from sympy.functions.special.bessel import BesselBase
from sympy import Sum, Product
from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.concrete.summations import Sum
from sympy.core.exprtools import factor_terms
from sympy.concrete.summations import Sum
from sympy.concrete.summations import Sum
from sympy import Mul
from sympy.concrete.products import Product
from sympy.concrete.products import Product
from sympy.polys.numberfields import _minimal_polynomial_sq
from sympy.solvers import solve

def logcombine(expr, force=False):

    def f(rv):
        if not (rv.is_Add or rv.is_Mul):
            return rv

        def gooda(a):
            return a is not S.NegativeOne and (a.is_real or (force and a.is_real is not False))

        def goodlog(l):
            a = l.args[0]
            return a.is_positive or (force and a.is_nonpositive is not False)
        other = []
        logs = []
        log1 = defaultdict(list)
        for a in Add.make_args(rv):
            if isinstance(a, log) and goodlog(a):
                log1[()].append(([], a))
            elif not a.is_Mul:
                other.append(a)
            else:
                ot = []
                co = []
                lo = []
                for ai in a.args:
                    if ai.is_Rational and ai < 0:
                        ot.append(S.NegativeOne)
                        co.append(-ai)
                    elif isinstance(ai, log) and goodlog(ai):
                        lo.append(ai)
                    elif gooda(ai):
                        co.append(ai)
                    else:
                        ot.append(ai)
                if len(lo) > 1:
                    logs.append((ot, co, lo))
                elif lo:
                    log1[tuple(ot)].append((co, lo[0]))
                else:
                    other.append(a)
        if not logs and all((len(log1[k]) == 1 and log1[k][0] == [] for k in log1)):
            return rv
        for o, e, l in logs:
            l = list(ordered(l))
            e = log(l.pop(0).args[0] ** Mul(*e))
            while l:
                li = l.pop(0)
                e = log(li.args[0] ** e)
            c, l = (Mul(*o), e)
            if isinstance(l, log):
                log1[c,].append(([], l))
            else:
                other.append(c * l)
        for k in list(log1.keys()):
            log1[Mul(*k)] = log(logcombine(Mul(*[l.args[0] ** Mul(*c) for c, l in log1.pop(k)]), force=force), evaluate=False)
        for k in ordered(list(log1.keys())):
            if not k in log1:
                continue
            if -k in log1:
                num, den = (k, -k)
                if num.count_ops() > den.count_ops():
                    num, den = (den, num)
                other.append(num * log(log1.pop(num).args[0] / log1.pop(den).args[0], evaluate=False))
            else:
                other.append(k * log1.pop(k))
        return Add(*other)
    return bottom_up(expr, f)