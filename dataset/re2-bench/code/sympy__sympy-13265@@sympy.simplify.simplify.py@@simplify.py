from __future__ import print_function, division
from collections import defaultdict
from sympy.core import (Basic, S, Add, Mul, Pow,
    Symbol, sympify, expand_mul, expand_func,
    Function, Dummy, Expr, factor_terms,
    symbols, expand_power_exp)
from sympy.core.compatibility import (iterable,
    ordered, range, as_int)
from sympy.core.numbers import Float, I, pi, Rational, Integer
from sympy.core.function import expand_log, count_ops, _mexpand, _coeff_isneg, nfloat
from sympy.core.rules import Transform
from sympy.core.evaluate import global_evaluate
from sympy.functions import (
    gamma, exp, sqrt, log, exp_polar, piecewise_fold)
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
from sympy.polys import (together, cancel, factor)
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



def simplify(expr, ratio=1.7, measure=count_ops, rational=False):
    expr = sympify(expr)

    try:
        return expr._eval_simplify(ratio=ratio, measure=measure)
    except AttributeError:
        pass

    original_expr = expr = signsimp(expr)

    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.functions.special.bessel import BesselBase
    from sympy import Sum, Product

    if not isinstance(expr, Basic) or not expr.args:  # XXX: temporary hack
        return expr

    if not isinstance(expr, (Add, Mul, Pow, ExpBase)):
        if isinstance(expr, Function) and hasattr(expr, "inverse"):
            if len(expr.args) == 1 and len(expr.args[0].args) == 1 and \
               isinstance(expr.args[0], expr.inverse(argindex=1)):
                return simplify(expr.args[0].args[0], ratio=ratio,
                                measure=measure, rational=rational)
        return expr.func(*[simplify(x, ratio=ratio, measure=measure, rational=rational)
                         for x in expr.args])

    # TODO: Apply different strategies, considering expression pattern:
    # is it a purely rational function? Is there any trigonometric function?...
    # See also https://github.com/sympy/sympy/pull/185.

    def shorter(*choices):
        '''Return the choice that has the fewest ops. In case of a tie,
        the expression listed first is selected.'''
        if not has_variety(choices):
            return choices[0]
        return min(choices, key=measure)

    # rationalize Floats
    floats = False
    if rational is not False and expr.has(Float):
        floats = True
        expr = nsimplify(expr, rational=True)

    expr = bottom_up(expr, lambda w: w.normal())
    expr = Mul(*powsimp(expr).as_content_primitive())
    _e = cancel(expr)
    expr1 = shorter(_e, _mexpand(_e).cancel())  # issue 6829
    expr2 = shorter(together(expr, deep=True), together(expr1, deep=True))

    if ratio is S.Infinity:
        expr = expr2
    else:
        expr = shorter(expr2, expr1, expr)
    if not isinstance(expr, Basic):  # XXX: temporary hack
        return expr

    expr = factor_terms(expr, sign=False)

    # hyperexpand automatically only works on hypergeometric terms
    expr = hyperexpand(expr)

    expr = piecewise_fold(expr)

    if expr.has(BesselBase):
        expr = besselsimp(expr)

    if expr.has(TrigonometricFunction, HyperbolicFunction):
        expr = trigsimp(expr, deep=True)

    if expr.has(log):
        expr = shorter(expand_log(expr, deep=True), logcombine(expr))

    if expr.has(CombinatorialFunction, gamma):
        expr = combsimp(expr)

    if expr.has(Sum):
        expr = sum_simplify(expr)

    if expr.has(Product):
        expr = product_simplify(expr)

    short = shorter(powsimp(expr, combine='exp', deep=True), powsimp(expr), expr)
    short = shorter(short, cancel(short))
    short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
    if short.has(TrigonometricFunction, HyperbolicFunction, ExpBase):
        short = exptrigsimp(short)

    # get rid of hollow 2-arg Mul factorization
    hollow_mul = Transform(
        lambda x: Mul(*x.args),
        lambda x:
        x.is_Mul and
        len(x.args) == 2 and
        x.args[0].is_Number and
        x.args[1].is_Add and
        x.is_commutative)
    expr = short.xreplace(hollow_mul)

    numer, denom = expr.as_numer_denom()
    if denom.is_Add:
        n, d = fraction(radsimp(1/denom, symbolic=False, max_terms=1))
        if n is not S.One:
            expr = (numer*n).expand()/d

    if expr.could_extract_minus_sign():
        n, d = fraction(expr)
        if d != 0:
            expr = signsimp(-n/(-d))

    if measure(expr) > ratio*measure(original_expr):
        expr = original_expr

    # restore floats
    if floats and rational is None:
        expr = nfloat(expr, exponent=False)

    return expr
