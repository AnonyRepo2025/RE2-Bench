from __future__ import print_function, division
from collections import defaultdict
from sympy.core import (Basic, S, Add, Mul, Pow, Symbol, sympify, expand_mul,
                        expand_func, Function, Dummy, Expr, factor_terms,
                        expand_power_exp)
from sympy.core.compatibility import iterable, ordered, range, as_int
from sympy.core.evaluate import global_evaluate
from sympy.core.function import expand_log, count_ops, _mexpand, _coeff_isneg, nfloat
from sympy.core.numbers import Float, I, pi, Rational, Integer
from sympy.core.rules import Transform
from sympy.core.sympify import _sympify
from sympy.functions import gamma, exp, sqrt, log, exp_polar, piecewise_fold
from sympy.functions.combinatorial.factorials import CombinatorialFunction
from sympy.functions.elementary.complexes import unpolarify
from sympy.functions.elementary.exponential import ExpBase
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.special.bessel import besselj, besseli, besselk, jn, bessely
from sympy.polys import together, cancel, factor
from sympy.simplify.combsimp import combsimp
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import radsimp, fraction
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp, exptrigsimp
from sympy.utilities.iterables import has_variety
import mpmath
from sympy.simplify.hyperexpand import hyperexpand
from sympy.functions.special.bessel import BesselBase
from sympy import Sum, Product
from sympy.physics.units import Quantity
from sympy.physics.units.util import quantity_simplify
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
from sympy.matrices.expressions import (MatrixExpr, MatAdd, MatMul,
                                                MatPow, MatrixSymbol)
from sympy.core.exprtools import factor_nc



def posify(eq):
    """Return eq (with generic symbols made positive) and a
    dictionary containing the mapping between the old and new
    symbols.

    Any symbol that has positive=None will be replaced with a positive dummy
    symbol having the same name. This replacement will allow more symbolic
    processing of expressions, especially those involving powers and
    logarithms.

    A dictionary that can be sent to subs to restore eq to its original
    symbols is also returned.

    >>> from sympy import posify, Symbol, log, solve
    >>> from sympy.abc import x
    >>> posify(x + Symbol('p', positive=True) + Symbol('n', negative=True))
    (_x + n + p, {_x: x})

    >>> eq = 1/x
    >>> log(eq).expand()
    log(1/x)
    >>> log(posify(eq)[0]).expand()
    -log(_x)
    >>> p, rep = posify(eq)
    >>> log(p).expand().subs(rep)
    -log(x)

    It is possible to apply the same transformations to an iterable
    of expressions:

    >>> eq = x**2 - 4
    >>> solve(eq, x)
    [-2, 2]
    >>> eq_x, reps = posify([eq, x]); eq_x
    [_x**2 - 4, _x]
    >>> solve(*eq_x)
    [2]
    """
    eq = sympify(eq)
    if iterable(eq):
        f = type(eq)
        eq = list(eq)
        syms = set()
        for e in eq:
            syms = syms.union(e.atoms(Symbol))
        reps = {}
        for s in syms:
            reps.update(dict((v, k) for k, v in posify(s)[1].items()))
        for i, e in enumerate(eq):
            eq[i] = e.subs(reps)
        return f(eq), {r: s for s, r in reps.items()}

    reps = {s: Dummy(s.name, positive=True, **s.assumptions0)
                 for s in eq.free_symbols if s.is_positive is None}
    eq = eq.subs(reps)
    return eq, {r: s for s, r in reps.items()}
