from __future__ import print_function, division
from collections import defaultdict
from sympy.simplify.simplify import bottom_up
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)
from sympy.functions.elementary.hyperbolic import (
    cosh, sinh, tanh, coth, sech, csch, HyperbolicFunction)
from sympy.core.compatibility import ordered, range
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import expand_mul
from sympy.core.add import Add
from sympy.core.symbol import Dummy
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.basic import S
from sympy.core.numbers import pi, I
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy.polys.polytools import factor
from sympy.ntheory.factor_ import perfect_power
from sympy import SYMPY_DEBUG
from sympy.simplify.simplify import signsimp
from sympy import factor
from sympy.simplify.simplify import signsimp
from sympy.simplify.radsimp import collect

CTR1 = [(TR5, TR0), (TR6, TR0), identity]
CTR2 = (TR11, [(TR5, TR0), (TR6, TR0), TR0])
CTR3 = [(TRmorrie, TR8, TR0), (TRmorrie, TR8, TR10i, TR0), identity]
CTR4 = [(TR4, TR10i), identity]
RL1 = (TR4, TR3, TR4, TR12, TR4, TR13, TR4, TR0)
RL2 = [
    (TR4, TR3, TR10, TR4, TR3, TR11),
    (TR5, TR7, TR11, TR4),
    (CTR3, CTR1, TR9, CTR2, TR4, TR9, TR9, CTR4),
    identity,
    ]
fufuncs = '''
    TR0 TR1 TR2 TR3 TR4 TR5 TR6 TR7 TR8 TR9 TR10 TR10i TR11
    TR12 TR13 L TR2i TRmorrie TR12i
    TR14 TR15 TR16 TR111 TR22'''.split()
FU = dict(list(zip(fufuncs, list(map(locals().get, fufuncs)))))
_ROOT2 = None

def _osborne(e, d):
    """Replace all hyperbolic functions with trig functions using
    the Osborne rule.

    Notes
    =====

    ``d`` is a dummy variable to prevent automatic evaluation
    of trigonometric/hyperbolic functions.


    References
    ==========

    http://en.wikipedia.org/wiki/Hyperbolic_function
    """

    def f(rv):
        if not isinstance(rv, HyperbolicFunction):
            return rv
        a = rv.args[0]
        a = a*d if not a.is_Add else Add._from_args([i*d for i in a.args])
        if rv.func is sinh:
            return I*sin(a)
        elif rv.func is cosh:
            return cos(a)
        elif rv.func is tanh:
            return I*tan(a)
        elif rv.func is coth:
            return cot(a)/I
        elif rv.func is sech:
            return sec(a)
        elif rv.func is csch:
            return csc(a)/I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)

    return bottom_up(e, f)
