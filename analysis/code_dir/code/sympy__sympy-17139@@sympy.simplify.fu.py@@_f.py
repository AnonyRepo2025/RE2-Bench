from __future__ import print_function, division
from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import S
from sympy.core.compatibility import ordered, range
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh, coth, sech, csch, HyperbolicFunction
from sympy.functions.elementary.trigonometric import cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.simplify.simplify import bottom_up
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
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
RL2 = [(TR4, TR3, TR10, TR4, TR3, TR11), (TR5, TR7, TR11, TR4), (CTR3, CTR1, TR9, CTR2, TR4, TR9, TR9, CTR4), identity]
fufuncs = '\n    TR0 TR1 TR2 TR3 TR4 TR5 TR6 TR7 TR8 TR9 TR10 TR10i TR11\n    TR12 TR13 L TR2i TRmorrie TR12i\n    TR14 TR15 TR16 TR111 TR22'.split()
FU = dict(list(zip(fufuncs, list(map(locals().get, fufuncs)))))
_ROOT2 = None

def _TR56(rv, f, g, h, max, pow):

    def _f(rv):
        if not (rv.is_Pow and rv.base.func == f):
            return rv
        if not rv.exp.is_real:
            return rv
        if (rv.exp < 0) == True:
            return rv
        if (rv.exp > max) == True:
            return rv
        if rv.exp == 2:
            return h(g(rv.base.args[0]) ** 2)
        else:
            if rv.exp == 4:
                e = 2
            elif not pow:
                if rv.exp % 2:
                    return rv
                e = rv.exp // 2
            else:
                p = perfect_power(rv.exp)
                if not p:
                    return rv
                e = rv.exp // 2
            return h(g(rv.base.args[0]) ** 2) ** e
    return bottom_up(rv, _f)