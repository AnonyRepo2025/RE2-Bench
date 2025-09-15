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
from sympy.utilities.iterables import common_prefix, common_suffix, variations, ordered
from collections import defaultdict
from sympy.simplify.simplify import powsimp
from sympy.polys import gcd, factor
from sympy.concrete.summations import Sum
from sympy.simplify.simplify import factor_sum
from sympy import Dummy
from sympy.polys.polytools import real_roots
from sympy.polys.polyroots import roots
from sympy.polys.polyerrors import PolynomialError
_eps = Dummy(positive=True)

def _mask_nc(eq, name=None):
    name = name or 'mask'

    def numbered_names():
        i = 0
        while True:
            yield (name + str(i))
            i += 1
    names = numbered_names()

    def Dummy(*args, **kwargs):
        from sympy import Dummy
        return Dummy(next(names), *args, **kwargs)
    expr = eq
    if expr.is_commutative:
        return (eq, {}, [])
    rep = []
    nc_obj = set()
    nc_syms = set()
    pot = preorder_traversal(expr, keys=default_sort_key)
    for i, a in enumerate(pot):
        if any((a == r[0] for r in rep)):
            pot.skip()
        elif not a.is_commutative:
            if a.is_Symbol:
                nc_syms.add(a)
            elif not (a.is_Add or a.is_Mul or a.is_Pow):
                nc_obj.add(a)
                pot.skip()
    if len(nc_obj) == 1 and (not nc_syms):
        rep.append((nc_obj.pop(), Dummy()))
    elif len(nc_syms) == 1 and (not nc_obj):
        rep.append((nc_syms.pop(), Dummy()))
    nc_obj = sorted(nc_obj, key=default_sort_key)
    for n in nc_obj:
        nc = Dummy(commutative=False)
        rep.append((n, nc))
        nc_syms.add(nc)
    expr = expr.subs(rep)
    nc_syms = list(nc_syms)
    nc_syms.sort(key=default_sort_key)
    return (expr, {v: k for k, v in rep} or None, nc_syms)