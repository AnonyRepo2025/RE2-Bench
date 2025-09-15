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
from sympy.simplify.simplify import factor_sum
from sympy import Dummy
from sympy.polys.polytools import real_roots
from sympy.polys.polyroots import roots
from sympy.polys.polyerrors import PolynomialError

_eps = Dummy(positive=True)

def _mask_nc(eq, name=None):
    """
    Return ``eq`` with non-commutative objects replaced with Dummy
    symbols. A dictionary that can be used to restore the original
    values is returned: if it is None, the expression is noncommutative
    and cannot be made commutative. The third value returned is a list
    of any non-commutative symbols that appear in the returned equation.

    ``name``, if given, is the name that will be used with numered Dummy
    variables that will replace the non-commutative objects and is mainly
    used for doctesting purposes.

    Notes
    =====
    All non-commutative objects other than Symbols are replaced with
    a non-commutative Symbol. Identical objects will be identified
    by identical symbols.

    If there is only 1 non-commutative object in an expression it will
    be replaced with a commutative symbol. Otherwise, the non-commutative
    entities are retained and the calling routine should handle
    replacements in this case since some care must be taken to keep
    track of the ordering of symbols when they occur within Muls.

    Examples
    ========

    >>> from sympy.physics.secondquant import Commutator, NO, F, Fd
    >>> from sympy import symbols, Mul
    >>> from sympy.core.exprtools import _mask_nc
    >>> from sympy.abc import x, y
    >>> A, B, C = symbols('A,B,C', commutative=False)

    One nc-symbol:

    >>> _mask_nc(A**2 - x**2, 'd')
    (_d0**2 - x**2, {_d0: A}, [])

    Multiple nc-symbols:

    >>> _mask_nc(A**2 - B**2, 'd')
    (A**2 - B**2, None, [A, B])

    An nc-object with nc-symbols but no others outside of it:

    >>> _mask_nc(1 + x*Commutator(A, B), 'd')
    (_d0*x + 1, {_d0: Commutator(A, B)}, [])
    >>> _mask_nc(NO(Fd(x)*F(y)), 'd')
    (_d0, {_d0: NO(CreateFermion(x)*AnnihilateFermion(y))}, [])

    Multiple nc-objects:

    >>> eq = x*Commutator(A, B) + x*Commutator(A, C)*Commutator(A, B)
    >>> _mask_nc(eq, 'd')
    (x*_d0 + x*_d1*_d0, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1])

    Multiple nc-objects and nc-symbols:

    >>> eq = A*Commutator(A, B) + B*Commutator(A, C)
    >>> _mask_nc(eq, 'd')
    (A*_d0 + B*_d1, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1, A, B])

    If there is an object that:

        - doesn't contain nc-symbols
        - but has arguments which derive from Basic, not Expr
        - and doesn't define an _eval_is_commutative routine

    then it will give False (or None?) for the is_commutative test. Such
    objects are also removed by this routine:

    >>> from sympy import Basic
    >>> eq = (1 + Mul(Basic(), Basic(), evaluate=False))
    >>> eq.is_commutative
    False
    >>> _mask_nc(eq, 'd')
    (_d0**2 + 1, {_d0: Basic()}, [])

    """
    name = name or 'mask'
    # Make Dummy() append sequential numbers to the name

    def numbered_names():
        i = 0
        while True:
            yield name + str(i)
            i += 1

    names = numbered_names()

    def Dummy(*args, **kwargs):
        from sympy import Dummy
        return Dummy(next(names), *args, **kwargs)

    expr = eq
    if expr.is_commutative:
        return eq, {}, []

    # identify nc-objects; symbols and other
    rep = []
    nc_obj = set()
    nc_syms = set()
    pot = preorder_traversal(expr, keys=default_sort_key)
    for i, a in enumerate(pot):
        if any(a == r[0] for r in rep):
            pot.skip()
        elif not a.is_commutative:
            if a.is_Symbol:
                nc_syms.add(a)
            elif not (a.is_Add or a.is_Mul or a.is_Pow):
                nc_obj.add(a)
                pot.skip()

    # If there is only one nc symbol or object, it can be factored regularly
    # but polys is going to complain, so replace it with a Dummy.
    if len(nc_obj) == 1 and not nc_syms:
        rep.append((nc_obj.pop(), Dummy()))
    elif len(nc_syms) == 1 and not nc_obj:
        rep.append((nc_syms.pop(), Dummy()))

    # Any remaining nc-objects will be replaced with an nc-Dummy and
    # identified as an nc-Symbol to watch out for
    nc_obj = sorted(nc_obj, key=default_sort_key)
    for n in nc_obj:
        nc = Dummy(commutative=False)
        rep.append((n, nc))
        nc_syms.add(nc)
    expr = expr.subs(rep)

    nc_syms = list(nc_syms)
    nc_syms.sort(key=default_sort_key)
    return expr, {v: k for k, v in rep} or None, nc_syms
