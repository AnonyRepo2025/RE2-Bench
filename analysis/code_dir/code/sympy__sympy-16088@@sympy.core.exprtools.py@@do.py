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
from sympy.integrals.integrals import Integral
from sympy import Dummy
from sympy.polys.polytools import real_roots
from sympy.polys.polyroots import roots
from sympy.polys.polyerrors import PolynomialError
_eps = Dummy(positive=True)

def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):

    def do(expr):
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral
        is_iterable = iterable(expr)
        if not isinstance(expr, Basic) or expr.is_Atom:
            if is_iterable:
                return type(expr)([do(i) for i in expr])
            return expr
        if expr.is_Pow or expr.is_Function or is_iterable or (not hasattr(expr, 'args_cnc')):
            args = expr.args
            newargs = tuple([do(i) for i in args])
            if newargs == args:
                return expr
            return expr.func(*newargs)
        if isinstance(expr, (Sum, Integral)):
            return _factor_sum_int(expr, radical=radical, clear=clear, fraction=fraction, sign=sign)
        cont, p = expr.as_content_primitive(radical=radical, clear=clear)
        if p.is_Add:
            list_args = [do(a) for a in Add.make_args(p)]
            if all((a.as_coeff_Mul()[0].extract_multiplicatively(-1) is not None for a in list_args)):
                cont = -cont
                list_args = [-a for a in list_args]
            special = {}
            for i, a in enumerate(list_args):
                b, e = a.as_base_exp()
                if e.is_Mul and e != Mul(*e.args):
                    list_args[i] = Dummy()
                    special[list_args[i]] = a
            p = Add._from_args(list_args)
            p = gcd_terms(p, isprimitive=True, clear=clear, fraction=fraction).xreplace(special)
        elif p.args:
            p = p.func(*[do(a) for a in p.args])
        rv = _keep_coeff(cont, p, clear=clear, sign=sign)
        return rv
    expr = sympify(expr)
    return do(expr)