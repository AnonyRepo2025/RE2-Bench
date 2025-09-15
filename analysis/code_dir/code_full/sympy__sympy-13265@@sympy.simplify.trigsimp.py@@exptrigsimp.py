def bottom_up(rv, F, atoms=False, nonbasic=False):
    try:
        if rv.args:
            args = tuple([bottom_up(a, F, atoms, nonbasic) for a in rv.args])
            if args != rv.args:
                rv = rv.func(*args)
            rv = F(rv)
        elif atoms:
            rv = F(rv)
    except AttributeError:
        if nonbasic:
            try:
                rv = F(rv)
            except TypeError:
                pass
    return rv

def args(self):
    return self._args

def has(self, *patterns):
    return any((self._has(pattern) for pattern in patterns))

def _has(self, pattern):
    from sympy.core.function import UndefinedFunction, Function
    if isinstance(pattern, UndefinedFunction):
        return any((f.func == pattern or f == pattern for f in self.atoms(Function, UndefinedFunction)))
    pattern = sympify(pattern)
    if isinstance(pattern, BasicMeta):
        return any((isinstance(arg, pattern) for arg in preorder_traversal(self)))
    try:
        match = pattern._has_matcher()
        return any((match(arg) for arg in preorder_traversal(self)))
    except AttributeError:
        return any((arg == pattern for arg in preorder_traversal(self)))

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    if evaluate is None:
        if global_evaluate[0] is False:
            evaluate = global_evaluate[0]
        else:
            evaluate = True
    try:
        if a in sympy_classes:
            return a
    except TypeError:
        pass
    try:
        cls = a.__class__
    except AttributeError:
        cls = type(a)
    if cls in sympy_classes:
        return a
    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a
    if type(a).__module__ == 'numpy':
        import numpy as np
        if np.isscalar(a):
            if not isinstance(a, np.floating):
                func = converter[complex] if np.iscomplex(a) else sympify
                return func(np.asscalar(a))
            else:
                try:
                    from sympy.core.numbers import Float
                    prec = np.finfo(a).nmant
                    a = str(list(np.reshape(np.asarray(a), (1, np.size(a)))[0]))[1:-1]
                    return Float(a, precision=prec)
                except NotImplementedError:
                    raise SympifyError('Translation for numpy float : %s is not implemented' % a)
    try:
        return converter[cls](a)
    except KeyError:
        for superclass in getmro(cls):
            try:
                return converter[superclass](a)
            except KeyError:
                continue
    if isinstance(a, CantSympify):
        raise SympifyError(a)
    try:
        return a._sympy_()
    except AttributeError:
        pass
    if not isinstance(a, string_types):
        for coerce in (float, int):
            try:
                return sympify(coerce(a))
            except (TypeError, ValueError, AttributeError, SympifyError):
                continue
    if strict:
        raise SympifyError(a)
    try:
        from ..tensor.array import Array
        return Array(a.flat, a.shape)
    except AttributeError:
        pass
    if iterable(a):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor, rational=rational) for x in a])
        except TypeError:
            pass
    if isinstance(a, dict):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor, rational=rational) for x in a.items()])
        except TypeError:
            pass
    try:
        from .compatibility import unicode
        a = unicode(a)
    except Exception as exc:
        raise SympifyError(a, exc)
    from sympy.parsing.sympy_parser import parse_expr, TokenError, standard_transformations
    from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
    from sympy.parsing.sympy_parser import rationalize as t_rationalize
    transformations = standard_transformations
    if rational:
        transformations += (t_rationalize,)
    if convert_xor:
        transformations += (t_convert_xor,)
    try:
        a = a.replace('\n', '')
        expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
    except (TokenError, SyntaxError) as exc:
        raise SympifyError('could not parse %r' % a, exc)
    return expr

def __init__(self, node, keys=None):
    self._skip_flag = False
    self._pt = self._preorder_traversal(node, keys)

def __iter__(self):
    return self

def __next__(self):
    return next(self._pt)

def _preorder_traversal(self, node, keys):
    yield node
    if self._skip_flag:
        self._skip_flag = False
        return
    if isinstance(node, Basic):
        if not keys and hasattr(node, '_argset'):
            args = node._argset
        else:
            args = node.args
        if keys:
            if keys != True:
                args = ordered(args, keys, default=False)
            else:
                args = ordered(args)
        for arg in args:
            for subtree in self._preorder_traversal(arg, keys):
                yield subtree
    elif iterable(node):
        for item in node:
            for subtree in self._preorder_traversal(item, keys):
                yield subtree

def rewrite(self, *args, **hints):
    if not args:
        return self
    else:
        pattern = args[:-1]
        if isinstance(args[-1], string_types):
            rule = '_eval_rewrite_as_' + args[-1]
        else:
            try:
                rule = '_eval_rewrite_as_' + args[-1].__name__
            except:
                rule = '_eval_rewrite_as_' + args[-1].__class__.__name__
        if not pattern:
            return self._eval_rewrite(None, rule, **hints)
        else:
            if iterable(pattern[0]):
                pattern = pattern[0]
            pattern = [p for p in pattern if self.has(p)]
            if pattern:
                return self._eval_rewrite(tuple(pattern), rule, **hints)
            else:
                return self

def _eval_rewrite(self, pattern, rule, **hints):
    if self.is_Atom:
        if hasattr(self, rule):
            return getattr(self, rule)()
        return self
    if hints.get('deep', True):
        args = [a._eval_rewrite(pattern, rule, **hints) if isinstance(a, Basic) else a for a in self.args]
    else:
        args = self.args
    if pattern is None or isinstance(self, pattern):
        if hasattr(self, rule):
            rewritten = getattr(self, rule)(*args)
            if rewritten is not None:
                return rewritten
    return self.func(*args)

def func(self):
    return self.__class__

def __new__(cls, *args, **options):
    from sympy import Order
    args = list(map(_sympify, args))
    args = [a for a in args if a is not cls.identity]
    if not options.pop('evaluate', global_evaluate[0]):
        return cls._from_args(args)
    if len(args) == 0:
        return cls.identity
    if len(args) == 1:
        return args[0]
    c_part, nc_part, order_symbols = cls.flatten(args)
    is_commutative = not nc_part
    obj = cls._from_args(c_part + nc_part, is_commutative)
    obj = cls._exec_constructor_postprocessors(obj)
    if order_symbols is not None:
        return Order(obj, *order_symbols)
    return obj

def _sympify(a):
    return sympify(a, strict=True)

def __hash__(self):
    return super(Float, self).__hash__()



from __future__ import print_function, division
from collections import defaultdict
from sympy.core.cache import cacheit
from sympy.core import sympify, Basic, S, Expr, expand_mul, factor_terms, Mul, Dummy, igcd, FunctionClass, Add, symbols, Wild, expand
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
from sympy.simplify.fu import TR1, TR2, TR3, TR2i, TR10, L, TR10i, TR8, TR6, TR15, TR16, TR111, TR5, TRmorrie, TR11, TR14, TR22, TR12
from sympy.core.compatibility import _nodes
_trigs = (TrigonometricFunction, HyperbolicFunction)
_trigpat = None
_idn = lambda x: x
_midn = lambda x: -x
_one = lambda x: S.One

def exptrigsimp(expr):
    from sympy.simplify.fu import hyper_as_trig, TR2i
    from sympy.simplify.simplify import bottom_up

    def exp_trig(e):
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
                return (sign, 1)
            elif isinstance(expr, exp):
                return (sign, expr.args[0])
            elif sign == 1:
                return signlog(-expr, sign=-1)
            else:
                return (None, None)
        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                c = k.args[0]
                sign, x = signlog(k.args[1] / c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x * m / 2:
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2 * c * cosh(x / 2)] += m
                    else:
                        newd[-2 * c * sinh(x / 2)] += m
                elif newd[1 - sign * S.Exp1 ** x] == -m:
                    del newd[1 - sign * S.Exp1 ** x]
                    if sign == 1:
                        newd[-c / tanh(x / 2)] += m
                    else:
                        newd[-c * tanh(x / 2)] += m
                else:
                    newd[1 + sign * S.Exp1 ** x] += m
                    newd[c] += m
        return Mul(*[k ** newd[k] for k in newd])
    newexpr = bottom_up(newexpr, f)
    if newexpr.has(HyperbolicFunction):
        e, f = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)
    if not (newexpr.has(I) and (not expr.has(I))):
        expr = newexpr
    return expr