def free_symbols(self):
    return set().union(*[a.free_symbols for a in self.args])

def args(self):
    return tuple(ordered(self._argset))

def ordered(seq, keys=None, default=True, warn=False):
    d = defaultdict(list)
    if keys:
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        keys = list(keys)
        f = keys.pop(0)
        for a in seq:
            d[f(a)].append(a)
    else:
        if not default:
            raise ValueError('if default=False then keys must be provided')
        d[None].extend(seq)
    for k in sorted(d.keys()):
        if len(d[k]) > 1:
            if keys:
                d[k] = ordered(d[k], keys, default, warn)
            elif default:
                d[k] = ordered(d[k], (_nodes, default_sort_key), default=False, warn=warn)
            elif warn:
                from sympy.utilities.iterables import uniq
                u = list(uniq(d[k]))
                if len(u) > 1:
                    raise ValueError('not enough keys to break ties: %s' % u)
        for v in d[k]:
            yield v
        d.pop(k)

def _nodes(e):
    from .basic import Basic
    if isinstance(e, Basic):
        return e.count(Basic)
    elif iterable(e):
        return 1 + sum((_nodes(ei) for ei in e))
    elif isinstance(e, dict):
        return 1 + sum((_nodes(k) + _nodes(v) for k, v in e.items()))
    else:
        return 1

def count(self, query):
    query = _make_find_query(query)
    return sum((bool(query(sub)) for sub in preorder_traversal(self)))

def _make_find_query(query):
    try:
        query = sympify(query)
    except SympifyError:
        pass
    if isinstance(query, type):
        return lambda expr: isinstance(expr, query)
    elif isinstance(query, Basic):
        return lambda expr: expr.match(query) is not None
    return query

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
            return _convert_numpy_types(a)
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
    if not strict:
        try:
            from ..tensor.array import Array
            return Array(a.flat, a.shape)
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

def args(self):
    return self._args

def default_sort_key(item, order=None):
    from .singleton import S
    from .basic import Basic
    from .sympify import sympify, SympifyError
    from .compatibility import iterable
    if isinstance(item, Basic):
        return item.sort_key(order=order)
    if iterable(item, exclude=string_types):
        if isinstance(item, dict):
            args = item.items()
            unordered = True
        elif isinstance(item, set):
            args = item
            unordered = True
        else:
            args = list(item)
            unordered = False
        args = [default_sort_key(arg, order=order) for arg in args]
        if unordered:
            args = sorted(args)
        cls_index, args = (10, (len(args), tuple(args)))
    else:
        if not isinstance(item, string_types):
            try:
                item = sympify(item)
            except SympifyError:
                pass
            else:
                if isinstance(item, Basic):
                    return default_sort_key(item)
        cls_index, args = (0, (1, (str(item),)))
    return ((cls_index, 0, item.__class__.__name__), args, S.One.sort_key(), S.One)

def sort_key(self, order=None):

    def inner_key(arg):
        if isinstance(arg, Basic):
            return arg.sort_key(order)
        else:
            return arg
    args = self._sorted_args
    args = (len(args), tuple([inner_key(arg) for arg in args]))
    return (self.class_key(), args, S.One.sort_key(), S.One)

def _sorted_args(self):
    return self.args



from __future__ import print_function, division
from collections import defaultdict
from itertools import combinations, product
from sympy.core.basic import Basic, as_Basic
from sympy.core.cache import cacheit
from sympy.core.numbers import Number, oo
from sympy.core.operations import LatticeOp
from sympy.core.function import Application, Derivative
from sympy.core.compatibility import ordered, range, with_metaclass, as_int, reduce
from sympy.core.sympify import converter, _sympify, sympify
from sympy.core.singleton import Singleton, S
from sympy.utilities.misc import filldedent
from sympy.core.symbol import Symbol
from sympy.logic.inference import satisfiable
from sympy.core.relational import Relational
from sympy.calculus.util import periodicity
from sympy.core.relational import Relational
from sympy.core.relational import Eq, Ne
from sympy.utilities.misc import filldedent
from sympy.utilities.misc import filldedent
from sympy.core.relational import Relational, Eq, Ne
from sympy.core.relational import Eq, Relational
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.sets import Intersection
from sympy.sets.sets import Union
from sympy import Equality, GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality
from sympy.core.relational import Relational
from sympy.core.relational import Eq, Ne
from sympy.core.relational import Eq, Ne
from sympy.functions import Piecewise
from sympy.simplify.simplify import simplify
true = BooleanTrue()
false = BooleanFalse()
S.true = true
S.false = false
converter[bool] = lambda x: S.true if x else S.false

def _finger(eq):
    f = eq.free_symbols
    d = dict(list(zip(f, [[0] * 5 for fi in f])))
    for a in eq.args:
        if a.is_Symbol:
            d[a][0] += 1
        elif a.is_Not:
            d[a.args[0]][1] += 1
        else:
            o = len(a.args) + sum((isinstance(ai, Not) for ai in a.args))
            for ai in a.args:
                if ai.is_Symbol:
                    d[ai][2] += 1
                    d[ai][-1] += o
                elif ai.is_Not:
                    d[ai.args[0]][3] += 1
                else:
                    raise NotImplementedError('unexpected level of nesting')
    inv = defaultdict(list)
    for k, v in ordered(iter(d.items())):
        inv[tuple(v)].append(k)
    return inv