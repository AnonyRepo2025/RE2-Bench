def as_independent(self, *deps, **hint):
    from .symbol import Symbol
    from .add import _unevaluated_Add
    from .mul import _unevaluated_Mul
    from sympy.utilities.iterables import sift
    if self.is_zero:
        return (S.Zero, S.Zero)
    func = self.func
    if hint.get('as_Add', func is Add):
        want = Add
    else:
        want = Mul
    sym = set()
    other = []
    for d in deps:
        if isinstance(d, Symbol):
            sym.add(d)
        else:
            other.append(d)

    def has(e):
        has_other = e.has(*other)
        if not sym:
            return has_other
        return has_other or e.has(*e.free_symbols & sym)
    if want is not func or (func is not Add and func is not Mul):
        if has(self):
            return (want.identity, self)
        else:
            return (self, want.identity)
    elif func is Add:
        args = list(self.args)
    else:
        args, nc = self.args_cnc()
    d = sift(args, lambda x: has(x))
    depend = d[True]
    indep = d[False]
    if func is Add:
        return (Add(*indep), _unevaluated_Add(*depend))
    else:
        for i, n in enumerate(nc):
            if has(n):
                depend.extend(nc[i:])
                break
            indep.append(n)
        return (Mul(*indep), Mul(*depend, evaluate=False) if nc else _unevaluated_Mul(*depend))

def getit(self):
    try:
        return self._assumptions[fact]
    except KeyError:
        if self._assumptions is self.default_assumptions:
            self._assumptions = self.default_assumptions.copy()
        return _ask(fact, self)

def func(self):
    return self.__class__

def args(self):
    return self._args

def sift(seq, keyfunc):
    m = defaultdict(list)
    for i in seq:
        m[keyfunc(i)].append(i)
    return m

def has(e):
    has_other = e.has(*other)
    if not sym:
        return has_other
    return has_other or e.has(*e.free_symbols & sym)

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

def _sympify(a):
    return sympify(a, strict=True)

def __hash__(self):
    return hash(self.p)



from __future__ import print_function, division
from sympy.core.sympify import _sympify, sympify
from sympy.core.basic import Basic, _aresame
from sympy.core.cache import cacheit
from sympy.core.compatibility import ordered, range
from sympy.core.logic import fuzzy_and
from sympy.core.evaluate import global_evaluate
from sympy import Order
from .expr import Add, Expr
from sympy import Mul
from .function import WildFunction
from .symbol import Wild
from .add import Add
from .mul import Mul
from .symbol import Symbol
from .function import AppliedUndef
from sympy.simplify.radsimp import collect

class AssocOp(Basic):
    __slots__ = ['is_commutative']

    @cacheit
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

    @classmethod
    def _from_args(cls, args, is_commutative=None):
        if len(args) == 0:
            return cls.identity
        elif len(args) == 1:
            return args[0]
        obj = super(AssocOp, cls).__new__(cls, *args)
        if is_commutative is None:
            is_commutative = fuzzy_and((a.is_commutative for a in args))
        obj.is_commutative = is_commutative
        return obj

    def _eval_evalf(self, prec):
        from .add import Add
        from .mul import Mul
        from .symbol import Symbol
        from .function import AppliedUndef
        if isinstance(self, (Mul, Add)):
            x, tail = self.as_independent(Symbol, AppliedUndef)
            if not (tail is self.identity or (isinstance(x, AssocOp) and x.is_Function) or (x is self.identity and isinstance(tail, AssocOp))):
                x = x._evalf(prec) if x is not self.identity else self.identity
                args = []
                tail_args = tuple(self.func.make_args(tail))
                for a in tail_args:
                    newa = a._eval_evalf(prec)
                    if newa is None:
                        args.append(a)
                    else:
                        args.append(newa)
                return self.func(x, *args)
        args = []
        for a in self.args:
            newa = a._eval_evalf(prec)
            if newa is None:
                args.append(a)
            else:
                args.append(newa)
        return self.func(*args)

    @classmethod
    def make_args(cls, expr):
        if isinstance(expr, cls):
            return expr.args
        else:
            return (sympify(expr),)