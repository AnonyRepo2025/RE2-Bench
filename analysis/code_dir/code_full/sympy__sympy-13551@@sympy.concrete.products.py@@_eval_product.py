def __cacheit_nocache(func):
    return func

def free_symbols(self):
    return set().union(*[a.free_symbols for a in self.args])

def args(self):
    return self._args

def free_symbols(self):
    return {self}

def __hash__(self):
    h = self._mhash
    if h is None:
        h = hash((type(self).__name__,) + self._hashable_content())
        self._mhash = h
    return h

def __eq__(self, other):
    from sympy import Pow
    if self is other:
        return True
    if type(self) is not type(other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if type(self) != type(other):
            return False
    return self._hashable_content() == other._hashable_content()

def _sympify(a):
    return sympify(a, strict=True)

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

def __sympifyit_wrapper(a, b):
    try:
        if not hasattr(b, '_op_priority'):
            b = sympify(b, strict=True)
        return func(a, b)
    except SympifyError:
        return retval



from __future__ import print_function, division
from sympy.tensor.indexed import Idx
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.concrete.expr_with_intlimits import ExprWithIntLimits
from sympy.functions.elementary.exponential import exp, log
from sympy.polys import quo, roots
from sympy.simplify import powsimp
from sympy.core.compatibility import range
from sympy.concrete.summations import Sum
from sympy.concrete.delta import deltaproduct, _has_simple_delta
from sympy.concrete.summations import summation
from sympy.functions import KroneckerDelta, RisingFactorial
from sympy.simplify.simplify import product_simplify
from sympy.concrete.summations import Sum
from sympy.concrete.summations import Sum

class Product(ExprWithIntLimits):
    __slots__ = ['is_commutative']
    function = term

    def _eval_product(self, term, limits):
        from sympy.concrete.delta import deltaproduct, _has_simple_delta
        from sympy.concrete.summations import summation
        from sympy.functions import KroneckerDelta, RisingFactorial
        k, a, n = limits
        if k not in term.free_symbols:
            if (term - 1).is_zero:
                return S.One
            return term ** (n - a + 1)
        if a == n:
            return term.subs(k, a)
        if term.has(KroneckerDelta) and _has_simple_delta(term, limits[0]):
            return deltaproduct(term, limits)
        dif = n - a
        if dif.is_Integer:
            return Mul(*[term.subs(k, a + i) for i in range(dif + 1)])
        elif term.is_polynomial(k):
            poly = term.as_poly(k)
            A = B = Q = S.One
            all_roots = roots(poly)
            M = 0
            for r, m in all_roots.items():
                M += m
                A *= RisingFactorial(a - r, n - a + 1) ** m
                Q *= (n - r) ** m
            if M < poly.degree():
                arg = quo(poly, Q.as_poly(k))
                B = self.func(arg, (k, a, n)).doit()
            return poly.LC() ** (n - a + 1) * A * B
        elif term.is_Add:
            p, q = term.as_numer_denom()
            q = self._eval_product(q, (k, a, n))
            if q.is_Number:
                from sympy.concrete.summations import Sum
                p = exp(Sum(log(p), (k, a, n)))
            else:
                p = self._eval_product(p, (k, a, n))
            return p / q
        elif term.is_Mul:
            exclude, include = ([], [])
            for t in term.args:
                p = self._eval_product(t, (k, a, n))
                if p is not None:
                    exclude.append(p)
                else:
                    include.append(t)
            if not exclude:
                return None
            else:
                arg = term._new_rawargs(*include)
                A = Mul(*exclude)
                B = self.func(arg, (k, a, n)).doit()
                return A * B
        elif term.is_Pow:
            if not term.base.has(k):
                s = summation(term.exp, (k, a, n))
                return term.base ** s
            elif not term.exp.has(k):
                p = self._eval_product(term.base, (k, a, n))
                if p is not None:
                    return p ** term.exp
        elif isinstance(term, Product):
            evaluated = term.doit()
            f = self._eval_product(evaluated, limits)
            if f is None:
                return self.func(evaluated, limits)
            else:
                return f