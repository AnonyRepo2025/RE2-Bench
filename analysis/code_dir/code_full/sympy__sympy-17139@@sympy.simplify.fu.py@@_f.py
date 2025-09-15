def base(self):
    return self._args[0]

def func(self):
    return self.__class__

def exp(self):
    return self._args[1]

def __lt__(self, other):
    try:
        other = _sympify(other)
    except SympifyError:
        raise TypeError('Invalid comparison %s < %s' % (self, other))
    if other.is_Integer:
        return _sympify(self.p < other.p)
    return Rational.__lt__(self, other)

def _sympify(a):
    return sympify(a, strict=True)

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    try:
        if a in sympy_classes:
            return a
    except TypeError:
        pass
    cls = getattr(a, '__class__', None)
    if cls is None:
        cls = type(a)
    if cls in sympy_classes:
        return a
    if isinstance(a, CantSympify):
        raise SympifyError(a)
    try:
        return converter[cls](a)
    except KeyError:
        for superclass in getmro(cls):
            try:
                return converter[superclass](a)
            except KeyError:
                continue
    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a
    if evaluate is None:
        if global_evaluate[0] is False:
            evaluate = global_evaluate[0]
        else:
            evaluate = True
    if type(a).__module__ == 'numpy':
        import numpy as np
        if np.isscalar(a):
            return _convert_numpy_types(a, locals=locals, convert_xor=convert_xor, strict=strict, rational=rational, evaluate=evaluate)
    _sympy_ = getattr(a, '_sympy_', None)
    if _sympy_ is not None:
        try:
            return a._sympy_()
        except AttributeError:
            pass
    if not strict:
        flat = getattr(a, 'flat', None)
        if flat is not None:
            shape = getattr(a, 'shape', None)
            if shape is not None:
                from ..tensor.array import Array
                return Array(a.flat, a.shape)
    if not isinstance(a, string_types):
        for coerce in (float, int):
            try:
                coerced = coerce(a)
            except (TypeError, ValueError):
                continue
            except AttributeError:
                continue
            try:
                return sympify(coerced)
            except SympifyError:
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

def __new__(cls, i):
    if isinstance(i, string_types):
        i = i.replace(' ', '')
    try:
        ival = int(i)
    except TypeError:
        raise TypeError('Argument of Integer should be of numeric type, got %s.' % i)
    if ival == 1:
        return S.One
    if ival == -1:
        return S.NegativeOne
    if ival == 0:
        return S.Zero
    obj = Expr.__new__(cls)
    obj.p = ival
    return obj

def __eq__(self, other):
    if self is other:
        return True
    tself = type(self)
    tother = type(other)
    if tself is not tother:
        try:
            other = _sympify(other)
            tother = type(other)
        except SympifyError:
            return NotImplemented
        if PY3 or type(tself).__ne__ is not type.__ne__:
            if tself != tother:
                return False
        elif tself is not tother:
            return False
    return self._hashable_content() == other._hashable_content()

def __gt__(self, other):
    try:
        other = _sympify(other)
    except SympifyError:
        raise TypeError('Invalid comparison %s > %s' % (self, other))
    if other.is_Integer:
        return _sympify(self.p > other.p)
    return Rational.__gt__(self, other)

def __new__(cls, *args):
    obj = object.__new__(cls)
    obj._assumptions = cls.default_assumptions
    obj._mhash = None
    obj._args = args
    return obj

def __eq__(self, other):
    if isinstance(other, integer_types):
        return self.p == other
    elif isinstance(other, Integer):
        return self.p == other.p
    return Rational.__eq__(self, other)

def __mod__(self, other):
    if global_evaluate[0]:
        if isinstance(other, integer_types):
            return Integer(self.p % other)
        elif isinstance(other, Integer):
            return Integer(self.p % other.p)
        return Rational.__mod__(self, other)
    return Rational.__mod__(self, other)

def _hashable_content(self):
    return self._args

def __nonzero__(self):
    return False

def __floordiv__(self, other):
    if isinstance(other, Integer):
        return Integer(self.p // other)
    return Integer(divmod(self, other)[0])



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