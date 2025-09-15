def shape(self):
    return self.args[1:3]

def args(self):
    return self._args

def __eq__(self, other):
    if isinstance(other, integer_types):
        return self.p == other
    elif isinstance(other, Integer):
        return self.p == other.p
    return Rational.__eq__(self, other)

def shape(self):
    return (self.args[0], self.args[1])

def __sympifyit_wrapper(a, b):
    try:
        b = sympify(b, strict=True)
        return func(a, b)
    except SympifyError:
        return retval

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
            return _convert_numpy_types(a, locals=locals, convert_xor=convert_xor, strict=strict, rational=rational, evaluate=evaluate)
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

def __hash__(self):
    h = self._mhash
    if h is None:
        h = hash((type(self).__name__,) + self._hashable_content())
        self._mhash = h
    return h

def binary_op_wrapper(self, other):
    if hasattr(other, '_op_priority'):
        if other._op_priority > self._op_priority:
            try:
                f = getattr(other, method_name)
            except AttributeError:
                pass
            else:
                return f(self)
    return func(self, other)

def __new__(cls, *args, **kwargs):
    check = kwargs.get('check', True)
    if not args:
        return GenericIdentity()
    args = filter(lambda i: GenericIdentity() != i, args)
    args = list(map(sympify, args))
    obj = Basic.__new__(cls, *args)
    factor, matrices = obj.as_coeff_matrices()
    if check:
        validate(*matrices)
    if not matrices:
        return factor
    return obj

def __new__(cls):
    return super(Identity, cls).__new__(cls)

def __new__(cls, *args):
    obj = object.__new__(cls)
    obj._assumptions = cls.default_assumptions
    obj._mhash = None
    obj._args = args
    return obj

def __ne__(self, other):
    return not self == other

def __eq__(self, other):
    return isinstance(other, GenericIdentity)

def __hash__(self):
    return hash(self.p)

def as_coeff_matrices(self):
    scalars = [x for x in self.args if not x.is_Matrix]
    matrices = [x for x in self.args if x.is_Matrix]
    coeff = Mul(*scalars)
    return (coeff, matrices)



from __future__ import print_function, division
from functools import wraps, reduce
import collections
from sympy.core import S, Symbol, Tuple, Integer, Basic, Expr, Eq
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import range, SYMPY_INTS, default_sort_key
from sympy.core.sympify import SympifyError, sympify
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices import ShapeError
from sympy.simplify import simplify
from sympy.utilities.misc import filldedent
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from sympy import Derivative
from sympy.matrices.expressions.adjoint import Adjoint
from sympy.matrices.expressions.transpose import Transpose
from sympy import I
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.adjoint import Adjoint
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.immutable import ImmutableDenseMatrix
from numpy import empty
from sympy import Sum, Mul, Add, MatMul, transpose, trace
from sympy.strategies.traverse import bottom_up
from .applyfunc import ElementwiseApplyFunction
from sympy import MatrixBase
from sympy import Sum, symbols, Dummy
from sympy.matrices.expressions.slice import MatrixSlice
from sympy import MatrixBase
from sympy.matrices.expressions.slice import MatrixSlice

class MatrixExpr(Expr):
    _iterable = False
    _op_priority = 11.0
    is_Matrix = True
    is_MatrixExpr = True
    is_Identity = None
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False
    is_commutative = False
    is_number = False
    is_symbol = False
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    T = property(transpose, None, None, 'Matrix transposition.')
    inv = inverse

    def __new__(cls, *args, **kwargs):
        args = map(sympify, args)
        return Basic.__new__(cls, *args, **kwargs)

    def __neg__(self):
        return MatMul(S.NegativeOne, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return MatAdd(self, -other, check=True).doit()

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    def _eval_Eq(self, other):
        if not isinstance(other, MatrixExpr):
            return False
        if self.shape != other.shape:
            return False
        if (self - other).is_ZeroMatrix:
            return True
        return Eq(self, other, evaluate=False)