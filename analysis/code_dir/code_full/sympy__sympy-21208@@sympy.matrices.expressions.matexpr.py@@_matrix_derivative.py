def _eval_derivative_matrix_lines(self, x):
    from sympy.matrices.expressions.matexpr import _LeftRightArgs
    return [_LeftRightArgs([S.One, S.One], higher=self._eval_derivative(x))]

def _eval_derivative(self, arg):
    return self.applyfunc(lambda x: x.diff(arg))

def applyfunc(self, f):
    if not callable(f):
        raise TypeError('`f` must be callable.')
    return self._eval_applyfunc(f)

def _eval_applyfunc(self, f):
    out = self._new(self.rows, self.cols, [f(x) for x in self])
    return out

def rows(self):
    return self._rows

def cols(self):
    return self._cols

def __getitem__(self, key):
    if isinstance(key, tuple):
        i, j = key
        try:
            i, j = self.key2ij(key)
            return self._mat[i * self.cols + j]
        except (TypeError, IndexError):
            if isinstance(i, Expr) and (not i.is_number) or (isinstance(j, Expr) and (not j.is_number)):
                if (j < 0) is True or (j >= self.shape[1]) is True or (i < 0) is True or ((i >= self.shape[0]) is True):
                    raise ValueError('index out of boundary')
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)
            if isinstance(i, slice):
                i = range(self.rows)[i]
            elif is_sequence(i):
                pass
            else:
                i = [i]
            if isinstance(j, slice):
                j = range(self.cols)[j]
            elif is_sequence(j):
                pass
            else:
                j = [j]
            return self.extract(i, j)
    else:
        if isinstance(key, slice):
            return self._mat[key]
        return self._mat[a2idx(key)]

def a2idx(j, n=None):
    if type(j) is not int:
        jindex = getattr(j, '__index__', None)
        if jindex is not None:
            j = jindex()
        else:
            raise IndexError('Invalid index a[%r]' % (j,))
    if n is not None:
        if j < 0:
            j += n
        if not (j >= 0 and j < n):
            raise IndexError('Index out of range: a[%s]' % (j,))
    return int(j)

def diff(self, *symbols, **assumptions):
    assumptions.setdefault('evaluate', True)
    return _derivative_dispatch(self, *symbols, **assumptions)

def _derivative_dispatch(expr, *variables, **kwargs):
    from sympy.matrices.common import MatrixCommon
    from sympy import MatrixExpr
    from sympy import NDimArray
    array_types = (MatrixCommon, MatrixExpr, NDimArray, list, tuple, Tuple)
    if isinstance(expr, array_types) or any((isinstance(i[0], array_types) if isinstance(i, (tuple, list, Tuple)) else isinstance(i, array_types) for i in variables)):
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        return ArrayDerivative(expr, *variables, **kwargs)
    return Derivative(expr, *variables, **kwargs)

def __new__(cls, expr, *variables, **kwargs):
    from sympy.matrices.common import MatrixCommon
    from sympy import Integer, MatrixExpr
    from sympy.tensor.array import Array, NDimArray
    from sympy.utilities.misc import filldedent
    expr = sympify(expr)
    symbols_or_none = getattr(expr, 'free_symbols', None)
    has_symbol_set = isinstance(symbols_or_none, set)
    if not has_symbol_set:
        raise ValueError(filldedent('\n            Since there are no variables in the expression %s,\n            it cannot be differentiated.' % expr))
    if not variables:
        variables = expr.free_symbols
        if len(variables) != 1:
            if expr.is_number:
                return S.Zero
            if len(variables) == 0:
                raise ValueError(filldedent('\n                    Since there are no variables in the expression,\n                    the variable(s) of differentiation must be supplied\n                    to differentiate %s' % expr))
            else:
                raise ValueError(filldedent('\n                    Since there is more than one variable in the\n                    expression, the variable(s) of differentiation\n                    must be supplied to differentiate %s' % expr))
    variables = list(sympify(variables))
    variable_count = []
    array_likes = (tuple, list, Tuple)
    for i, v in enumerate(variables):
        if isinstance(v, Integer):
            if i == 0:
                raise ValueError('First variable cannot be a number: %i' % v)
            count = v
            prev, prevcount = variable_count[-1]
            if prevcount != 1:
                raise TypeError('tuple {} followed by number {}'.format((prev, prevcount), v))
            if count == 0:
                variable_count.pop()
            else:
                variable_count[-1] = Tuple(prev, count)
        else:
            if isinstance(v, array_likes):
                if len(v) == 0:
                    continue
                if isinstance(v[0], array_likes):
                    if len(v) == 1:
                        v = Array(v[0])
                        count = 1
                    else:
                        v, count = v
                        v = Array(v)
                else:
                    v, count = v
                if count == 0:
                    continue
            elif isinstance(v, UndefinedFunction):
                raise TypeError('cannot differentiate wrt UndefinedFunction: %s' % v)
            else:
                count = 1
            variable_count.append(Tuple(v, count))
    merged = []
    for t in variable_count:
        v, c = t
        if c.is_negative:
            raise ValueError('order of differentiation must be nonnegative')
        if merged and merged[-1][0] == v:
            c += merged[-1][1]
            if not c:
                merged.pop()
            else:
                merged[-1] = Tuple(v, c)
        else:
            merged.append(t)
    variable_count = merged
    for v, c in variable_count:
        if not v._diff_wrt:
            __ = ''
            raise ValueError(filldedent("\n                Can't calculate derivative wrt %s.%s" % (v, __)))
    if len(variable_count) == 0:
        return expr
    evaluate = kwargs.get('evaluate', False)
    if evaluate:
        if isinstance(expr, Derivative):
            expr = expr.canonical
        variable_count = [(v.canonical if isinstance(v, Derivative) else v, c) for v, c in variable_count]
        zero = False
        free = expr.free_symbols
        for v, c in variable_count:
            vfree = v.free_symbols
            if c.is_positive and vfree:
                if isinstance(v, AppliedUndef):
                    D = Dummy()
                    if not expr.xreplace({v: D}).has(D):
                        zero = True
                        break
                elif isinstance(v, MatrixExpr):
                    zero = False
                    break
                elif isinstance(v, Symbol) and v not in free:
                    zero = True
                    break
                elif not free & vfree:
                    zero = True
                    break
        if zero:
            return cls._get_zero_with_shape_like(expr)
        variable_count = cls._sort_variable_count(variable_count)
    if isinstance(expr, Derivative):
        variable_count = list(expr.variable_count) + variable_count
        expr = expr.expr
        return _derivative_dispatch(expr, *variable_count, **kwargs)
    if not evaluate or not hasattr(expr, '_eval_derivative'):
        if evaluate and variable_count == [(expr, 1)] and expr.is_scalar:
            return S.One
        return Expr.__new__(cls, expr, *variable_count)
    nderivs = 0
    unhandled = []
    for i, (v, count) in enumerate(variable_count):
        old_expr = expr
        old_v = None
        is_symbol = v.is_symbol or isinstance(v, (Iterable, Tuple, MatrixCommon, NDimArray))
        if not is_symbol:
            old_v = v
            v = Dummy('xi')
            expr = expr.xreplace({old_v: v})
            clashing = not (isinstance(old_v, Derivative) or isinstance(old_v, AppliedUndef))
            if not v in expr.free_symbols and (not clashing):
                return expr.diff(v)
            if not old_v.is_scalar and (not hasattr(old_v, '_eval_derivative')):
                expr *= old_v.diff(old_v)
        obj = cls._dispatch_eval_derivative_n_times(expr, v, count)
        if obj is not None and obj.is_zero:
            return obj
        nderivs += count
        if old_v is not None:
            if obj is not None:
                obj = obj.subs(v, old_v)
            expr = old_expr
        if obj is None:
            unhandled = variable_count[i:]
            break
        expr = obj
    expr = expr.replace(lambda x: isinstance(x, Derivative), lambda x: x.canonical)
    if unhandled:
        if isinstance(expr, Derivative):
            unhandled = list(expr.variable_count) + unhandled
            expr = expr.expr
        expr = Expr.__new__(cls, expr, *unhandled)
    if (nderivs > 1) == True and kwargs.get('simplify', True):
        from sympy.core.exprtools import factor_terms
        from sympy.simplify.simplify import signsimp
        expr = factor_terms(signsimp(expr))
    return expr

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    is_sympy = getattr(a, '__sympy__', None)
    if is_sympy is True:
        return a
    elif is_sympy is not None:
        if not strict:
            return a
        else:
            raise SympifyError(a)
    if isinstance(a, CantSympify):
        raise SympifyError(a)
    cls = getattr(a, '__class__', None)
    if cls is None:
        cls = type(a)
    conv = converter.get(cls, None)
    if conv is not None:
        return conv(a)
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
        evaluate = global_parameters.evaluate
    if _is_numpy_instance(a):
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
    if not isinstance(a, str):
        if _is_numpy_instance(a):
            import numpy as np
            assert not isinstance(a, np.number)
            if isinstance(a, np.ndarray):
                if a.ndim == 0:
                    try:
                        return sympify(a.item(), locals=locals, convert_xor=convert_xor, strict=strict, rational=rational, evaluate=evaluate)
                    except SympifyError:
                        pass
        else:
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
    if not isinstance(a, str):
        try:
            a = str(a)
        except Exception as exc:
            raise SympifyError(a, exc)
        from sympy.utilities.exceptions import SymPyDeprecationWarning
        SymPyDeprecationWarning(feature='String fallback in sympify', useinstead='sympify(str(obj)) or ' + 'sympy.core.sympify.converter or obj._sympy_', issue=18066, deprecated_since_version='1.6').warn()
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

def free_symbols(self):
    return {self}

def __hash__(self) -> int:
    h = self._mhash
    if h is None:
        h = hash((type(self).__name__,) + self._hashable_content())
        self._mhash = h
    return h

def __new__(cls, *args, **kwargs):
    if kwargs.get('sympify', True):
        args = (sympify(arg) for arg in args)
    obj = Basic.__new__(cls, *args)
    return obj



from typing import Tuple as tTuple
from sympy.core.logic import FuzzyBool
from functools import wraps, reduce
import collections
from sympy.core import S, Symbol, Integer, Basic, Expr, Mul, Add
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import SYMPY_INTS, default_sort_key
from sympy.core.symbol import Str
from sympy.core.sympify import SympifyError, _sympify
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.simplify import simplify
from sympy.matrices.matrices import MatrixKind
from sympy.utilities.misc import filldedent
from sympy.multipledispatch import dispatch
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.tensor.array.expressions.conv_array_to_matrix import convert_array_to_matrix
from sympy import ImmutableDenseMatrix
from sympy.matrices.expressions.adjoint import Adjoint
from sympy.matrices.expressions.transpose import Transpose
from sympy import I
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.adjoint import Adjoint
from sympy.core.assumptions import check_assumptions
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.immutable import ImmutableDenseMatrix
from numpy import empty
from sympy import Sum, Mul, Add, MatMul, transpose, trace
from sympy.strategies.traverse import bottom_up
from .applyfunc import ElementwiseApplyFunction
from sympy import MatrixBase
from sympy import Sum, symbols, Dummy
from sympy.core.expr import ExprBuilder
from sympy.core.expr import ExprBuilder
from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
from ...tensor.array.expressions.array_expressions import ArrayContraction
from sympy.matrices.expressions.slice import MatrixSlice
from sympy import MatrixBase
from sympy.matrices.expressions.slice import MatrixSlice
Basic._constructor_postprocessor_mapping[MatrixExpr] = {'Mul': [get_postprocessor(Mul)], 'Add': [get_postprocessor(Add)]}

def _matrix_derivative(expr, x):
    from sympy.tensor.array.array_derivatives import ArrayDerivative
    lines = expr._eval_derivative_matrix_lines(x)
    parts = [i.build() for i in lines]
    from sympy.tensor.array.expressions.conv_array_to_matrix import convert_array_to_matrix
    parts = [[convert_array_to_matrix(j) for j in i] for i in parts]

    def _get_shape(elem):
        if isinstance(elem, MatrixExpr):
            return elem.shape
        return (1, 1)

    def get_rank(parts):
        return sum([j not in (1, None) for i in parts for j in _get_shape(i)])
    ranks = [get_rank(i) for i in parts]
    rank = ranks[0]

    def contract_one_dims(parts):
        if len(parts) == 1:
            return parts[0]
        else:
            p1, p2 = parts[:2]
            if p2.is_Matrix:
                p2 = p2.T
            if p1 == Identity(1):
                pbase = p2
            elif p2 == Identity(1):
                pbase = p1
            else:
                pbase = p1 * p2
            if len(parts) == 2:
                return pbase
            else:
                if pbase.is_Matrix:
                    raise ValueError('')
                return pbase * Mul.fromiter(parts[2:])
    if rank <= 2:
        return Add.fromiter([contract_one_dims(i) for i in parts])
    return ArrayDerivative(expr, x)