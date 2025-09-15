def _sympify(a):
    return sympify(a, strict=True)

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    is_sympy = getattr(a, '__sympy__', None)
    if is_sympy is not None:
        return a
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

def _hashable_content(self):
    return (self.name,) + tuple(sorted(self.assumptions0.items()))

def assumptions0(self):
    return dict(((key, value) for key, value in self._assumptions.items() if value is not None))

def _hashable_content(self):
    return Symbol._hashable_content(self) + (self.dummy_index,)

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

def __new__(cls, *args):
    obj = object.__new__(cls)
    obj._assumptions = cls.default_assumptions
    obj._mhash = None
    obj._args = args
    return obj

def __init__(self, expr, base_exc=None):
    self.expr = expr
    self.base_exc = base_exc

def __new__(cls, num, dps=None, prec=None, precision=None):
    if prec is not None:
        SymPyDeprecationWarning(feature="Using 'prec=XX' to denote decimal precision", useinstead="'dps=XX' for decimal precision and 'precision=XX' for binary precision", issue=12820, deprecated_since_version='1.1').warn()
        dps = prec
    del prec
    if dps is not None and precision is not None:
        raise ValueError('Both decimal and binary precision supplied. Supply only one. ')
    if isinstance(num, string_types):
        num = num.replace(' ', '').lower()
        if '_' in num:
            parts = num.split('_')
            if not (all(parts) and all((parts[i][-1].isdigit() for i in range(0, len(parts), 2))) and all((parts[i][0].isdigit() for i in range(1, len(parts), 2)))):
                raise ValueError("could not convert string to float: '%s'" % num)
            num = ''.join(parts)
        if num.startswith('.') and len(num) > 1:
            num = '0' + num
        elif num.startswith('-.') and len(num) > 2:
            num = '-0.' + num[2:]
        elif num in ('inf', '+inf'):
            return S.Infinity
        elif num == '-inf':
            return S.NegativeInfinity
    elif isinstance(num, float) and num == 0:
        num = '0'
    elif isinstance(num, float) and num == float('inf'):
        return S.Infinity
    elif isinstance(num, float) and num == float('-inf'):
        return S.NegativeInfinity
    elif isinstance(num, float) and num == float('nan'):
        return S.NaN
    elif isinstance(num, (SYMPY_INTS, Integer)):
        num = str(num)
    elif num is S.Infinity:
        return num
    elif num is S.NegativeInfinity:
        return num
    elif num is S.NaN:
        return num
    elif type(num).__module__ == 'numpy':
        num = _convert_numpy_types(num)
    elif isinstance(num, mpmath.mpf):
        if precision is None:
            if dps is None:
                precision = num.context.prec
        num = num._mpf_
    if dps is None and precision is None:
        dps = 15
        if isinstance(num, Float):
            return num
        if isinstance(num, string_types) and _literal_float(num):
            try:
                Num = decimal.Decimal(num)
            except decimal.InvalidOperation:
                pass
            else:
                isint = '.' not in num
                num, dps = _decimal_to_Rational_prec(Num)
                if num.is_Integer and isint:
                    dps = max(dps, len(str(num).lstrip('-')))
                dps = max(15, dps)
                precision = mlib.libmpf.dps_to_prec(dps)
    elif precision == '' and dps is None or (precision is None and dps == ''):
        if not isinstance(num, string_types):
            raise ValueError('The null string can only be used when the number to Float is passed as a string or an integer.')
        ok = None
        if _literal_float(num):
            try:
                Num = decimal.Decimal(num)
            except decimal.InvalidOperation:
                pass
            else:
                isint = '.' not in num
                num, dps = _decimal_to_Rational_prec(Num)
                if num.is_Integer and isint:
                    dps = max(dps, len(str(num).lstrip('-')))
                    precision = mlib.libmpf.dps_to_prec(dps)
                ok = True
        if ok is None:
            raise ValueError('string-float not recognized: %s' % num)
    if precision is None or precision == '':
        precision = mlib.libmpf.dps_to_prec(dps)
    precision = int(precision)
    if isinstance(num, float):
        _mpf_ = mlib.from_float(num, precision, rnd)
    elif isinstance(num, string_types):
        _mpf_ = mlib.from_str(num, precision, rnd)
    elif isinstance(num, decimal.Decimal):
        if num.is_finite():
            _mpf_ = mlib.from_str(str(num), precision, rnd)
        elif num.is_nan():
            return S.NaN
        elif num.is_infinite():
            if num > 0:
                return S.Infinity
            return S.NegativeInfinity
        else:
            raise ValueError('unexpected decimal value %s' % str(num))
    elif isinstance(num, tuple) and len(num) in (3, 4):
        if type(num[1]) is str:
            num = list(num)
            if num[1].endswith('L'):
                num[1] = num[1][:-1]
            num[1] = MPZ(num[1], 16)
            _mpf_ = tuple(num)
        elif len(num) == 4:
            return Float._new(num, precision)
        else:
            if not all((num[0] in (0, 1), num[1] >= 0, all((type(i) in (long, int) for i in num)))):
                raise ValueError('malformed mpf: %s' % (num,))
            return Float._new((num[0], num[1], num[2], bitcount(num[1])), precision)
    else:
        try:
            _mpf_ = num._as_mpf_val(precision)
        except (NotImplementedError, AttributeError):
            _mpf_ = mpmath.mpf(num, prec=precision)._mpf_
    return cls._new(_mpf_, precision, zero=False)

def __eq__(self, other):
    if isinstance(other, Basic):
        return super(Tuple, self).__eq__(other)
    return self.args == other

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

def _hashable_content(self):
    return self._args

def __eq__(self, other):
    try:
        other = _sympify(other)
    except SympifyError:
        return NotImplemented
    if self is other:
        return True
    if other.is_Number and self.is_irrational:
        return False
    return False

def __eq__(self, other):
    if isinstance(other, integer_types):
        return self.p == other
    elif isinstance(other, Integer):
        return self.p == other.p
    return Rational.__eq__(self, other)



from __future__ import print_function, division
from .sympify import sympify, _sympify, SympifyError
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex
from .decorators import _sympifyit, call_highest_priority
from .cache import cacheit
from .compatibility import reduce, as_int, default_sort_key, range, Iterable
from sympy.utilities.misc import func_name
from mpmath.libmp import mpf_log, prec_to_dps
from collections import defaultdict
from .mul import Mul
from .add import Add
from .power import Pow
from .function import Derivative, Function
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Integer, Rational
from math import log10, ceil, log
from sympy import Float
from sympy import Abs
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy import Dummy
from sympy import GreaterThan
from sympy import LessThan
from sympy import StrictGreaterThan
from sympy import StrictLessThan
from sympy import Float
from sympy.simplify.simplify import nsimplify, simplify
from sympy.solvers.solvers import solve
from sympy.polys.polyerrors import NotAlgebraic
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.polyerrors import NotAlgebraic
from sympy.series import limit, Limit
from sympy.solvers.solveset import solveset
from sympy.sets.sets import Interval
from sympy.functions.elementary.exponential import log
from sympy.calculus.util import AccumBounds
from sympy.functions.elementary.complexes import conjugate as c
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.complexes import transpose
from sympy.functions.elementary.complexes import conjugate, transpose
from sympy.functions.elementary.complexes import adjoint
from sympy.polys.orderings import monomial_key
from .numbers import Number, NumberSymbol
from .add import Add
from .mul import Mul
from .exprtools import decompose_power
from sympy import Dummy, Symbol
from .function import count_ops
from .symbol import Symbol
from .add import _unevaluated_Add
from .mul import _unevaluated_Mul
from sympy.utilities.iterables import sift
from sympy import im, re
from .mul import _unevaluated_Mul
from .add import _unevaluated_Add
from sympy import exp_polar, pi, I, ceiling, Add
from sympy import collect, Dummy, Order, Rational, Symbol, ceiling
from sympy import Order, Dummy
from sympy.functions import exp, log
from sympy.series.gruntz import mrv, rewrite
from sympy import Dummy, factorial
from sympy.utilities.misc import filldedent
from sympy.series.limits import limit
from sympy import Dummy, log, Piecewise, piecewise_fold
from sympy.series.gruntz import calculate_series
from sympy import powsimp
from sympy import collect
from sympy import Dummy, log
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.simplify.radsimp import fraction
from sympy.integrals import integrate
from sympy.simplify import simplify
from sympy.simplify import nsimplify
from sympy.core.function import expand_power_base
from sympy.simplify import collect
from sympy.polys import together
from sympy.polys import apart
from sympy.simplify import ratsimp
from sympy.simplify import trigsimp
from sympy.simplify import radsimp
from sympy.simplify import powsimp
from sympy.simplify import combsimp
from sympy.simplify import gammasimp
from sympy.polys import factor
from sympy.assumptions import refine
from sympy.polys import cancel
from sympy.polys.polytools import invert
from sympy.core.numbers import mod_inverse
from sympy.core.numbers import Float
from sympy.matrices.expressions.matexpr import _LeftRightArgs
from sympy import Piecewise, Eq
from sympy import Tuple, MatrixExpr
from sympy.matrices.common import MatrixCommon
from sympy.utilities.randtest import random_complex_number
from mpmath.libmp.libintmath import giant_steps
from sympy.core.evalf import DEFAULT_MAXPREC as target
from sympy.solvers.solvers import denoms
from sympy.utilities.misc import filldedent
from sympy.core.numbers import mod_inverse

class Expr(Basic, EvalfMixin):
    __slots__ = []
    is_scalar = True
    _op_priority = 10.0
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __long__ = __int__
    __round__ = round

    def _hashable_content(self):
        return self._args

    def __eq__(self, other):
        try:
            other = _sympify(other)
            if not isinstance(other, Expr):
                return False
        except (SympifyError, SyntaxError):
            return False
        if not (self.is_Number and other.is_Number) and type(self) != type(other):
            return False
        a, b = (self._hashable_content(), other._hashable_content())
        if a != b:
            return False
        for a, b in zip(a, b):
            if not isinstance(a, Expr):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True