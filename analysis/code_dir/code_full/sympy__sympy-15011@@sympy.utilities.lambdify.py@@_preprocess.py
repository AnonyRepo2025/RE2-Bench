def flatten(iterable, levels=None, cls=None):
    if levels is not None:
        if not levels:
            return iterable
        elif levels > 0:
            levels -= 1
        else:
            raise ValueError('expected non-negative number of levels, got %s' % levels)
    if cls is None:
        reducible = lambda x: is_sequence(x, set)
    else:
        reducible = lambda x: isinstance(x, cls)
    result = []
    for el in iterable:
        if reducible(el):
            if hasattr(el, 'args'):
                el = el.args
            result.extend(flatten(el, levels=levels, cls=cls))
        else:
            result.append(el)
    return result

def is_sequence(i, include=None):
    return hasattr(i, '__getitem__') and iterable(i) or (bool(include) and isinstance(i, include))

def iterable(i, exclude=(string_types, dict, NotIterable)):
    if hasattr(i, '_iterable'):
        return i._iterable
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True

def doprint(self, expr, assign_to=None):
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(assign_to, string_types):
        if expr.is_Matrix:
            assign_to = MatrixSymbol(assign_to, *expr.shape)
        else:
            assign_to = Symbol(assign_to)
    elif not isinstance(assign_to, (Basic, type(None))):
        raise TypeError('{0} cannot assign to object of type {1}'.format(type(self).__name__, type(assign_to)))
    if assign_to:
        expr = Assignment(assign_to, expr)
    else:
        expr = sympify(expr)
    self._not_supported = set()
    self._number_symbols = set()
    lines = self._print(expr).splitlines()
    if self._settings['human']:
        frontlines = []
        if len(self._not_supported) > 0:
            frontlines.append(self._get_comment('Not supported in {0}:'.format(self.language)))
            for expr in sorted(self._not_supported, key=str):
                frontlines.append(self._get_comment(type(expr).__name__))
        for name, value in sorted(self._number_symbols, key=str):
            frontlines.append(self._declare_number_const(name, value))
        lines = frontlines + lines
        lines = self._format_code(lines)
        result = '\n'.join(lines)
    else:
        lines = self._format_code(lines)
        num_syms = set([(k, self._print(v)) for k, v in self._number_symbols])
        result = (num_syms, self._not_supported, '\n'.join(lines))
    del self._not_supported
    del self._number_symbols
    return result

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

def __hash__(self):
    h = self._mhash
    if h is None:
        h = hash((type(self).__name__,) + self._hashable_content())
        self._mhash = h
    return h

def _print(self, expr, **kwargs):
    self._print_level += 1
    try:
        if self.printmethod and hasattr(expr, self.printmethod) and (not isinstance(expr, BasicMeta)):
            return getattr(expr, self.printmethod)(self, **kwargs)
        classes = type(expr).__mro__
        if AppliedUndef in classes:
            classes = classes[classes.index(AppliedUndef):]
        if UndefinedFunction in classes:
            classes = classes[classes.index(UndefinedFunction):]
        if Function in classes:
            i = classes.index(Function)
            classes = tuple((c for c in classes[:i] if c.__name__ == classes[0].__name__ or c.__name__.endswith('Base'))) + classes[i:]
        for cls in classes:
            printmethod = '_print_' + cls.__name__
            if hasattr(self, printmethod):
                return getattr(self, printmethod)(expr, **kwargs)
        return self.emptyPrinter(expr)
    finally:
        self._print_level -= 1

def _print_Symbol(self, expr):
    name = super(CodePrinter, self)._print_Symbol(expr)
    if name in self.reserved_words:
        if self._settings['error_on_reserved']:
            msg = 'This expression includes the symbol "{}" which is a reserved keyword in this language.'
            raise ValueError(msg.format(name))
        return name + self._settings['reserved_word_suffix']
    else:
        return name

def _print_Symbol(self, expr):
    return expr.name

def _format_code(self, lines):
    return lines

def _is_safe_ident(cls, ident):
    return isinstance(ident, str) and cls._safe_ident_re.match(ident) and (not (keyword.iskeyword(ident) or ident == 'None'))

def _hashable_content(self):
    return (self.name,) + tuple(sorted(self.assumptions0.items()))

def assumptions0(self):
    return dict(((key, value) for key, value in self._assumptions.items() if value is not None))

def split_symbols_custom(predicate):

    def _split_symbols(tokens, local_dict, global_dict):
        result = []
        split = False
        split_previous = False
        for tok in tokens:
            if split_previous:
                split_previous = False
                continue
            split_previous = False
            if tok[0] == NAME and tok[1] == 'Symbol':
                split = True
            elif split and tok[0] == NAME:
                symbol = tok[1][1:-1]
                if predicate(symbol):
                    for char in symbol:
                        if char in local_dict or char in global_dict:
                            del result[-2:]
                            result.extend([(NAME, '%s' % char), (NAME, 'Symbol'), (OP, '(')])
                        else:
                            result.extend([(NAME, "'%s'" % char), (OP, ')'), (NAME, 'Symbol'), (OP, '(')])
                    del result[-2:]
                    split = False
                    split_previous = True
                    continue
                else:
                    split = False
            result.append(tok)
        return result
    return _split_symbols

def parse_expr(s, local_dict=None, transformations=standard_transformations, global_dict=None, evaluate=True):
    if local_dict is None:
        local_dict = {}
    if global_dict is None:
        global_dict = {}
        exec_('from sympy import *', global_dict)
    code = stringify_expr(s, local_dict, global_dict, transformations)
    if not evaluate:
        code = compile(evaluateFalse(code), '<string>', 'eval')
    return eval_expr(code, local_dict, global_dict)



from __future__ import print_function, division
from functools import wraps
import inspect
import keyword
import re
import textwrap
import linecache
from sympy.core.compatibility import exec_, is_sequence, iterable, NotIterable, string_types, range, builtins, integer_types, PY3
from sympy.utilities.decorator import doctest_depends_on
from sympy.external import import_module
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import flatten
from sympy.matrices import DeferredVector
from sympy import Dummy, sympify, Symbol, Function, flatten
from sympy.core.function import FunctionClass
from sympy.core.function import UndefinedFunction
from sympy.printing.lambdarepr import lambdarepr
from sympy.printing.lambdarepr import LambdaPrinter
from sympy import Dummy
from sympy import Dummy, Symbol, MatrixSymbol, Function, flatten
from sympy.matrices import DeferredVector
from sympy.matrices import DeferredVector
from sympy import sympify
from sympy import flatten
from sympy.printing.pycode import MpmathPrinter as Printer
from sympy.printing.pycode import NumPyPrinter as Printer
from sympy.printing.lambdarepr import NumExprPrinter as Printer
from sympy.printing.lambdarepr import TensorflowPrinter as Printer
from sympy.printing.pycode import SymPyPrinter as Printer
from sympy.printing.pycode import PythonCodePrinter as Printer
MATH = {}
MPMATH = {}
NUMPY = {}
TENSORFLOW = {}
SYMPY = {}
NUMEXPR = {}
MATH_DEFAULT = {}
MPMATH_DEFAULT = {}
NUMPY_DEFAULT = {'I': 1j}
TENSORFLOW_DEFAULT = {}
SYMPY_DEFAULT = {}
NUMEXPR_DEFAULT = {}
MATH_TRANSLATIONS = {'ceiling': 'ceil', 'E': 'e', 'ln': 'log'}
MPMATH_TRANSLATIONS = {'Abs': 'fabs', 'elliptic_k': 'ellipk', 'elliptic_f': 'ellipf', 'elliptic_e': 'ellipe', 'elliptic_pi': 'ellippi', 'ceiling': 'ceil', 'chebyshevt': 'chebyt', 'chebyshevu': 'chebyu', 'E': 'e', 'I': 'j', 'ln': 'log', 'oo': 'inf', 'LambertW': 'lambertw', 'MutableDenseMatrix': 'matrix', 'ImmutableDenseMatrix': 'matrix', 'conjugate': 'conj', 'dirichlet_eta': 'altzeta', 'Ei': 'ei', 'Shi': 'shi', 'Chi': 'chi', 'Si': 'si', 'Ci': 'ci', 'RisingFactorial': 'rf', 'FallingFactorial': 'ff'}
NUMPY_TRANSLATIONS = {}
TENSORFLOW_TRANSLATIONS = {'Abs': 'abs', 'ceiling': 'ceil', 'im': 'imag', 'ln': 'log', 'Mod': 'mod', 'conjugate': 'conj', 're': 'real'}
NUMEXPR_TRANSLATIONS = {}
MODULES = {'math': (MATH, MATH_DEFAULT, MATH_TRANSLATIONS, ('from math import *',)), 'mpmath': (MPMATH, MPMATH_DEFAULT, MPMATH_TRANSLATIONS, ('from mpmath import *',)), 'numpy': (NUMPY, NUMPY_DEFAULT, NUMPY_TRANSLATIONS, ('import numpy; from numpy import *',)), 'tensorflow': (TENSORFLOW, TENSORFLOW_DEFAULT, TENSORFLOW_TRANSLATIONS, ("import_module('tensorflow')",)), 'sympy': (SYMPY, SYMPY_DEFAULT, {}, ('from sympy.functions import *', 'from sympy.matrices import *', 'from sympy import Integral, pi, oo, nan, zoo, E, I')), 'numexpr': (NUMEXPR, NUMEXPR_DEFAULT, NUMEXPR_TRANSLATIONS, ("import_module('numexpr')",))}
_lambdify_generated_counter = 1

class _EvaluatorPrinter(object):

    def _preprocess(self, args, expr):
        from sympy import Dummy, Symbol, MatrixSymbol, Function, flatten
        from sympy.matrices import DeferredVector
        dummify = self._dummify
        if not dummify:
            dummify = any((isinstance(arg, Dummy) for arg in flatten(args)))
        argstrs = []
        for arg in args:
            if iterable(arg):
                nested_argstrs, expr = self._preprocess(arg, expr)
                argstrs.append(nested_argstrs)
            elif isinstance(arg, DeferredVector):
                argstrs.append(str(arg))
            elif isinstance(arg, Symbol) or isinstance(arg, MatrixSymbol):
                argrep = self._argrepr(arg)
                if dummify or not self._is_safe_ident(argrep):
                    dummy = Dummy()
                    argstrs.append(self._argrepr(dummy))
                    expr = self._subexpr(expr, {arg: dummy})
                else:
                    argstrs.append(argrep)
            elif isinstance(arg, Function):
                dummy = Dummy()
                argstrs.append(self._argrepr(dummy))
                expr = self._subexpr(expr, {arg: dummy})
            else:
                argrep = self._argrepr(arg)
                if dummify:
                    dummy = Dummy()
                    argstrs.append(self._argrepr(dummy))
                    expr = self._subexpr(expr, {arg: dummy})
                else:
                    argstrs.append(str(arg))
        return (argstrs, expr)

    def _subexpr(self, expr, dummies_dict):
        from sympy.matrices import DeferredVector
        from sympy import sympify
        try:
            expr = sympify(expr).xreplace(dummies_dict)
        except Exception:
            if isinstance(expr, DeferredVector):
                pass
            elif isinstance(expr, dict):
                k = [self._subexpr(sympify(a), dummies_dict) for a in expr.keys()]
                v = [self._subexpr(sympify(a), dummies_dict) for a in expr.values()]
                expr = dict(zip(k, v))
            elif isinstance(expr, tuple):
                expr = tuple((self._subexpr(sympify(a), dummies_dict) for a in expr))
            elif isinstance(expr, list):
                expr = [self._subexpr(sympify(a), dummies_dict) for a in expr]
        return expr