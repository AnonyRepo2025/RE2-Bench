from __future__ import print_function, division
from functools import wraps
import inspect
import keyword
import re
import textwrap
import linecache
from sympy.core.compatibility import (exec_, is_sequence, iterable,
    NotIterable, string_types, range, builtins, integer_types, PY3)
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
NUMPY_DEFAULT = {"I": 1j}
TENSORFLOW_DEFAULT = {}
SYMPY_DEFAULT = {}
NUMEXPR_DEFAULT = {}
MATH_TRANSLATIONS = {
    "ceiling": "ceil",
    "E": "e",
    "ln": "log",
}
MPMATH_TRANSLATIONS = {
    "Abs": "fabs",
    "elliptic_k": "ellipk",
    "elliptic_f": "ellipf",
    "elliptic_e": "ellipe",
    "elliptic_pi": "ellippi",
    "ceiling": "ceil",
    "chebyshevt": "chebyt",
    "chebyshevu": "chebyu",
    "E": "e",
    "I": "j",
    "ln": "log",
    #"lowergamma":"lower_gamma",
    "oo": "inf",
    #"uppergamma":"upper_gamma",
    "LambertW": "lambertw",
    "MutableDenseMatrix": "matrix",
    "ImmutableDenseMatrix": "matrix",
    "conjugate": "conj",
    "dirichlet_eta": "altzeta",
    "Ei": "ei",
    "Shi": "shi",
    "Chi": "chi",
    "Si": "si",
    "Ci": "ci",
    "RisingFactorial": "rf",
    "FallingFactorial": "ff",
}
NUMPY_TRANSLATIONS = {}
TENSORFLOW_TRANSLATIONS = {
    "Abs": "abs",
    "ceiling": "ceil",
    "im": "imag",
    "ln": "log",
    "Mod": "mod",
    "conjugate": "conj",
    "re": "real",
}
NUMEXPR_TRANSLATIONS = {}
MODULES = {
    "math": (MATH, MATH_DEFAULT, MATH_TRANSLATIONS, ("from math import *",)),
    "mpmath": (MPMATH, MPMATH_DEFAULT, MPMATH_TRANSLATIONS, ("from mpmath import *",)),
    "numpy": (NUMPY, NUMPY_DEFAULT, NUMPY_TRANSLATIONS, ("import numpy; from numpy import *",)),
    "tensorflow": (TENSORFLOW, TENSORFLOW_DEFAULT, TENSORFLOW_TRANSLATIONS, ("import_module('tensorflow')",)),
    "sympy": (SYMPY, SYMPY_DEFAULT, {}, (
        "from sympy.functions import *",
        "from sympy.matrices import *",
        "from sympy import Integral, pi, oo, nan, zoo, E, I",)),
    "numexpr" : (NUMEXPR, NUMEXPR_DEFAULT, NUMEXPR_TRANSLATIONS,
                 ("import_module('numexpr')", )),
}
_lambdify_generated_counter = 1

class _EvaluatorPrinter(object):
    def _preprocess(self, args, expr):
        """Preprocess args, expr to replace arguments that do not map
        to valid Python identifiers.

        Returns string form of args, and updated expr.
        """
        from sympy import Dummy, Symbol, MatrixSymbol, Function, flatten
        from sympy.matrices import DeferredVector

        dummify = self._dummify

        # Args of type Dummy can cause name collisions with args
        # of type Symbol.  Force dummify of everything in this
        # situation.
        if not dummify:
            dummify = any(isinstance(arg, Dummy) for arg in flatten(args))

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

        return argstrs, expr
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
                expr = tuple(self._subexpr(sympify(a), dummies_dict) for a in expr)
            elif isinstance(expr, list):
                expr = [self._subexpr(sympify(a), dummies_dict) for a in expr]
        return expr