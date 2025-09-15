from __future__ import print_function, division
from .add import Add
from .assumptions import ManagedProperties
from .basic import Basic
from .cache import cacheit
from .compatibility import iterable, is_sequence, as_int, ordered
from .decorators import _sympifyit
from .expr import Expr, AtomicExpr
from .numbers import Rational, Float
from .operations import LatticeOp
from .rules import Transform
from .singleton import S
from .sympify import sympify
from sympy.core.containers import Tuple, Dict
from sympy.core.logic import fuzzy_and
from sympy.core.compatibility import string_types, with_metaclass, range
from sympy.utilities import default_sort_key
from sympy.utilities.misc import filldedent
from sympy.utilities.iterables import uniq
from sympy.core.evaluate import global_evaluate
import sys
import mpmath
import mpmath.libmp as mlib
import inspect
import collections
from sympy.core.symbol import Dummy, Symbol
from sympy import Integral, Symbol
from sympy.simplify.radsimp import fraction
from sympy.logic.boolalg import BooleanFunction
from sympy.core.power import Pow
from sympy.polys.rootoftools import RootOf
from sympy.sets.sets import FiniteSet
from sympy.sets.fancysets import Naturals0
from sympy.sets.sets import FiniteSet
from sympy.core.symbol import Wild
from sympy.sets.fancysets import Naturals0
from sympy.utilities.misc import filldedent
from sympy import Order
from sympy.sets.sets import FiniteSet
from sympy import Order
import sage.all as sage
import sage.all as sage
from sympy.sets.sets import Set, FiniteSet
import mpmath
from sympy.core.expr import Expr
import sage.all as sage
from ..calculus.finite_diff import _as_finite_diff
from sympy.sets.sets import FiniteSet
from sympy import Symbol
from sympy.printing import StrPrinter
from inspect import signature
from sympy import oo, zoo, nan
from sympy.core.exprtools import factor_terms
from sympy.simplify.simplify import signsimp
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from mpmath import mpf, mpc
from sympy.utilities.misc import filldedent
from sympy.utilities.misc import filldedent

UndefinedFunction.__eq__ = lambda s, o: (isinstance(o, s.__class__) and
                                         (s.class_key() == o.class_key()))

class Function(Application, Expr):
    @cacheit
    def __new__(cls, *args, **options):
        # Handle calls like Function('f')
        if cls is Function:
            return UndefinedFunction(*args, **options)

        n = len(args)
        if n not in cls.nargs:
            # XXX: exception message must be in exactly this format to
            # make it work with NumPy's functions like vectorize(). See,
            # for example, https://github.com/numpy/numpy/issues/1697.
            # The ideal solution would be just to attach metadata to
            # the exception and change NumPy to take advantage of this.
            temp = ('%(name)s takes %(qual)s %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise TypeError(temp % {
                'name': cls,
                'qual': 'exactly' if len(cls.nargs) == 1 else 'at least',
                'args': min(cls.nargs),
                'plural': 's'*(min(cls.nargs) != 1),
                'given': n})

        evaluate = options.get('evaluate', global_evaluate[0])
        result = super(Function, cls).__new__(cls, *args, **options)
        if evaluate and isinstance(result, cls) and result.args:
            pr2 = min(cls._should_evalf(a) for a in result.args)
            if pr2 > 0:
                pr = max(cls._should_evalf(a) for a in result.args)
                result = result.evalf(mlib.libmpf.prec_to_dps(pr))

        return result
    @classmethod
    def class_key(cls):
        from sympy.sets.fancysets import Naturals0
        funcs = {
            'exp': 10,
            'log': 11,
            'sin': 20,
            'cos': 21,
            'tan': 22,
            'cot': 23,
            'sinh': 30,
            'cosh': 31,
            'tanh': 32,
            'coth': 33,
            'conjugate': 40,
            're': 41,
            'im': 42,
            'arg': 43,
        }
        name = cls.__name__

        try:
            i = funcs[name]
        except KeyError:
            i = 0 if isinstance(cls.nargs, Naturals0) else 10000

        return 4, i, name