from __future__ import print_function, division
from .add import Add
from .assumptions import ManagedProperties, _assume_defined
from .basic import Basic
from .cache import cacheit
from .compatibility import iterable, is_sequence, as_int, ordered, Iterable
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
from sympy.utilities.iterables import has_dups
from sympy.core.evaluate import global_evaluate
import sys
import mpmath
import mpmath.libmp as mlib
import inspect
from collections import Counter
from sympy.core.symbol import Dummy, Symbol
from sympy import Integral, Symbol
from sympy.core.relational import Relational
from sympy.simplify.radsimp import fraction
from sympy.logic.boolalg import BooleanFunction
from sympy.utilities.misc import func_name
from sympy.core.power import Pow
from sympy.polys.rootoftools import RootOf
from sympy.sets.sets import FiniteSet
from sympy.sets.fancysets import Naturals0
from sympy.sets.sets import FiniteSet
from sympy.core.evalf import pure_complex
from sympy.sets.fancysets import Naturals0
from sympy.utilities.misc import filldedent
from sympy import Order
from sympy.sets.sets import FiniteSet
from sympy import Order
import sage.all as sage
import sage.all as sage
from sympy.sets.sets import Set, FiniteSet
from sympy.matrices.common import MatrixCommon
from sympy import Integer
from sympy.tensor.array import Array, NDimArray, derive_by_array
from sympy.utilities.misc import filldedent
from sympy import Integer
import mpmath
from sympy.core.expr import Expr
import sage.all as sage
from ..calculus.finite_diff import _as_finite_diff
from sympy.sets.sets import FiniteSet
from sympy import Symbol
from sympy.printing import StrPrinter
from inspect import signature
from sympy import oo, zoo, nan
import sympy
from sympy.core.exprtools import factor_terms
from sympy.simplify.simplify import signsimp
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from mpmath import mpf, mpc



class Application(with_metaclass(FunctionClass, Basic)
):
    is_Function = True
    @cacheit
    def __new__(cls, *args, **options):
        from sympy.sets.fancysets import Naturals0
        from sympy.sets.sets import FiniteSet

        args = list(map(sympify, args))
        evaluate = options.pop('evaluate', global_evaluate[0])
        # WildFunction (and anything else like it) may have nargs defined
        # and we throw that value away here
        options.pop('nargs', None)

        if options:
            raise ValueError("Unknown options: %s" % options)

        if evaluate:
            evaluated = cls.eval(*args)
            if evaluated is not None:
                return evaluated

        obj = super(Application, cls).__new__(cls, *args, **options)

        # make nargs uniform here
        try:
            # things passing through here:
            #  - functions subclassed from Function (e.g. myfunc(1).nargs)
            #  - functions like cos(1).nargs
            #  - AppliedUndef with given nargs like Function('f', nargs=1)(1).nargs
            # Canonicalize nargs here
            if is_sequence(obj.nargs):
                nargs = tuple(ordered(set(obj.nargs)))
            elif obj.nargs is not None:
                nargs = (as_int(obj.nargs),)
            else:
                nargs = None
        except AttributeError:
            # things passing through here:
            #  - WildFunction('f').nargs
            #  - AppliedUndef with no nargs like Function('f')(1).nargs
            nargs = obj._nargs  # note the underscore here
        # convert to FiniteSet
        obj.nargs = FiniteSet(*nargs) if nargs else Naturals0()
        return obj
    @classmethod
    def eval(cls, *args):
        """
        Returns a canonical form of cls applied to arguments args.

        The eval() method is called when the class cls is about to be
        instantiated and it should return either some simplified instance
        (possible of some other class), or if the class cls should be
        unmodified, return None.

        Examples of eval() for the function "sign"
        ---------------------------------------------

        .. code-block:: python

            @classmethod
            def eval(cls, arg):
                if arg is S.NaN:
                    return S.NaN
                if arg is S.Zero: return S.Zero
                if arg.is_positive: return S.One
                if arg.is_negative: return S.NegativeOne
                if isinstance(arg, Mul):
                    coeff, terms = arg.as_coeff_Mul(rational=True)
                    if coeff is not S.One:
                        return cls(coeff) * cls(terms)

        """
        return
    @property
    def func(self):
        return self.__class__
    def _eval_subs(self, old, new):
        if (old.is_Function and new.is_Function and
            callable(old) and callable(new) and
            old == self.func and len(self.args) in new.nargs):
            return new(*[i._subs(old, new) for i in self.args])