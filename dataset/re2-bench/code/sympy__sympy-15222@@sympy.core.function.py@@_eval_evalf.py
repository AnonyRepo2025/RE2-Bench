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



class Function(Application, Expr):
    def _eval_evalf(self, prec):
        # Lookup mpmath function based on name
        try:
            if isinstance(self, AppliedUndef):
                # Shouldn't lookup in mpmath but might have ._imp_
                raise AttributeError
            fname = self.func.__name__
            if not hasattr(mpmath, fname):
                from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
                fname = MPMATH_TRANSLATIONS[fname]
            func = getattr(mpmath, fname)
        except (AttributeError, KeyError):
            try:
                return Float(self._imp_(*[i.evalf(prec) for i in self.args]), prec)
            except (AttributeError, TypeError, ValueError):
                return

        # Convert all args to mpf or mpc
        # Convert the arguments to *higher* precision than requested for the
        # final result.
        # XXX + 5 is a guess, it is similar to what is used in evalf.py. Should
        #     we be more intelligent about it?
        try:
            args = [arg._to_mpmath(prec + 5) for arg in self.args]
            def bad(m):
                from mpmath import mpf, mpc
                # the precision of an mpf value is the last element
                # if that is 1 (and m[1] is not 1 which would indicate a
                # power of 2), then the eval failed; so check that none of
                # the arguments failed to compute to a finite precision.
                # Note: An mpc value has two parts, the re and imag tuple;
                # check each of those parts, too. Anything else is allowed to
                # pass
                if isinstance(m, mpf):
                    m = m._mpf_
                    return m[1] !=1 and m[-1] == 1
                elif isinstance(m, mpc):
                    m, n = m._mpc_
                    return m[1] !=1 and m[-1] == 1 and \
                        n[1] !=1 and n[-1] == 1
                else:
                    return False
            if any(bad(a) for a in args):
                raise ValueError  # one or more args failed to compute with significance
        except ValueError:
            return

        with mpmath.workprec(prec):
            v = func(*args)

        return Expr._from_mpmath(v, prec)