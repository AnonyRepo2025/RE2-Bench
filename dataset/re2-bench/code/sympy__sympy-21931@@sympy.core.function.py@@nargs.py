from typing import Any, Dict as tDict, Optional, Set as tSet, Tuple as tTuple, Union
from .add import Add
from .assumptions import ManagedProperties
from .basic import Basic, _atomic
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
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from sympy.utilities import default_sort_key
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.iterables import has_dups, sift
from sympy.utilities.misc import filldedent
import mpmath
import mpmath.libmp as mlib
import inspect
from collections import Counter
from sympy.core.symbol import Dummy, Symbol
from sympy.matrices.common import MatrixCommon
from sympy import MatrixExpr
from sympy import NDimArray
from sympy import Mul, log
from sympy import Integral, Sum, Symbol
from sympy.core.relational import Relational
from sympy.simplify.radsimp import fraction
from sympy.logic.boolalg import BooleanFunction
from sympy.utilities.misc import func_name
from sympy.core.power import Pow
from sympy.polys.rootoftools import RootOf
from sympy import MatrixBase
from sympy.sets.sets import FiniteSet
from sympy.sets.fancysets import Naturals0
from sympy.sets.sets import FiniteSet
from sympy.core.evalf import pure_complex
from sympy.sets.fancysets import Naturals0
from sympy.utilities.misc import filldedent
from sympy import Order
from sympy.core.symbol import uniquely_named_symbol
from sympy.sets.sets import FiniteSet
from sympy import Order
import sage.all as sage
import sage.all as sage
import sage.all as sage
from .symbol import _filter_assumptions
from sympy.sets.sets import Set, FiniteSet
from sympy.matrices.common import MatrixCommon
from sympy import Integer, MatrixExpr
from sympy.tensor.array import Array, NDimArray
from sympy.utilities.misc import filldedent
from sympy.utilities.iterables import uniq, topological_sort
import sage.all as sage
from ..calculus.finite_diff import _as_finite_diff
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.sets.sets import FiniteSet
from sympy.core.evalf import prec_to_dps
from sympy import Symbol
from sympy.printing import StrPrinter
from sympy.utilities.exceptions import SymPyDeprecationWarning
from inspect import signature
from sympy import oo, zoo, nan
import sympy
from sympy.core.exprtools import factor_terms
from sympy.simplify.simplify import signsimp
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from mpmath import mpf, mpc

_undef_sage_helper = UndefSageHelper()

class FunctionClass(ManagedProperties):
    _new = type.__new__
    @property
    def nargs(self):
        """Return a set of the allowed number of arguments for the function.

        Examples
        ========

        >>> from sympy.core.function import Function
        >>> f = Function('f')

        If the function can take any number of arguments, the set of whole
        numbers is returned:

        >>> Function('f').nargs
        Naturals0

        If the function was initialized to accept one or more arguments, a
        corresponding set will be returned:

        >>> Function('f', nargs=1).nargs
        {1}
        >>> Function('f', nargs=(2, 1)).nargs
        {1, 2}

        The undefined function, after application, also has the nargs
        attribute; the actual number of arguments is always available by
        checking the ``args`` attribute:

        >>> f = Function('f')
        >>> f(1).nargs
        Naturals0
        >>> len(f(1).args)
        1
        """
        from sympy.sets.sets import FiniteSet
        # XXX it would be nice to handle this in __init__ but there are import
        # problems with trying to import FiniteSet there
        return FiniteSet(*self._nargs) if self._nargs else S.Naturals0