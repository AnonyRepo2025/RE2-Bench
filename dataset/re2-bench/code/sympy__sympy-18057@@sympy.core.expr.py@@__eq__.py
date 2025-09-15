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
    is_scalar = True  # self derivative is 1
    _op_priority = 10.0
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __long__ = __int__
    __round__ = round
    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.
        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args
    def __eq__(self, other):
        try:
            other = _sympify(other)
            if not isinstance(other, Expr):
                return False
        except (SympifyError, SyntaxError):
            return False
        # check for pure number expr
        if  not (self.is_Number and other.is_Number) and (
                type(self) != type(other)):
            return False
        a, b = self._hashable_content(), other._hashable_content()
        if a != b:
            return False
        # check number *in* an expression
        for a, b in zip(a, b):
            if not isinstance(a, Expr):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True