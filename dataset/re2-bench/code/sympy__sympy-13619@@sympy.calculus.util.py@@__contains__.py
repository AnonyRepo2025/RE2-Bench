from sympy import Order, S, log, limit, lcm_list, pi, Abs
from sympy.core.basic import Basic
from sympy.core import Add, Mul, Pow
from sympy.logic.boolalg import And
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.numbers import _sympifyit, oo
from sympy.core.sympify import _sympify
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
                             Complement, EmptySet)
from sympy.sets.conditionset import ConditionSet
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.utilities import filldedent
from sympy.simplify.radsimp import denom
from sympy.polys.rationaltools import together
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.solvers.solveset import solveset, _has_rational_power
from sympy.solvers.solveset import solveset
from sympy.core.function import diff
from sympy.core.mod import Mod
from sympy.core.relational import Relational
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import (
        TrigonometricFunction, sin, cos, csc, sec)
from sympy.simplify.simplify import simplify
from sympy.solvers.decompogen import decompogen
from sympy.polys.polytools import degree, lcm_list
from sympy.solvers.solveset import solveset
from sympy.functions.elementary.miscellaneous import real_root
import sys
from sympy.solvers.decompogen import compogen

AccumBounds = AccumulationBounds

class AccumulationBounds(AtomicExpr):
    is_real = True
    _op_priority = 11.0
    __radd__ = __add__
    __rmul__ = __mul__
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    def __contains__(self, other):
        """
        Returns True if other is contained in self, where other
        belongs to extended real numbers, False if not contained,
        otherwise TypeError is raised.

        Examples
        ========

        >>> from sympy import AccumBounds, oo
        >>> 1 in AccumBounds(-1, 3)
        True

        -oo and oo go together as limits (in AccumulationBounds).

        >>> -oo in AccumBounds(1, oo)
        True

        >>> oo in AccumBounds(-oo, 0)
        True

        """
        other = _sympify(other)

        if other is S.Infinity or other is S.NegativeInfinity:
            if self.min is S.NegativeInfinity or self.max is S.Infinity:
                return True
            return False

        rv = And(self.min <= other, self.max >= other)
        if rv not in (True, False):
            raise TypeError("input failed to evaluate")
        return rv