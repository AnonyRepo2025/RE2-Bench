from __future__ import print_function, division
from sympy.logic.boolalg import And
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.compatibility import as_int, with_metaclass, range, PY3
from sympy.core.expr import Expr
from sympy.core.function import Lambda, _coeff_isneg
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.core.sympify import _sympify, sympify, converter
from sympy.sets.sets import (Set, Interval, Intersection, EmptySet, Union,
                             FiniteSet, imageset)
from sympy.sets.conditionset import ConditionSet
from sympy.utilities.misc import filldedent, func_name
from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
from sympy.functions.elementary.integers import floor, ceiling
from sympy.matrices import Matrix
from sympy.solvers.solveset import solveset, linsolve
from sympy.utilities.iterables import is_sequence, iterable, cartes
from sympy.solvers.diophantine import diophantine
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import ceiling
from sympy.core.function import expand_mul
from sympy import sin, cos
from sympy.functions import arg, Abs
from sympy.core.containers import Tuple
from sympy.solvers.solveset import solveset_real
from sympy.core.function import expand_complex
from sympy.solvers.diophantine import diop_linear
from sympy.core.numbers import ilcm
from sympy.solvers.solveset import (invert_real, invert_complex,
                                                solveset)
from sympy.sets import ProductSet



class ComplexRegion(Set):
    is_ComplexRegion = True
    def _union(self, other):

        if other.is_subset(S.Reals):
            # treat a subset of reals as a complex region
            other = ComplexRegion.from_real(other)

        if other.is_ComplexRegion:

            # self in rectangular form
            if (not self.polar) and (not other.polar):
                return ComplexRegion(Union(self.sets, other.sets))

            # self in polar form
            elif self.polar and other.polar:
                return ComplexRegion(Union(self.sets, other.sets), polar=True)

        return None