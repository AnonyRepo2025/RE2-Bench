from sympy.core.backend import (S, sympify, expand, sqrt, Add, zeros,
    ImmutableMatrix as Matrix)
from sympy import trigsimp
from sympy.core.compatibility import unicode
from sympy.utilities.misc import filldedent
from sympy.physics.vector.dyadic import Dyadic
from sympy.physics.vector.dyadic import Dyadic
from sympy.physics.vector.printing import VectorLatexPrinter
from sympy.physics.vector.printing import VectorPrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.vector.dyadic import Dyadic
from sympy.physics.vector.printing import VectorStrPrinter
from sympy.physics.vector.dyadic import Dyadic
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector import express
from sympy.physics.vector import time_derivative

__all__ = ['Vector']

class Vector(object):
    simp = False
    __truediv__ = __div__
    _sympystr = __str__
    _sympyrepr = _sympystr
    __repr__ = __str__
    __radd__ = __add__
    __rand__ = __and__
    __rmul__ = __mul__
    dot.__doc__ = __and__.__doc__
    cross.__doc__ = __xor__.__doc__
    outer.__doc__ = __or__.__doc__
    def __add__(self, other):
        """The add operator for Vector. """
        if other == 0:
            return self
        other = _check_vector(other)
        return Vector(self.args + other.args)