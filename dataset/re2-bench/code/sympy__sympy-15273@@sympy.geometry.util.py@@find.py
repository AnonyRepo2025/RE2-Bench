from __future__ import division, print_function
from sympy import Function, Symbol, solve
from sympy.core.compatibility import (
    is_sequence, range, string_types, ordered)
from sympy.core.containers import OrderedSet
from .point import Point, Point2D
from sympy.geometry.line import LinearEntity3D
from sympy.geometry.point import Point3D
from sympy.geometry.plane import Plane
from .exceptions import GeometryError
from sympy.geometry import Polygon, Segment, Point
from collections import deque
from math import hypot, sqrt as _sqrt
from sympy.functions.elementary.miscellaneous import sqrt
from .entity import GeometryEntity
from .point import Point
from .line import Segment
from .polygon import Polygon
from math import hypot, sqrt as _sqrt
from .entity import GeometryEntity
from .point import Point



def find(x, equation):
    """
    Checks whether the parameter 'x' is present in 'equation' or not.
    If it is present then it returns the passed parameter 'x' as a free
    symbol, else, it returns a ValueError.
    """

    free = equation.free_symbols
    xs = [i for i in free if (i.name if type(x) is str else i) == x]
    if not xs:
        raise ValueError('could not find %s' % x)
    if len(xs) != 1:
        raise ValueError('ambiguous %s' % x)
    return xs[0]
