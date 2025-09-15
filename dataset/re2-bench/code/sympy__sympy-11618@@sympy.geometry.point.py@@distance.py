from __future__ import division, print_function
from sympy.core import S, sympify
from sympy.core.compatibility import iterable
from sympy.core.containers import Tuple
from sympy.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.matrices import Matrix
from sympy.core.numbers import Float
from sympy.core.evaluate import global_evaluate
from sympy.core.add import Add
from .entity import GeometryEntity
from sympy import cos, sin, Point
from sympy.geometry.plane import Plane
from sympy.matrices.expressions import Transpose
from .ellipse import Circle



class Point(GeometryEntity):
    is_Point = True
    n = evalf
    __truediv__ = __div__
    def distance(self, p):
        """The Euclidean distance from self to point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        distance : number or symbolic expression.

        See Also
        ========

        sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy.geometry import Point
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> p1.distance(p2)
        5

        >>> from sympy.abc import x, y
        >>> p3 = Point(x, y)
        >>> p3.distance(Point(0, 0))
        sqrt(x**2 + y**2)

        """
        if type(p) is not type(self):
            if len(p) == len(self):
                return sqrt(sum([(a - b)**2 for a, b in zip(
                    self.args, p.args if isinstance(p, Point) else p)]))
            else:
                p1 = [0] * max(len(p), len(self))
                p2 = p.args if len(p.args) > len(self.args) else self.args

                for i in range(min(len(p), len(self))):
                    p1[i] = p.args[i] if len(p) < len(self) else self.args[i]

                return sqrt(sum([(a - b)**2 for a, b in zip(
                    p1, p2)]))

        return sqrt(sum([(a - b)**2 for a, b in zip(
            self.args, p.args if isinstance(p, Point) else p)]))
    def __len__(self):
        return len(self.args)