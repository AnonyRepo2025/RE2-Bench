from __future__ import division, print_function
from sympy import simplify
from sympy.core import Dummy, Rational, S, Symbol
from sympy.core.symbol import _symbol
from sympy.core.compatibility import is_sequence
from sympy.functions.elementary.trigonometric import cos, sin, acos, asin, sqrt
from sympy.matrices import Matrix
from sympy.polys.polytools import cancel
from sympy.solvers import solve, linsolve
from sympy.utilities.iterables import uniq
from sympy.utilities.misc import filldedent, func_name
from .entity import GeometryEntity
from .point import Point, Point3D
from .line import Line, Ray, Segment, Line3D, LinearEntity3D, Ray3D, Segment3D
from sympy.geometry.line import LinearEntity, LinearEntity3D
from sympy.geometry.line import LinearEntity3D
from sympy.geometry.line import LinearEntity3D
from sympy.geometry.line import LinearEntity, LinearEntity3D
from sympy.geometry.line import LinearEntity3D
from sympy.geometry.line import LinearEntity3D
from sympy.geometry.line import LinearEntity, LinearEntity3D
import random
from sympy.geometry.point import Point
from sympy.core.symbol import Dummy
from sympy.solvers.solvers import solve



class Plane(GeometryEntity):
    def distance(self, o):
        """Distance between the plane and another geometric entity.

        Parameters
        ==========

        Point3D, LinearEntity3D, Plane.

        Returns
        =======

        distance

        Notes
        =====

        This method accepts only 3D entities as it's parameter, but if you want
        to calculate the distance between a 2D entity and a plane you should
        first convert to a 3D entity by projecting onto a desired plane and
        then proceed to calculate the distance.

        Examples
        ========

        >>> from sympy import Point, Point3D, Line, Line3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))
        >>> b = Point3D(1, 2, 3)
        >>> a.distance(b)
        sqrt(3)
        >>> c = Line3D(Point3D(2, 3, 1), Point3D(1, 2, 2))
        >>> a.distance(c)
        0

        """
        from sympy.geometry.line import LinearEntity3D
        if self.intersection(o) != []:
            return S.Zero

        if isinstance(o, (Segment3D, Ray3D)):
            a, b = o.p1, o.p2
            pi, = self.intersection(Line3D(a, b))
            if pi in o:
                return self.distance(pi)
            elif a in Segment3D(pi, b):
                return self.distance(a)
            else:
                assert isinstance(o, Segment3D) is True
                return self.distance(b)

        # following code handles `Point3D`, `LinearEntity3D`, `Plane`
        a = o if isinstance(o, Point3D) else o.p1
        n = Point3D(self.normal_vector).unit
        d = (a - self.p1).dot(n)
        return abs(d)