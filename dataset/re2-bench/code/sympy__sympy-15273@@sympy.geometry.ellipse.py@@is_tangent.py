from __future__ import division, print_function
from sympy import Expr, Eq
from sympy.core import S, pi, sympify
from sympy.core.logic import fuzzy_bool
from sympy.core.numbers import Rational, oo
from sympy.core.compatibility import ordered
from sympy.core.symbol import Dummy, _uniquely_named_symbol, _symbol
from sympy.simplify import simplify, trigsimp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.elliptic_integrals import elliptic_e
from sympy.geometry.exceptions import GeometryError
from sympy.geometry.line import Ray2D, Segment2D, Line2D, LinearEntity3D
from sympy.polys import DomainError, Poly, PolynomialError
from sympy.polys.polyutils import _not_a_coeff, _nsort
from sympy.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import filldedent, func_name
from .entity import GeometryEntity, GeometrySet
from .point import Point, Point2D, Point3D
from .line import Line, LinearEntity
from .util import idiff
import random
from .polygon import Polygon
from sympy.core.evalf import N
from sympy import sin, cos, Rational
from sympy.geometry.util import find
from .polygon import Triangle



class Ellipse(GeometrySet):
    def __contains__(self, o):
        if isinstance(o, Point):
            x = Dummy('x', real=True)
            y = Dummy('y', real=True)

            res = self.equation(x, y).subs({x: o.x, y: o.y})
            return trigsimp(simplify(res)) is S.Zero
        elif isinstance(o, Ellipse):
            return self == o
        return False
    def __eq__(self, o):
        """Is the other GeometryEntity the same as this ellipse?"""
        return isinstance(o, Ellipse) and (self.center == o.center and
                                           self.hradius == o.hradius and
                                           self.vradius == o.vradius)
    @property
    def center(self):
        """The center of the ellipse.

        Returns
        =======

        center : number

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.center
        Point2D(0, 0)

        """
        return self.args[0]
    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        -----
        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Ellipse, S
        >>> from sympy.abc import t
        >>> e = Ellipse((0, 0), 3, 2)
        >>> e.encloses_point((0, 0))
        True
        >>> e.encloses_point(e.arbitrary_point(t).subs(t, S.Half))
        False
        >>> e.encloses_point((4, 0))
        False

        """
        p = Point(p, dim=2)
        if p in self:
            return False

        if len(self.foci) == 2:
            # if the combined distance from the foci to p (h1 + h2) is less
            # than the combined distance from the foci to the minor axis
            # (which is the same as the major axis length) then p is inside
            # the ellipse
            h1, h2 = [f.distance(p) for f in self.foci]
            test = 2*self.major - (h1 + h2)
        else:
            test = self.radius - self.center.distance(p)

        return fuzzy_bool(test.is_positive)
    @property
    def foci(self):
        """The foci of the ellipse.

        Notes
        -----
        The foci can only be calculated if the major/minor axes are known.

        Raises
        ======

        ValueError
            When the major and minor axis cannot be determined.

        See Also
        ========

        sympy.geometry.point.Point
        focus_distance : Returns the distance between focus and center

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.foci
        (Point2D(-2*sqrt(2), 0), Point2D(2*sqrt(2), 0))

        """
        c = self.center
        hr, vr = self.hradius, self.vradius
        if hr == vr:
            return (c, c)

        # calculate focus distance manually, since focus_distance calls this
        # routine
        fd = sqrt(self.major**2 - self.minor**2)
        if hr == self.minor:
            # foci on the y-axis
            return (c + Point(0, -fd), c + Point(0, fd))
        elif hr == self.major:
            # foci on the x-axis
            return (c + Point(-fd, 0), c + Point(fd, 0))
    @property
    def hradius(self):
        """The horizontal radius of the ellipse.

        Returns
        =======

        hradius : number

        See Also
        ========

        vradius, major, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.hradius
        3

        """
        return self.args[1]
    def intersection(self, o):
        """The intersection of this ellipse and another geometrical entity
        `o`.

        Parameters
        ==========

        o : GeometryEntity

        Returns
        =======

        intersection : list of GeometryEntity objects

        Notes
        -----
        Currently supports intersections with Point, Line, Segment, Ray,
        Circle and Ellipse types.

        See Also
        ========

        sympy.geometry.entity.GeometryEntity

        Examples
        ========

        >>> from sympy import Ellipse, Point, Line, sqrt
        >>> e = Ellipse(Point(0, 0), 5, 7)
        >>> e.intersection(Point(0, 0))
        []
        >>> e.intersection(Point(5, 0))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(0,0), Point(0, 1)))
        [Point2D(0, -7), Point2D(0, 7)]
        >>> e.intersection(Line(Point(5,0), Point(5, 1)))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(6,0), Point(6, 1)))
        []
        >>> e = Ellipse(Point(-1, 0), 4, 3)
        >>> e.intersection(Ellipse(Point(1, 0), 4, 3))
        [Point2D(0, -3*sqrt(15)/4), Point2D(0, 3*sqrt(15)/4)]
        >>> e.intersection(Ellipse(Point(5, 0), 4, 3))
        [Point2D(2, -3*sqrt(7)/4), Point2D(2, 3*sqrt(7)/4)]
        >>> e.intersection(Ellipse(Point(100500, 0), 4, 3))
        []
        >>> e.intersection(Ellipse(Point(0, 0), 3, 4))
        [Point2D(3, 0), Point2D(-363/175, -48*sqrt(111)/175), Point2D(-363/175, 48*sqrt(111)/175)]
        >>> e.intersection(Ellipse(Point(-1, 0), 3, 4))
        [Point2D(-17/5, -12/5), Point2D(-17/5, 12/5), Point2D(7/5, -12/5), Point2D(7/5, 12/5)]
        """
        # TODO: Replace solve with nonlinsolve, when nonlinsolve will be able to solve in real domain
        x = Dummy('x', real=True)
        y = Dummy('y', real=True)

        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []

        elif isinstance(o, (Segment2D, Ray2D)):
            ellipse_equation = self.equation(x, y)
            result = solve([ellipse_equation, Line(o.points[0], o.points[1]).equation(x, y)], [x, y])
            return list(ordered([Point(i) for i in result if i in o]))

        elif isinstance(o, Polygon):
            return o.intersection(self)

        elif isinstance(o, (Ellipse, Line2D)):
            if o == self:
                return self
            else:
                ellipse_equation = self.equation(x, y)
                return list(ordered([Point(i) for i in solve([ellipse_equation, o.equation(x, y)], [x, y])]))
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Intersection not handled for %s' % func_name(o))
    def is_tangent(self, o):
        """Is `o` tangent to the ellipse?

        Parameters
        ==========

        o : GeometryEntity
            An Ellipse, LinearEntity or Polygon

        Raises
        ======

        NotImplementedError
            When the wrong type of argument is supplied.

        Returns
        =======

        is_tangent: boolean
            True if o is tangent to the ellipse, False otherwise.

        See Also
        ========

        tangent_lines

        Examples
        ========

        >>> from sympy import Point, Ellipse, Line
        >>> p0, p1, p2 = Point(0, 0), Point(3, 0), Point(3, 3)
        >>> e1 = Ellipse(p0, 3, 2)
        >>> l1 = Line(p1, p2)
        >>> e1.is_tangent(l1)
        True

        """
        if isinstance(o, Point2D):
            return False
        elif isinstance(o, Ellipse):
            intersect = self.intersection(o)
            if isinstance(intersect, Ellipse):
                return True
            elif intersect:
                return all((self.tangent_lines(i)[0]).equals((o.tangent_lines(i)[0])) for i in intersect)
            else:
                return False
        elif isinstance(o, Line2D):
            return len(self.intersection(o)) == 1
        elif isinstance(o, Ray2D):
            intersect = self.intersection(o)
            if len(intersect) == 1:
                return intersect[0] != o.source and not self.encloses_point(o.source)
            else:
                return False
        elif isinstance(o, (Segment2D, Polygon)):
            all_tangents = False
            segments = o.sides if isinstance(o, Polygon) else [o]
            for segment in segments:
                intersect = self.intersection(segment)
                if len(intersect) == 1:
                    if not any(intersect[0] in i for i in segment.points) \
                        and all(not self.encloses_point(i) for i in segment.points):
                        all_tangents = True
                        continue
                    else:
                        return False
                else:
                    return all_tangents
            return all_tangents
        elif isinstance(o, (LinearEntity3D, Point3D)):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Is_tangent not handled for %s' % func_name(o))
    @property
    def major(self):
        """Longer axis of the ellipse (if it can be determined) else hradius.

        Returns
        =======

        major : number or expression

        See Also
        ========

        hradius, vradius, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.major
        3

        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).major
        a
        >>> Ellipse(p1, b, a).major
        b

        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).major
        m + 1

        """
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        a, b = ab
        o = b - a < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.hradius