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
from .line import Line, LinearEntity, Segment
from .util import idiff
import random
from .polygon import Polygon
from sympy.core.evalf import N
from sympy import sin, cos, Rational
from sympy.geometry.util import find
from .polygon import Triangle



class Ellipse(GeometrySet):
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
    def equation(self, x='x', y='y', _slope=None):
        """
        Returns the equation of an ellipse aligned with the x and y axes;
        when slope is given, the equation returned corresponds to an ellipse
        with a major axis having that slope.

        Parameters
        ==========

        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.
        _slope : Expr, optional
                The slope of the major axis. Ignored when 'None'.

        Returns
        =======

        equation : sympy expression

        See Also
        ========

        arbitrary_point : Returns parameterized point on ellipse

        Examples
        ========

        >>> from sympy import Point, Ellipse, pi
        >>> from sympy.abc import x, y
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> eq1 = e1.equation(x, y); eq1
        y**2/4 + (x/3 - 1/3)**2 - 1
        >>> eq2 = e1.equation(x, y, _slope=1); eq2
        (-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1

        A point on e1 satisfies eq1. Let's use one on the x-axis:

        >>> p1 = e1.center + Point(e1.major, 0)
        >>> assert eq1.subs(x, p1.x).subs(y, p1.y) == 0

        When rotated the same as the rotated ellipse, about the center
        point of the ellipse, it will satisfy the rotated ellipse's
        equation, too:

        >>> r1 = p1.rotate(pi/4, e1.center)
        >>> assert eq2.subs(x, r1.x).subs(y, r1.y) == 0

        References
        ==========

        .. [1] https://math.stackexchange.com/questions/108270/what-is-the-equation-of-an-ellipse-that-is-not-aligned-with-the-axis
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Equation_of_a_shifted_ellipse

        """

        x = _symbol(x, real=True)
        y = _symbol(y, real=True)

        dx = x - self.center.x
        dy = y - self.center.y

        if _slope is not None:
            L = (dy - _slope*dx)**2
            l = (_slope*dy + dx)**2
            h = 1 + _slope**2
            b = h*self.major**2
            a = h*self.minor**2
            return l/b + L/a - 1

        else:
            t1 = (dx/self.hradius)**2
            t2 = (dy/self.vradius)**2
            return t1 + t2 - 1