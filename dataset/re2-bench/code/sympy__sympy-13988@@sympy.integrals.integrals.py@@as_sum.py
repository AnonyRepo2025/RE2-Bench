from __future__ import print_function, division
from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.compatibility import is_sequence, range
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import diff
from sympy.core.mul import Mul
from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.core.sympify import sympify
from sympy.integrals.manualintegrate import manualintegrate
from sympy.integrals.trigonometry import trigintegrate
from sympy.integrals.meijerint import meijerint_definite, meijerint_indefinite
from sympy.matrices import MatrixBase
from sympy.utilities.misc import filldedent
from sympy.polys import Poly, PolynomialError
from sympy.functions import Piecewise, sqrt, sign, piecewise_fold
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.series import limit
from sympy.series.order import Order
from sympy.series.formal import FormalPowerSeries
from sympy.geometry import Curve
from sympy.solvers.solvers import solve, posify
from sympy.integrals.deltafunctions import deltaintegrate
from sympy.integrals.singularityfunctions import singularityintegrate
from sympy.integrals.heurisch import heurisch, heurisch_wrapper
from sympy.integrals.rationaltools import ratint
from sympy.integrals.risch import risch_integrate
from sympy.concrete.summations import Sum
import sage.all as sage
from sympy.integrals.risch import NonElementaryIntegral
from sympy.integrals.meijerint import _debug
from sympy.integrals.meijerint import _debug



class Integral(AddWithLimits):
    __slots__ = ['is_commutative']
    def as_sum(self, n=None, method="midpoint", evaluate=True):
        """
        Approximates a definite integral by a sum.

        Arguments
        ---------
        n
            The number of subintervals to use, optional.
        method
            One of: 'left', 'right', 'midpoint', 'trapezoid'.
        evaluate
            If False, returns an unevaluated Sum expression. The default
            is True, evaluate the sum.

        These methods of approximate integration are described in [1].

        [1] https://en.wikipedia.org/wiki/Riemann_sum#Methods

        Examples
        ========

        >>> from sympy import sin, sqrt
        >>> from sympy.abc import x, n
        >>> from sympy.integrals import Integral
        >>> e = Integral(sin(x), (x, 3, 7))
        >>> e
        Integral(sin(x), (x, 3, 7))

        For demonstration purposes, this interval will only be split into 2
        regions, bounded by [3, 5] and [5, 7].

        The left-hand rule uses function evaluations at the left of each
        interval:

        >>> e.as_sum(2, 'left')
        2*sin(5) + 2*sin(3)

        The midpoint rule uses evaluations at the center of each interval:

        >>> e.as_sum(2, 'midpoint')
        2*sin(4) + 2*sin(6)

        The right-hand rule uses function evaluations at the right of each
        interval:

        >>> e.as_sum(2, 'right')
        2*sin(5) + 2*sin(7)

        The trapezoid rule uses function evaluations on both sides of the
        intervals. This is equivalent to taking the average of the left and
        right hand rule results:

        >>> e.as_sum(2, 'trapezoid')
        2*sin(5) + sin(3) + sin(7)
        >>> (e.as_sum(2, 'left') + e.as_sum(2, 'right'))/2 == _
        True

        Here, the discontinuity at x = 0 can be avoided by using the
        midpoint or right-hand method:

        >>> e = Integral(1/sqrt(x), (x, 0, 1))
        >>> e.as_sum(5).n(4)
        1.730
        >>> e.as_sum(10).n(4)
        1.809
        >>> e.doit().n(4)  # the actual value is 2
        2.000

        The left- or trapezoid method will encounter the discontinuity and
        return infinity:

        >>> e.as_sum(5, 'left')
        zoo

        The number of intervals can be symbolic. If omitted, a dummy symbol
        will be used for it.
        >>> e = Integral(x**2, (x, 0, 2))
        >>> e.as_sum(n, 'right').expand()
        8/3 + 4/n + 4/(3*n**2)

        This shows that the midpoint rule is more accurate, as its error
        term decays as the square of n:
        >>> e.as_sum(method='midpoint').expand()
        8/3 - 2/(3*_n**2)

        A symbolic sum is returned with evaluate=False:
        >>> e.as_sum(n, 'midpoint', evaluate=False)
        2*Sum((2*_k/n - 1/n)**2, (_k, 1, n))/n

        See Also
        ========

        Integral.doit : Perform the integration using any hints
        """

        from sympy.concrete.summations import Sum
        limits = self.limits
        if len(limits) > 1:
            raise NotImplementedError(
                "Multidimensional midpoint rule not implemented yet")
        else:
            limit = limits[0]
            if (len(limit) != 3 or limit[1].is_finite is False or
                limit[2].is_finite is False):
                raise ValueError("Expecting a definite integral over "
                                  "a finite interval.")
        if n is None:
            n = Dummy('n', integer=True, positive=True)
        else:
            n = sympify(n)
        if (n.is_positive is False or n.is_integer is False or
            n.is_finite is False):
            raise ValueError("n must be a positive integer, got %s" % n)
        x, a, b = limit
        dx = (b - a)/n
        k = Dummy('k', integer=True, positive=True)
        f = self.function

        if method == "left":
            result = dx*Sum(f.subs(x, a + (k-1)*dx), (k, 1, n))
        elif method == "right":
            result = dx*Sum(f.subs(x, a + k*dx), (k, 1, n))
        elif method == "midpoint":
            result = dx*Sum(f.subs(x, a + k*dx - dx/2), (k, 1, n))
        elif method == "trapezoid":
            result = dx*((f.subs(x, a) + f.subs(x, b))/2 +
                Sum(f.subs(x, a + k*dx), (k, 1, n - 1)))
        else:
            raise ValueError("Unknown method %s" % method)
        return result.doit() if evaluate else result