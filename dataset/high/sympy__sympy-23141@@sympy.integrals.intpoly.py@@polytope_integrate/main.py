from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
from sympy.plotting.plot import Plot, List2DSeries
from sympy.plotting.plot import plot3d, plot



def polytope_integrate(poly, expr=None, *, clockwise=False, max_degree=None):
    if clockwise:
        if isinstance(poly, Polygon):
            poly = Polygon(*point_sort(poly.vertices), evaluate=False)
        else:
            raise TypeError("clockwise=True works for only 2-Polytope"
                            "V-representation input")

    if isinstance(poly, Polygon):
        hp_params = hyperplane_parameters(poly)
        facets = poly.sides
    elif len(poly[0]) == 2:
        plen = len(poly)
        if len(poly[0][0]) == 2:
            intersections = [intersection(poly[(i - 1) % plen], poly[i],
                                          "plane2D")
                             for i in range(0, plen)]
            hp_params = poly
            lints = len(intersections)
            facets = [Segment2D(intersections[i],
                                intersections[(i + 1) % lints])
                      for i in range(0, lints)]
        else:
            raise NotImplementedError("Integration for H-representation 3D"
                                      "case not implemented yet.")
    else:
        vertices = poly[0]
        facets = poly[1:]
        hp_params = hyperplane_parameters(facets, vertices)

        if max_degree is None:
            if expr is None:
                raise TypeError('Input expression must be a valid SymPy expression')
            return main_integrate3d(expr, facets, vertices, hp_params)

    if max_degree is not None:
        result = {}
        if expr is not None:
            f_expr = []
            for e in expr:
                _ = decompose(e)
                if len(_) == 1 and not _.popitem()[0]:
                    f_expr.append(e)
                elif Poly(e).total_degree() <= max_degree:
                    f_expr.append(e)
            expr = f_expr

        if not isinstance(expr, list) and expr is not None:
            raise TypeError('Input polynomials must be list of expressions')

        if len(hp_params[0][0]) == 3:
            result_dict = main_integrate3d(0, facets, vertices, hp_params,
                                           max_degree)
        else:
            result_dict = main_integrate(0, facets, hp_params, max_degree)

        if expr is None:
            return result_dict

        for poly in expr:
            poly = _sympify(poly)
            if poly not in result:
                if poly.is_zero:
                    result[S.Zero] = S.Zero
                    continue
                integral_value = S.Zero
                monoms = decompose(poly, separate=True)
                for monom in monoms:
                    monom = nsimplify(monom)
                    coeff, m = strip(monom)
                    integral_value += result_dict[m] * coeff
                result[poly] = integral_value
        return result

    if expr is None:
        raise TypeError('Input expression must be a valid SymPy expression')

    return main_integrate(expr, facets, hp_params)
