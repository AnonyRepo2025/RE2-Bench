def hyperplane_parameters(poly, vertices=None):
    if isinstance(poly, Polygon):
        vertices = list(poly.vertices) + [poly.vertices[0]]  # Close the polygon
        params = [None] * (len(vertices) - 1)

        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            a1 = v1[1] - v2[1]
            a2 = v2[0] - v1[0]
            b = v2[0] * v1[1] - v2[1] * v1[0]

            factor = gcd_list([a1, a2, b])

            b = S(b) / factor
            a = (S(a1) / factor, S(a2) / factor)
            params[i] = (a, b)
    else:
        params = [None] * len(poly)
        for i, polygon in enumerate(poly):
            v1, v2, v3 = [vertices[vertex] for vertex in polygon[:3]]
            normal = cross_product(v1, v2, v3)
            b = sum([normal[j] * v1[j] for j in range(0, 3)])
            fac = gcd_list(normal)
            if fac.is_zero:
                fac = 1
            normal = [j / fac for j in normal]
            b = b / fac
            params[i] = (normal, b)
    return params

def main_integrate3d(expr, facets, vertices, hp_params, max_degree=None):
    result = {}
    dims = (x, y, z)
    dim_length = len(dims)
    if max_degree:
        grad_terms = gradient_terms(max_degree, 3)
        flat_list = [term for z_terms in grad_terms
                     for x_term in z_terms
                     for term in x_term]

        for term in flat_list:
            result[term[0]] = 0

        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            x0 = vertices[facets[facet_count][0]]

            for i, monom in enumerate(flat_list):
                #  Every monomial is a tuple :
                #  (term, x_degree, y_degree, z_degree, value over boundary)
                expr, x_d, y_d, z_d, z_index, y_index, x_index, _ = monom
                degree = x_d + y_d + z_d
                if b.is_zero:
                    value_over_face = S.Zero
                else:
                    value_over_face = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, expr, degree, dims,
                                                      x_index, y_index,
                                                      z_index, x0, grad_terms,
                                                      i, vertices, hp)
                monom[7] = value_over_face
                result[expr] += value_over_face * \
                    (b / norm(a)) / (dim_length + x_d + y_d + z_d)
        return result
    else:
        integral_value = S.Zero
        polynomials = decompose(expr)
        for deg in polynomials:
            poly_contribute = S.Zero
            facet_count = 0
            for i, facet in enumerate(facets):
                hp = hp_params[i]
                if hp[1].is_zero:
                    continue
                pi = polygon_integrate(facet, hp, i, facets, vertices, expr, deg)
                poly_contribute += pi *\
                    (hp[1] / norm(tuple(hp[0])))
                facet_count += 1
            poly_contribute /= (dim_length + deg)
            integral_value += poly_contribute
    return integral_value

def main_integrate(expr, facets, hp_params, max_degree=None):
    dims = (x, y)
    dim_length = len(dims)
    result = {}

    if max_degree:
        grad_terms = [[0, 0, 0, 0]] + gradient_terms(max_degree)

        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            x0 = facets[facet_count].points[0]

            for i, monom in enumerate(grad_terms):
                #  Every monomial is a tuple :
                #  (term, x_degree, y_degree, value over boundary)
                m, x_d, y_d, _ = monom
                value = result.get(m, None)
                degree = S.Zero
                if b.is_zero:
                    value_over_boundary = S.Zero
                else:
                    degree = x_d + y_d
                    value_over_boundary = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, m, degree, dims, x_d,
                                                      y_d, max_degree, x0,
                                                      grad_terms, i)
                monom[3] = value_over_boundary
                if value is not None:
                    result[m] += value_over_boundary * \
                                        (b / norm(a)) / (dim_length + degree)
                else:
                    result[m] = value_over_boundary * \
                                (b / norm(a)) / (dim_length + degree)
        return result
    else:
        if not isinstance(expr, list):
            polynomials = decompose(expr)
            return _polynomial_integrate(polynomials, facets, hp_params)
        else:
            return {e: _polynomial_integrate(decompose(e), facets, hp_params) for e in expr}


def decompose(expr, separate=False):
    poly_dict = {}

    if isinstance(expr, Expr) and not expr.is_number:
        if expr.is_Symbol:
            poly_dict[1] = expr
        elif expr.is_Add:
            symbols = expr.atoms(Symbol)
            degrees = [(sum(degree_list(monom, *symbols)), monom)
                       for monom in expr.args]
            if separate:
                return {monom[1] for monom in degrees}
            else:
                for monom in degrees:
                    degree, term = monom
                    if poly_dict.get(degree):
                        poly_dict[degree] += term
                    else:
                        poly_dict[degree] = term
        elif expr.is_Pow:
            _, degree = expr.args
            poly_dict[degree] = expr
        else:  # Now expr can only be of `Mul` type
            degree = 0
            for term in expr.args:
                term_type = len(term.args)
                if term_type == 0 and term.is_Symbol:
                    degree += 1
                elif term_type == 2:
                    degree += term.args[1]
            poly_dict[degree] = expr
    else:
        poly_dict[0] = expr

    if separate:
        return set(poly_dict.values())
    return poly_dict
