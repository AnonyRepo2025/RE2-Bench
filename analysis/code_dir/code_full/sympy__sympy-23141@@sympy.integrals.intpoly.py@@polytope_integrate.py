def hyperplane_parameters(poly, vertices=None):
    if isinstance(poly, Polygon):
        vertices = list(poly.vertices) + [poly.vertices[0]]
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

def vertices(self):
    return self.args

def args(self) -> 'Tuple[Basic, ...]':
    return self._args

def __getitem__(self, key):
    return self.args[key]

def __sub__(self, other):
    if global_parameters.evaluate:
        if isinstance(other, int):
            return Integer(self.p - other)
        elif isinstance(other, Integer):
            return Integer(self.p - other.p)
        elif isinstance(other, Rational):
            return Rational(self.p * other.q - other.p, other.q, 1)
        return Rational.__sub__(self, other)
    return Rational.__sub__(self, other)

def __new__(cls, i):
    if isinstance(i, str):
        i = i.replace(' ', '')
    try:
        ival = int(i)
    except TypeError:
        raise TypeError('Argument of Integer should be of numeric type, got %s.' % i)
    if ival == 1:
        return S.One
    if ival == -1:
        return S.NegativeOne
    if ival == 0:
        return S.Zero
    obj = Expr.__new__(cls)
    obj.p = ival
    return obj

def __mul__(self, other):
    if global_parameters.evaluate:
        if isinstance(other, int):
            return Integer(self.p * other)
        elif isinstance(other, Integer):
            return Integer(self.p * other.p)
        elif isinstance(other, Rational):
            return Rational(self.p * other.p, other.q, igcd(self.p, other.q))
        return Rational.__mul__(self, other)
    return Rational.__mul__(self, other)

def gcd_list(seq, *gens, **args):
    seq = sympify(seq)

    def try_non_polynomial_gcd(seq):
        if not gens and (not args):
            domain, numbers = construct_domain(seq)
            if not numbers:
                return domain.zero
            elif domain.is_Numerical:
                result, numbers = (numbers[0], numbers[1:])
                for number in numbers:
                    result = domain.gcd(result, number)
                    if domain.is_one(result):
                        break
                return domain.to_sympy(result)
        return None
    result = try_non_polynomial_gcd(seq)
    if result is not None:
        return result
    options.allowed_flags(args, ['polys'])
    try:
        polys, opt = parallel_poly_from_expr(seq, *gens, **args)
        if len(seq) > 1 and all((elt.is_algebraic and elt.is_irrational for elt in seq)):
            a = seq[-1]
            lst = [(a / elt).ratsimp() for elt in seq[:-1]]
            if all((frc.is_rational for frc in lst)):
                lc = 1
                for frc in lst:
                    lc = lcm(lc, frc.as_numer_denom()[0])
                return abs(a / lc)
    except PolificationFailed as exc:
        result = try_non_polynomial_gcd(exc.exprs)
        if result is not None:
            return result
        else:
            raise ComputationFailed('gcd_list', len(seq), exc)
    if not polys:
        if not opt.polys:
            return S.Zero
        else:
            return Poly(0, opt=opt)
    result, polys = (polys[0], polys[1:])
    for poly in polys:
        result = result.gcd(poly)
        if result.is_one:
            break
    if not opt.polys:
        return result.as_expr()
    else:
        return result

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    is_sympy = getattr(a, '__sympy__', None)
    if is_sympy is True:
        return a
    elif is_sympy is not None:
        if not strict:
            return a
        else:
            raise SympifyError(a)
    if isinstance(a, CantSympify):
        raise SympifyError(a)
    cls = getattr(a, '__class__', None)
    for superclass in getmro(cls):
        conv = _external_converter.get(superclass)
        if conv is None:
            conv = _sympy_converter.get(superclass)
        if conv is not None:
            return conv(a)
    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a
    if evaluate is None:
        evaluate = global_parameters.evaluate
    if _is_numpy_instance(a):
        import numpy as np
        if np.isscalar(a):
            return _convert_numpy_types(a, locals=locals, convert_xor=convert_xor, strict=strict, rational=rational, evaluate=evaluate)
    _sympy_ = getattr(a, '_sympy_', None)
    if _sympy_ is not None:
        try:
            return a._sympy_()
        except AttributeError:
            pass
    if not strict:
        flat = getattr(a, 'flat', None)
        if flat is not None:
            shape = getattr(a, 'shape', None)
            if shape is not None:
                from sympy.tensor.array import Array
                return Array(a.flat, a.shape)
    if not isinstance(a, str):
        if _is_numpy_instance(a):
            import numpy as np
            assert not isinstance(a, np.number)
            if isinstance(a, np.ndarray):
                if a.ndim == 0:
                    try:
                        return sympify(a.item(), locals=locals, convert_xor=convert_xor, strict=strict, rational=rational, evaluate=evaluate)
                    except SympifyError:
                        pass
        else:
            for coerce in (float, int):
                try:
                    return sympify(coerce(a))
                except (TypeError, ValueError, AttributeError, SympifyError):
                    continue
    if strict:
        raise SympifyError(a)
    if iterable(a):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor, rational=rational, evaluate=evaluate) for x in a])
        except TypeError:
            pass
    if not isinstance(a, str):
        try:
            a = str(a)
        except Exception as exc:
            raise SympifyError(a, exc)
        sympy_deprecation_warning(f'\nThe string fallback in sympify() is deprecated.\n\nTo explicitly convert the string form of an object, use\nsympify(str(obj)). To add define sympify behavior on custom\nobjects, use sympy.core.sympify.converter or define obj._sympy_\n(see the sympify() docstring).\n\nsympify() performed the string fallback resulting in the following string:\n\n{a!r}\n            ', deprecated_since_version='1.6', active_deprecations_target='deprecated-sympify-string-fallback')
    from sympy.parsing.sympy_parser import parse_expr, TokenError, standard_transformations
    from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
    from sympy.parsing.sympy_parser import rationalize as t_rationalize
    transformations = standard_transformations
    if rational:
        transformations += (t_rationalize,)
    if convert_xor:
        transformations += (t_convert_xor,)
    try:
        a = a.replace('\n', '')
        expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
    except (TokenError, SyntaxError) as exc:
        raise SympifyError('could not parse %r' % a, exc)
    return expr

def _is_numpy_instance(a):
    return any((type_.__module__ == 'numpy' for type_ in type(a).__mro__))

def iterable(i, exclude=(str, dict, NotIterable)):
    if hasattr(i, '_iterable'):
        return i._iterable
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True

def try_non_polynomial_gcd(seq):
    if not gens and (not args):
        domain, numbers = construct_domain(seq)
        if not numbers:
            return domain.zero
        elif domain.is_Numerical:
            result, numbers = (numbers[0], numbers[1:])
            for number in numbers:
                result = domain.gcd(result, number)
                if domain.is_one(result):
                    break
            return domain.to_sympy(result)
    return None

def construct_domain(obj, **args):
    opt = build_options(args)
    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            if not obj:
                monoms, coeffs = ([], [])
            else:
                monoms, coeffs = list(zip(*list(obj.items())))
        else:
            coeffs = obj
    else:
        coeffs = [obj]
    coeffs = list(map(sympify, coeffs))
    result = _construct_simple(coeffs, opt)
    if result is not None:
        if result is not False:
            domain, coeffs = result
        else:
            domain, coeffs = _construct_expression(coeffs, opt)
    else:
        if opt.composite is False:
            result = None
        else:
            result = _construct_composite(coeffs, opt)
        if result is not None:
            domain, coeffs = result
        else:
            domain, coeffs = _construct_expression(coeffs, opt)
    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            return (domain, dict(list(zip(monoms, coeffs))))
        else:
            return (domain, coeffs)
    else:
        return (domain, coeffs[0])

def build_options(gens, args=None):
    if args is None:
        gens, args = ((), gens)
    if len(args) != 1 or 'opt' not in args or gens:
        return Options(gens, args)
    else:
        return args['opt']

def __init__(self, gens, args, flags=None, strict=False):
    dict.__init__(self)
    if gens and args.get('gens', ()):
        raise OptionError("both '*gens' and keyword argument 'gens' supplied")
    elif gens:
        args = dict(args)
        args['gens'] = gens
    defaults = args.pop('defaults', {})

    def preprocess_options(args):
        for option, value in args.items():
            try:
                cls = self.__options__[option]
            except KeyError:
                raise OptionError("'%s' is not a valid option" % option)
            if issubclass(cls, Flag):
                if flags is None or option not in flags:
                    if strict:
                        raise OptionError("'%s' flag is not allowed in this context" % option)
            if value is not None:
                self[option] = cls.preprocess(value)
    preprocess_options(args)
    for key, value in dict(defaults).items():
        if key in self:
            del defaults[key]
        else:
            for option in self.keys():
                cls = self.__options__[option]
                if key in cls.excludes:
                    del defaults[key]
                    break
    preprocess_options(defaults)
    for option in self.keys():
        cls = self.__options__[option]
        for require_option in cls.requires:
            if self.get(require_option) is None:
                raise OptionError("'%s' option is only allowed together with '%s'" % (option, require_option))
        for exclude_option in cls.excludes:
            if self.get(exclude_option) is not None:
                raise OptionError("'%s' option is not allowed together with '%s'" % (option, exclude_option))
    for option in self.__order__:
        self.__options__[option].postprocess(self)



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
            raise TypeError('clockwise=True works for only 2-PolytopeV-representation input')
    if isinstance(poly, Polygon):
        hp_params = hyperplane_parameters(poly)
        facets = poly.sides
    elif len(poly[0]) == 2:
        plen = len(poly)
        if len(poly[0][0]) == 2:
            intersections = [intersection(poly[(i - 1) % plen], poly[i], 'plane2D') for i in range(0, plen)]
            hp_params = poly
            lints = len(intersections)
            facets = [Segment2D(intersections[i], intersections[(i + 1) % lints]) for i in range(0, lints)]
        else:
            raise NotImplementedError('Integration for H-representation 3Dcase not implemented yet.')
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
                if len(_) == 1 and (not _.popitem()[0]):
                    f_expr.append(e)
                elif Poly(e).total_degree() <= max_degree:
                    f_expr.append(e)
            expr = f_expr
        if not isinstance(expr, list) and expr is not None:
            raise TypeError('Input polynomials must be list of expressions')
        if len(hp_params[0][0]) == 3:
            result_dict = main_integrate3d(0, facets, vertices, hp_params, max_degree)
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