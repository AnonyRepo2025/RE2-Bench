from __future__ import print_function, division
from collections import defaultdict
from sympy import SYMPY_DEBUG
from sympy.core import expand_power_base, sympify, Add, S, Mul, Derivative, Pow, symbols, expand_mul
from sympy.core.add import _unevaluated_Add
from sympy.core.compatibility import iterable, ordered, default_sort_key
from sympy.core.parameters import global_parameters
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand
from sympy.core.mul import _keep_coeff, _unevaluated_Mul
from sympy.core.numbers import Rational
from sympy.functions import exp, sqrt, log
from sympy.functions.elementary.complexes import Abs
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.core.assumptions import assumptions
from sympy.utilities.iterables import sift
from sympy.core.symbol import Dummy, Wild
from sympy.simplify.simplify import signsimp
from sympy.core.mul import _mulsort
from sympy.simplify.simplify import nsimplify
from sympy.simplify.powsimp import powsimp, powdenest

expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand

def collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_term=True):
    """
    Collect additive terms of an expression.

    This function collects additive terms of an expression with respect
    to a list of expression up to powers with rational exponents. By the
    term symbol here are meant arbitrary expressions, which can contain
    powers, products, sums etc. In other words symbol is a pattern which
    will be searched for in the expression's terms.

    The input expression is not expanded by :func:`collect`, so user is
    expected to provide an expression is an appropriate form. This makes
    :func:`collect` more predictable as there is no magic happening behind the
    scenes. However, it is important to note, that powers of products are
    converted to products of powers using the :func:`~.expand_power_base`
    function.

    There are two possible types of output. First, if ``evaluate`` flag is
    set, this function will return an expression with collected terms or
    else it will return a dictionary with expressions up to rational powers
    as keys and collected coefficients as values.

    Examples
    ========

    >>> from sympy import S, collect, expand, factor, Wild
    >>> from sympy.abc import a, b, c, x, y, z

    This function can collect symbolic coefficients in polynomials or
    rational expressions. It will manage to find all integer or rational
    powers of collection variable::

        >>> collect(a*x**2 + b*x**2 + a*x - b*x + c, x)
        c + x**2*(a + b) + x*(a - b)

    The same result can be achieved in dictionary form::

        >>> d = collect(a*x**2 + b*x**2 + a*x - b*x + c, x, evaluate=False)
        >>> d[x**2]
        a + b
        >>> d[x]
        a - b
        >>> d[S.One]
        c

    You can also work with multivariate polynomials. However, remember that
    this function is greedy so it will care only about a single symbol at time,
    in specification order::

        >>> collect(x**2 + y*x**2 + x*y + y + a*y, [x, y])
        x**2*(y + 1) + x*y + y*(a + 1)

    Also more complicated expressions can be used as patterns::

        >>> from sympy import sin, log
        >>> collect(a*sin(2*x) + b*sin(2*x), sin(2*x))
        (a + b)*sin(2*x)

        >>> collect(a*x*log(x) + b*(x*log(x)), x*log(x))
        x*(a + b)*log(x)

    You can use wildcards in the pattern::

        >>> w = Wild('w1')
        >>> collect(a*x**y - b*x**y, w**y)
        x**y*(a - b)

    It is also possible to work with symbolic powers, although it has more
    complicated behavior, because in this case power's base and symbolic part
    of the exponent are treated as a single symbol::

        >>> collect(a*x**c + b*x**c, x)
        a*x**c + b*x**c
        >>> collect(a*x**c + b*x**c, x**c)
        x**c*(a + b)

    However if you incorporate rationals to the exponents, then you will get
    well known behavior::

        >>> collect(a*x**(2*c) + b*x**(2*c), x**c)
        x**(2*c)*(a + b)

    Note also that all previously stated facts about :func:`collect` function
    apply to the exponential function, so you can get::

        >>> from sympy import exp
        >>> collect(a*exp(2*x) + b*exp(2*x), exp(x))
        (a + b)*exp(2*x)

    If you are interested only in collecting specific powers of some symbols
    then set ``exact`` flag in arguments::

        >>> collect(a*x**7 + b*x**7, x, exact=True)
        a*x**7 + b*x**7
        >>> collect(a*x**7 + b*x**7, x**7, exact=True)
        x**7*(a + b)

    You can also apply this function to differential equations, where
    derivatives of arbitrary order can be collected. Note that if you
    collect with respect to a function or a derivative of a function, all
    derivatives of that function will also be collected. Use
    ``exact=True`` to prevent this from happening::

        >>> from sympy import Derivative as D, collect, Function
        >>> f = Function('f') (x)

        >>> collect(a*D(f,x) + b*D(f,x), D(f,x))
        (a + b)*Derivative(f(x), x)

        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), f)
        (a + b)*Derivative(f(x), (x, 2))

        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), D(f,x), exact=True)
        a*Derivative(f(x), (x, 2)) + b*Derivative(f(x), (x, 2))

        >>> collect(a*D(f,x) + b*D(f,x) + a*f + b*f, f)
        (a + b)*f(x) + (a + b)*Derivative(f(x), x)

    Or you can even match both derivative order and exponent at the same time::

        >>> collect(a*D(D(f,x),x)**2 + b*D(D(f,x),x)**2, D(f,x))
        (a + b)*Derivative(f(x), (x, 2))**2

    Finally, you can apply a function to each of the collected coefficients.
    For example you can factorize symbolic coefficients of polynomial::

        >>> f = expand((x + a + 1)**3)

        >>> collect(f, x, factor)
        x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + (a + 1)**3

    .. note:: Arguments are expected to be in expanded form, so you might have
              to call :func:`~.expand` prior to calling this function.

    See Also
    ========

    collect_const, collect_sqrt, rcollect
    """
    from sympy.core.assumptions import assumptions
    from sympy.utilities.iterables import sift
    from sympy.core.symbol import Dummy, Wild
    expr = sympify(expr)
    syms = [sympify(i) for i in (syms if iterable(syms) else [syms])]
    # replace syms[i] if it is not x, -x or has Wild symbols
    cond = lambda x: x.is_Symbol or (-x).is_Symbol or bool(
        x.atoms(Wild))
    _, nonsyms = sift(syms, cond, binary=True)
    if nonsyms:
        reps = dict(zip(nonsyms, [Dummy(**assumptions(i)) for i in nonsyms]))
        syms = [reps.get(s, s) for s in syms]
        rv = collect(expr.subs(reps), syms,
            func=func, evaluate=evaluate, exact=exact,
            distribute_order_term=distribute_order_term)
        urep = {v: k for k, v in reps.items()}
        if not isinstance(rv, dict):
            return rv.xreplace(urep)
        else:
            return {urep.get(k, k): v.xreplace(urep) for k, v in rv.items()}

    if evaluate is None:
        evaluate = global_parameters.evaluate

    def make_expression(terms):
        product = []

        for term, rat, sym, deriv in terms:
            if deriv is not None:
                var, order = deriv

                while order > 0:
                    term, order = Derivative(term, var), order - 1

            if sym is None:
                if rat is S.One:
                    product.append(term)
                else:
                    product.append(Pow(term, rat))
            else:
                product.append(Pow(term, rat*sym))

        return Mul(*product)

    def parse_derivative(deriv):
        # scan derivatives tower in the input expression and return
        # underlying function and maximal differentiation order
        expr, sym, order = deriv.expr, deriv.variables[0], 1

        for s in deriv.variables[1:]:
            if s == sym:
                order += 1
            else:
                raise NotImplementedError(
                    'Improve MV Derivative support in collect')

        while isinstance(expr, Derivative):
            s0 = expr.variables[0]

            for s in expr.variables:
                if s != s0:
                    raise NotImplementedError(
                        'Improve MV Derivative support in collect')

            if s0 == sym:
                expr, order = expr.expr, order + len(expr.variables)
            else:
                break

        return expr, (sym, Rational(order))

    def parse_term(expr):
        """Parses expression expr and outputs tuple (sexpr, rat_expo,
        sym_expo, deriv)
        where:
         - sexpr is the base expression
         - rat_expo is the rational exponent that sexpr is raised to
         - sym_expo is the symbolic exponent that sexpr is raised to
         - deriv contains the derivatives the the expression

         for example, the output of x would be (x, 1, None, None)
         the output of 2**x would be (2, 1, x, None)
        """
        rat_expo, sym_expo = S.One, None
        sexpr, deriv = expr, None

        if expr.is_Pow:
            if isinstance(expr.base, Derivative):
                sexpr, deriv = parse_derivative(expr.base)
            else:
                sexpr = expr.base

            if expr.exp.is_Number:
                rat_expo = expr.exp
            else:
                coeff, tail = expr.exp.as_coeff_Mul()

                if coeff.is_Number:
                    rat_expo, sym_expo = coeff, tail
                else:
                    sym_expo = expr.exp
        elif isinstance(expr, exp):
            arg = expr.args[0]
            if arg.is_Rational:
                sexpr, rat_expo = S.Exp1, arg
            elif arg.is_Mul:
                coeff, tail = arg.as_coeff_Mul(rational=True)
                sexpr, rat_expo = exp(tail), coeff
        elif isinstance(expr, Derivative):
            sexpr, deriv = parse_derivative(expr)

        return sexpr, rat_expo, sym_expo, deriv

    def parse_expression(terms, pattern):
        """Parse terms searching for a pattern.
        terms is a list of tuples as returned by parse_terms;
        pattern is an expression treated as a product of factors
        """
        pattern = Mul.make_args(pattern)

        if len(terms) < len(pattern):
            # pattern is longer than matched product
            # so no chance for positive parsing result
            return None
        else:
            pattern = [parse_term(elem) for elem in pattern]

            terms = terms[:]  # need a copy
            elems, common_expo, has_deriv = [], None, False

            for elem, e_rat, e_sym, e_ord in pattern:

                if elem.is_Number and e_rat == 1 and e_sym is None:
                    # a constant is a match for everything
                    continue

                for j in range(len(terms)):
                    if terms[j] is None:
                        continue

                    term, t_rat, t_sym, t_ord = terms[j]

                    # keeping track of whether one of the terms had
                    # a derivative or not as this will require rebuilding
                    # the expression later
                    if t_ord is not None:
                        has_deriv = True

                    if (term.match(elem) is not None and
                            (t_sym == e_sym or t_sym is not None and
                            e_sym is not None and
                            t_sym.match(e_sym) is not None)):
                        if exact is False:
                            # we don't have to be exact so find common exponent
                            # for both expression's term and pattern's element
                            expo = t_rat / e_rat

                            if common_expo is None:
                                # first time
                                common_expo = expo
                            else:
                                # common exponent was negotiated before so
                                # there is no chance for a pattern match unless
                                # common and current exponents are equal
                                if common_expo != expo:
                                    common_expo = 1
                        else:
                            # we ought to be exact so all fields of
                            # interest must match in every details
                            if e_rat != t_rat or e_ord != t_ord:
                                continue

                        # found common term so remove it from the expression
                        # and try to match next element in the pattern
                        elems.append(terms[j])
                        terms[j] = None

                        break

                else:
                    # pattern element not found
                    return None

            return [_f for _f in terms if _f], elems, common_expo, has_deriv

    if evaluate:
        if expr.is_Add:
            o = expr.getO() or 0
            expr = expr.func(*[
                    collect(a, syms, func, True, exact, distribute_order_term)
                    for a in expr.args if a != o]) + o
        elif expr.is_Mul:
            return expr.func(*[
                collect(term, syms, func, True, exact, distribute_order_term)
                for term in expr.args])
        elif expr.is_Pow:
            b = collect(
                expr.base, syms, func, True, exact, distribute_order_term)
            return Pow(b, expr.exp)

    syms = [expand_power_base(i, deep=False) for i in syms]

    order_term = None

    if distribute_order_term:
        order_term = expr.getO()

        if order_term is not None:
            if order_term.has(*syms):
                order_term = None
            else:
                expr = expr.removeO()

    summa = [expand_power_base(i, deep=False) for i in Add.make_args(expr)]

    collected, disliked = defaultdict(list), S.Zero
    for product in summa:
        c, nc = product.args_cnc(split_1=False)
        args = list(ordered(c)) + nc
        terms = [parse_term(i) for i in args]
        small_first = True

        for symbol in syms:
            if SYMPY_DEBUG:
                print("DEBUG: parsing of expression %s with symbol %s " % (
                    str(terms), str(symbol))
                )

            if isinstance(symbol, Derivative) and small_first:
                terms = list(reversed(terms))
                small_first = not small_first
            result = parse_expression(terms, symbol)

            if SYMPY_DEBUG:
                print("DEBUG: returned %s" % str(result))

            if result is not None:
                if not symbol.is_commutative:
                    raise AttributeError("Can not collect noncommutative symbol")

                terms, elems, common_expo, has_deriv = result

                # when there was derivative in current pattern we
                # will need to rebuild its expression from scratch
                if not has_deriv:
                    margs = []
                    for elem in elems:
                        if elem[2] is None:
                            e = elem[1]
                        else:
                            e = elem[1]*elem[2]
                        margs.append(Pow(elem[0], e))
                    index = Mul(*margs)
                else:
                    index = make_expression(elems)
                terms = expand_power_base(make_expression(terms), deep=False)
                index = expand_power_base(index, deep=False)
                collected[index].append(terms)
                break
        else:
            # none of the patterns matched
            disliked += product
    # add terms now for each key
    collected = {k: Add(*v) for k, v in collected.items()}

    if disliked is not S.Zero:
        collected[S.One] = disliked

    if order_term is not None:
        for key, val in collected.items():
            collected[key] = val + order_term

    if func is not None:
        collected = dict(
            [(key, func(val)) for key, val in collected.items()])

    if evaluate:
        return Add(*[key*val for key, val in collected.items()])
    else:
        return collected
