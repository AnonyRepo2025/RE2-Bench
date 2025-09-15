from __future__ import print_function, division
from sympy.core.add import Add
from sympy.core.compatibility import as_int, is_sequence, range
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
    divisors, factorint, multiplicity, perfect_power)
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solvers import check_assumptions
from sympy.solvers.solveset import solveset_real
from sympy.utilities import default_sort_key, numbered_symbols
from sympy.utilities.misc import filldedent
from sympy.utilities.iterables import (
        subsets, permute_signs, signed_permutations)
from sympy.ntheory.continued_fraction import continued_fraction_periodic
from sympy.simplify.simplify import clear_coefficients
from sympy.utilities.iterables import ordered_partitions

__all__ = ['diophantine', 'classify_diop']
diop_known = {
    "binary_quadratic",
    "cubic_thue",
    "general_pythagorean",
    "general_sum_of_even_powers",
    "general_sum_of_squares",
    "homogeneous_general_quadratic",
    "homogeneous_ternary_quadratic",
    "homogeneous_ternary_quadratic_normal",
    "inhomogeneous_general_quadratic",
    "inhomogeneous_ternary_quadratic",
    "linear",
    "univariate"}
classify_diop.func_doc = '''
    Helper routine used by diop_solve() to find information about ``eq``.

    Returns a tuple containing the type of the diophantine equation
    along with the variables (free symbols) and their coefficients.
    Variables are returned as a list and coefficients are returned
    as a dict with the key being the respective term and the constant
    term is keyed to 1. The type is one of the following:

    * %s

    Usage
    =====

    ``classify_diop(eq)``: Return variables, coefficients and type of the
    ``eq``.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``_dict`` is for internal use: when True (default) a dict is returned,
    otherwise a defaultdict which supplies 0 for missing keys is returned.

    Examples
    ========

    >>> from sympy.solvers.diophantine import classify_diop
    >>> from sympy.abc import x, y, z, w, t
    >>> classify_diop(4*x + 6*y - 4)
    ([x, y], {1: -4, x: 4, y: 6}, 'linear')
    >>> classify_diop(x + 3*y -4*z + 5)
    ([x, y, z], {1: 5, x: 1, y: 3, z: -4}, 'linear')
    >>> classify_diop(x**2 + y**2 - x*y + x + 5)
    ([x, y], {1: 5, x: 1, x**2: 1, y**2: 1, x*y: -1}, 'binary_quadratic')
    ''' % ('\n    * '.join(sorted(diop_known)))
sum_of_powers = power_representation

def diophantine(eq, param=symbols("t", integer=True), syms=None,
                permute=False):
    """
    Simplify the solution procedure of diophantine equation ``eq`` by
    converting it into a product of terms which should equal zero.

    For example, when solving, `x^2 - y^2 = 0` this is treated as
    `(x + y)(x - y) = 0` and `x + y = 0` and `x - y = 0` are solved
    independently and combined. Each term is solved by calling
    ``diop_solve()``. (Although it is possible to call ``diop_solve()``
    directly, one must be careful to pass an equation in the correct
    form and to interpret the output correctly; ``diophantine()`` is
    the public-facing function to use in general.)

    Output of ``diophantine()`` is a set of tuples. The elements of the
    tuple are the solutions for each variable in the equation and
    are arranged according to the alphabetic ordering of the variables.
    e.g. For an equation with two variables, `a` and `b`, the first
    element of the tuple is the solution for `a` and the second for `b`.

    Usage
    =====

    ``diophantine(eq, t, syms)``: Solve the diophantine
    equation ``eq``.
    ``t`` is the optional parameter to be used by ``diop_solve()``.
    ``syms`` is an optional list of symbols which determines the
    order of the elements in the returned tuple.

    By default, only the base solution is returned. If ``permute`` is set to
    True then permutations of the base solution and/or permutations of the
    signs of the values will be returned when applicable.

    >>> from sympy.solvers.diophantine import diophantine
    >>> from sympy.abc import a, b
    >>> eq = a**4 + b**4 - (2**4 + 3**4)
    >>> diophantine(eq)
    {(2, 3)}
    >>> diophantine(eq, permute=True)
    {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``t`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> diophantine(x**2 - y**2)
    {(t_0, -t_0), (t_0, t_0)}

    >>> diophantine(x*(2*x + 3*y - z))
    {(0, n1, n2), (t_0, t_1, 2*t_0 + 3*t_1)}
    >>> diophantine(x**2 + 3*x*y + 4*x)
    {(0, n1), (3*t_0 - 4, -t_0)}

    See Also
    ========

    diop_solve()
    sympy.utilities.iterables.permute_signs
    sympy.utilities.iterables.signed_permutations
    """

    from sympy.utilities.iterables import (
        subsets, permute_signs, signed_permutations)

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs

    try:
        var = list(eq.expand(force=True).free_symbols)
        var.sort(key=default_sort_key)
        if syms:
            if not is_sequence(syms):
                raise TypeError(
                    'syms should be given as a sequence, e.g. a list')
            syms = [i for i in syms if i in var]
            if syms != var:
                dict_sym_index = dict(zip(syms, range(len(syms))))
                return {tuple([t[dict_sym_index[i]] for i in var])
                            for t in diophantine(eq, param, permute=permute)}
        n, d = eq.as_numer_denom()
        if n.is_number:
            return set()
        if not d.is_number:
            dsol = diophantine(d)
            good = diophantine(n) - dsol
            return {s for s in good if _mexpand(d.subs(zip(var, s)))}
        else:
            eq = n
        eq = factor_terms(eq)
        assert not eq.is_number
        eq = eq.as_independent(*var, as_Add=False)[1]
        p = Poly(eq)
        assert not any(g.is_number for g in p.gens)
        eq = p.as_expr()
        assert eq.is_polynomial()
    except (GeneratorsNeeded, AssertionError, AttributeError):
        raise TypeError(filldedent('''
    Equation should be a polynomial with Rational coefficients.'''))

    # permute only sign
    do_permute_signs = False
    # permute sign and values
    do_permute_signs_var = False
    # permute few signs
    permute_few_signs = False
    try:
        # if we know that factoring should not be attempted, skip
        # the factoring step
        v, c, t = classify_diop(eq)

        # check for permute sign
        if permute:
            len_var = len(v)
            permute_signs_for = [
                'general_sum_of_squares',
                'general_sum_of_even_powers']
            permute_signs_check = [
                'homogeneous_ternary_quadratic',
                'homogeneous_ternary_quadratic_normal',
                'binary_quadratic']
            if t in permute_signs_for:
                do_permute_signs_var = True
            elif t in permute_signs_check:
                # if all the variables in eq have even powers
                # then do_permute_sign = True
                if len_var == 3:
                    var_mul = list(subsets(v, 2))
                    # here var_mul is like [(x, y), (x, z), (y, z)]
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = map(lambda a: a[0]*a[1], var_mul)
                    # if coeff(y*z), coeff(y*x), coeff(x*z) is not 0 then
                    # `xy_coeff` => True and do_permute_sign => False.
                    # Means no permuted solution.
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    # here var_mul is like [(x,), (y, )]
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any([xy_coeff, x_coeff]):
                        # means only x**2, y**2, z**2, const is present
                        do_permute_signs = True
                    elif not x_coeff:
                        permute_few_signs = True
                elif len_var == 2:
                    var_mul = list(subsets(v, 2))
                    # here var_mul is like [(x, y)]
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = map(lambda x: x[0]*x[1], var_mul)
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    # here var_mul is like [(x,), (y, )]
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any([xy_coeff, x_coeff]):
                        # means only x**2, y**2 and const is present
                        # so we can get more soln by permuting this soln.
                        do_permute_signs = True
                    elif not x_coeff:
                        # when coeff(x), coeff(y) is not present then signs of
                        #  x, y can be permuted such that their sign are same
                        # as sign of x*y.
                        # e.g 1. (x_val,y_val)=> (x_val,y_val), (-x_val,-y_val)
                        # 2. (-x_vall, y_val)=> (-x_val,y_val), (x_val,-y_val)
                        permute_few_signs = True
        if t == 'general_sum_of_squares':
            # trying to factor such expressions will sometimes hang
            terms = [(eq, 1)]
        else:
            raise TypeError
    except (TypeError, NotImplementedError):
        fl = factor_list(eq)
        if fl[0].is_Rational and fl[0] != 1:
            return diophantine(eq/fl[0], param=param, syms=syms, permute=permute)
        terms = fl[1]

    sols = set([])

    for term in terms:

        base, _ = term
        var_t, _, eq_type = classify_diop(base, _dict=False)
        _, base = signsimp(base, evaluate=False).as_coeff_Mul()
        solution = diop_solve(base, param)

        if eq_type in [
                "linear",
                "homogeneous_ternary_quadratic",
                "homogeneous_ternary_quadratic_normal",
                "general_pythagorean"]:
            sols.add(merge_solution(var, var_t, solution))

        elif eq_type in [
                "binary_quadratic",
                "general_sum_of_squares",
                "general_sum_of_even_powers",
                "univariate"]:
            for sol in solution:
                sols.add(merge_solution(var, var_t, sol))

        else:
            raise NotImplementedError('unhandled type: %s' % eq_type)

    # remove null merge results
    if () in sols:
        sols.remove(())
    null = tuple([0]*len(var))
    # if there is no solution, return trivial solution
    if not sols and eq.subs(zip(var, null)).is_zero:
        sols.add(null)
    final_soln = set([])
    for sol in sols:
        if all(_is_int(s) for s in sol):
            if do_permute_signs:
                permuted_sign = set(permute_signs(sol))
                final_soln.update(permuted_sign)
            elif permute_few_signs:
                lst = list(permute_signs(sol))
                lst = list(filter(lambda x: x[0]*x[1] == sol[1]*sol[0], lst))
                permuted_sign = set(lst)
                final_soln.update(permuted_sign)
            elif do_permute_signs_var:
                permuted_sign_var = set(signed_permutations(sol))
                final_soln.update(permuted_sign_var)
            else:
                final_soln.add(sol)
        else:
                final_soln.add(sol)
    return final_soln
