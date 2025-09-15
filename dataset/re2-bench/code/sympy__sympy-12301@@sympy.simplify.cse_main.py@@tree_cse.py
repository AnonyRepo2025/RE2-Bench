from __future__ import print_function, division
from sympy.core import Basic, Mul, Add, Pow, sympify, Symbol, Tuple, igcd
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.function import _coeff_isneg
from sympy.core.exprtools import factor_terms
from sympy.core.compatibility import iterable, range, as_int
from sympy.utilities.iterables import filter_symbols, \
    numbered_symbols, sift, topological_sort, ordered, subsets
from . import cse_opts
from sympy.utilities.iterables import subsets
from collections import defaultdict
from sympy.matrices.expressions import MatAdd, MatMul, MatPow
from sympy.matrices.expressions import MatrixExpr, MatrixSymbol, MatMul, MatAdd
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
                                SparseMatrix, ImmutableSparseMatrix)
from sympy.core.add import _addsort as inplace_sorter
from sympy.core.mul import _mulsort as inplace_sorter

basic_optimizations = [(cse_opts.sub_pre, cse_opts.sub_post),
                       (factor_terms, None)]

def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
    """Perform raw CSE on expression tree, taking opt_subs into account.

    Parameters
    ==========

    exprs : list of sympy expressions
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out.
    opt_subs : dictionary of expression substitutions
        The expressions to be substituted before any CSE action is performed.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.
    ignore : iterable of Symbols
        Substitutions containing any Symbol from ``ignore`` will be ignored.
    """
    from sympy.matrices.expressions import MatrixExpr, MatrixSymbol, MatMul, MatAdd

    if opt_subs is None:
        opt_subs = dict()

    ## Find repeated sub-expressions

    to_eliminate = set()

    seen_subexp = set()

    def _find_repeated(expr):
        if not isinstance(expr, Basic):
            return

        if expr.is_Atom or expr.is_Order:
            return

        if iterable(expr):
            args = expr

        else:
            if expr in seen_subexp:
                for ign in ignore:
                    if ign in expr.free_symbols:
                        break
                else:
                    to_eliminate.add(expr)
                    return

            seen_subexp.add(expr)

            if expr in opt_subs:
                expr = opt_subs[expr]

            args = expr.args

        list(map(_find_repeated, args))

    for e in exprs:
        if isinstance(e, Basic):
            _find_repeated(e)

    ## Rebuild tree

    replacements = []

    subs = dict()

    def _rebuild(expr):
        if not isinstance(expr, Basic):
            return expr

        if not expr.args:
            return expr

        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr]
            return expr.func(*new_args)

        if expr in subs:
            return subs[expr]

        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]

        # If enabled, parse Muls and Adds arguments by order to ensure
        # replacement order independent from hashes
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                c, nc = expr.args_cnc()
                if c == [1]:
                    args = nc
                else:
                    args = list(ordered(c)) + nc
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args

        new_args = list(map(_rebuild, args))
        if new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr

        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError("Symbols iterator ran out of symbols.")

            if isinstance(orig_expr, MatrixExpr):
                sym = MatrixSymbol(sym.name, orig_expr.rows,
                    orig_expr.cols)

            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym

        else:
            return new_expr

    reduced_exprs = []
    for e in exprs:
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        reduced_exprs.append(reduced_e)

    # don't allow hollow nesting
    # e.g if p = [b + 2*d + e + f, b + 2*d + f + g, a + c + d + f + g]
    # and R, C = cse(p) then
    #     R = [(x0, d + f), (x1, b + d)]
    #     C = [e + x0 + x1, g + x0 + x1, a + c + d + f + g]
    # but the args of C[-1] should not be `(a + c, d + f + g)`
    for i in range(len(exprs)):
        F = reduced_exprs[i].func
        if not (F is Mul or F is Add):
            continue
        if any(isinstance(a, F) for a in reduced_exprs[i].args):
            reduced_exprs[i] = F(*reduced_exprs[i].args)

    return replacements, reduced_exprs
