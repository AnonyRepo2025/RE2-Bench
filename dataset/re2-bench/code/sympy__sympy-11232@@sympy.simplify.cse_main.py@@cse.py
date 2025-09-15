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

def cse(exprs, symbols=None, optimizations=None, postprocess=None,
        order='canonical'):
    """ Perform common subexpression elimination on an expression.

    Parameters
    ==========

    exprs : list of sympy expressions, or a single sympy expression
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out. The ``numbered_symbols`` generator is useful. The default is a
        stream of symbols of the form "x0", "x1", etc. This must be an
        infinite iterator.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs of external optimization
        functions. Optionally 'basic' can be passed for a set of predefined
        basic optimizations. Such 'basic' optimizations were used by default
        in old implementation, however they can be really slow on larger
        expressions. Now, no pre or post optimizations are made by default.
    postprocess : a function which accepts the two return values of cse and
        returns the desired form of output from cse, e.g. if you want the
        replacements reversed the function might be the following lambda:
        lambda r, e: return reversed(r), e
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. If set to
        'canonical', arguments will be canonically ordered. If set to 'none',
        ordering will be faster but dependent on expressions hashes, thus
        machine dependent and variable. For large expressions where speed is a
        concern, use the setting order='none'.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of sympy expressions
        The reduced expressions with all of the replacements above.

    Examples
    ========

    >>> from sympy import cse, SparseMatrix
    >>> from sympy.abc import x, y, z, w
    >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)
    ([(x0, w + y + z)], [x0*(x + x0)/(w + x)**3])

    Note that currently, y + z will not get substituted if -y - z is used.

     >>> cse(((w + x + y + z)*(w - y - z))/(w + x)**3)
     ([(x0, w + x)], [(w - y - z)*(x0 + y + z)/x0**3])

    List of expressions with recursive substitutions:

    >>> m = SparseMatrix([x + y, x + y + z])
    >>> cse([(x+y)**2, x + y + z, y + z, x + z + y, m])
    ([(x0, x + y), (x1, x0 + z)], [x0**2, x1, y + z, x1, Matrix([
    [x0],
    [x1]])])

    Note: the type and mutability of input matrices is retained.

    >>> isinstance(_[1][-1], SparseMatrix)
    True
    """
    from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
                                SparseMatrix, ImmutableSparseMatrix)

    # Handle the case if just one expression was passed.
    if isinstance(exprs, (Basic, MatrixBase)):
        exprs = [exprs]

    copy = exprs
    temp = []
    for e in exprs:
        if isinstance(e, (Matrix, ImmutableMatrix)):
            temp.append(Tuple(*e._mat))
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            temp.append(Tuple(*e._smat.items()))
        else:
            temp.append(e)
    exprs = temp
    del temp

    if optimizations is None:
        optimizations = list()
    elif optimizations == 'basic':
        optimizations = basic_optimizations

    # Preprocess the expressions to give us better optimization opportunities.
    reduced_exprs = [preprocess_for_cse(e, optimizations) for e in exprs]

    excluded_symbols = set().union(*[expr.atoms(Symbol)
                                   for expr in reduced_exprs])

    if symbols is None:
        symbols = numbered_symbols()
    else:
        # In case we get passed an iterable with an __iter__ method instead of
        # an actual iterator.
        symbols = iter(symbols)

    symbols = filter_symbols(symbols, excluded_symbols)

    # Find other optimization opportunities.
    opt_subs = opt_cse(reduced_exprs, order)

    # Main CSE algorithm.
    replacements, reduced_exprs = tree_cse(reduced_exprs, symbols, opt_subs,
                                           order)

    # Postprocess the expressions to return the expressions to canonical form.
    exprs = copy
    for i, (sym, subtree) in enumerate(replacements):
        subtree = postprocess_for_cse(subtree, optimizations)
        replacements[i] = (sym, subtree)
    reduced_exprs = [postprocess_for_cse(e, optimizations)
                     for e in reduced_exprs]

    # Get the matrices back
    for i, e in enumerate(exprs):
        if isinstance(e, (Matrix, ImmutableMatrix)):
            reduced_exprs[i] = Matrix(e.rows, e.cols, reduced_exprs[i])
            if isinstance(e, ImmutableMatrix):
                reduced_exprs[i] = reduced_exprs[i].as_immutable()
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            m = SparseMatrix(e.rows, e.cols, {})
            for k, v in reduced_exprs[i]:
                m[k] = v
            if isinstance(e, ImmutableSparseMatrix):
                m = m.as_immutable()
            reduced_exprs[i] = m

    if postprocess is None:
        return replacements, reduced_exprs

    return postprocess(replacements, reduced_exprs)
