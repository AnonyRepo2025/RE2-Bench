from __future__ import print_function, division
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer, Rational, Float
from sympy.printing.repr import srepr

__all__ = ['dotprint']
default_styles = (
    (Basic, {'color': 'blue', 'shape': 'ellipse'}),
    (Expr,  {'color': 'black'})
)
slotClasses = (Symbol, Integer, Rational, Float)
template = \
"""digraph{

# Graph style
%(graphstyle)s

#########
# Nodes #
#########

%(nodes)s

#########
# Edges #
#########

%(edges)s
}"""
_graphstyle = {'rankdir': 'TD', 'ordering': 'out'}

def purestr(x, with_args=False):
    """A string that follows ```obj = type(obj)(*obj.args)``` exactly.

    Parameters
    ==========

    with_args : boolean, optional
        If ``True``, there will be a second argument for the return
        value, which is a tuple containing ``purestr`` applied to each
        of the subnodes.

        If ``False``, there will not be a second argument for the
        return.

        Default is ``False``

    Examples
    ========

    >>> from sympy import Float, Symbol, MatrixSymbol
    >>> from sympy import Integer # noqa: F401
    >>> from sympy.core.symbol import Str # noqa: F401
    >>> from sympy.printing.dot import purestr

    Applying ``purestr`` for basic symbolic object:
    >>> code = purestr(Symbol('x'))
    >>> code
    "Symbol('x')"
    >>> eval(code) == Symbol('x')
    True

    For basic numeric object:
    >>> purestr(Float(2))
    "Float('2.0', precision=53)"

    For matrix symbol:
    >>> code = purestr(MatrixSymbol('x', 2, 2))
    >>> code
    "MatrixSymbol(Str('x'), Integer(2), Integer(2))"
    >>> eval(code) == MatrixSymbol('x', 2, 2)
    True

    With ``with_args=True``:
    >>> purestr(Float(2), with_args=True)
    ("Float('2.0', precision=53)", ())
    >>> purestr(MatrixSymbol('x', 2, 2), with_args=True)
    ("MatrixSymbol(Str('x'), Integer(2), Integer(2))",
     ("Str('x')", 'Integer(2)', 'Integer(2)'))
    """
    sargs = ()
    if not isinstance(x, Basic):
        rv = str(x)
    elif not x.args:
        rv = srepr(x)
    else:
        args = x.args
        sargs = tuple(map(purestr, args))
        rv = "%s(%s)"%(type(x).__name__, ', '.join(sargs))
    if with_args:
        rv = rv, sargs
    return rv
