from __future__ import print_function, division
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer, Rational, Float
from sympy.core.compatibility import default_sort_key
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.printing.repr import srepr
from sympy.utilities.misc import func_name

__all__ = ['dotprint']
default_styles = ((Basic, {'color': 'blue', 'shape': 'ellipse'}),
          (Expr,  {'color': 'black'}))
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

def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
    """ String defining a node

    Examples
    ========

    >>> from sympy.printing.dot import dotnode
    >>> from sympy.abc import x
    >>> print(dotnode(x))
    "Symbol('x')_()" ["color"="black", "label"="x", "shape"="ellipse"];
    """
    style = styleof(expr, styles)

    if isinstance(expr, Basic) and not expr.is_Atom:
        label = str(expr.__class__.__name__)
    else:
        label = labelfunc(expr)
    style['label'] = label
    expr_str = purestr(expr)
    if repeat:
        expr_str += '_%s' % str(pos)
    return '"%s" [%s];' % (expr_str, attrprint(style))
