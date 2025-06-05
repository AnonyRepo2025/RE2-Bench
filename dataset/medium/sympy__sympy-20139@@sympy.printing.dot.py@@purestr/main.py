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
