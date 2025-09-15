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

def dotprint(expr, styles=default_styles, atom=lambda x: not isinstance(x,
    Basic), maxdepth=None, repeat=True, labelfunc=str, **kwargs):
    """
    DOT description of a SymPy expression tree

    Options are

    ``styles``: Styles for different classes.  The default is::

        [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
        (Expr, {'color': 'black'})]``

    ``atom``: Function used to determine if an arg is an atom.  The default is
          ``lambda x: not isinstance(x, Basic)``.  Another good choice is
          ``lambda x: not x.args``.

    ``maxdepth``: The maximum depth.  The default is None, meaning no limit.

    ``repeat``: Whether to different nodes for separate common subexpressions.
          The default is True.  For example, for ``x + x*y`` with
          ``repeat=True``, it will have two nodes for ``x`` and with
          ``repeat=False``, it will have one (warning: even if it appears
          twice in the same object, like Pow(x, x), it will still only appear
          once.  Hence, with repeat=False, the number of arrows out of an
          object might not equal the number of args it has).

    ``labelfunc``: How to label leaf nodes.  The default is ``str``.  Another
          good option is ``srepr``. For example with ``str``, the leaf nodes
          of ``x + 1`` are labeled, ``x`` and ``1``.  With ``srepr``, they
          are labeled ``Symbol('x')`` and ``Integer(1)``.

    Additional keyword arguments are included as styles for the graph.

    Examples
    ========

    >>> from sympy.printing.dot import dotprint
    >>> from sympy.abc import x
    >>> print(dotprint(x+2)) # doctest: +NORMALIZE_WHITESPACE
    digraph{
    <BLANKLINE>
    # Graph style
    "ordering"="out"
    "rankdir"="TD"
    <BLANKLINE>
    #########
    # Nodes #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol('x'))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
    "Integer(2)_(0,)" ["color"="black", "label"="2", "shape"="ellipse"];
    "Symbol('x')_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
    <BLANKLINE>
    #########
    # Edges #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
    }

    """
    # repeat works by adding a signature tuple to the end of each node for its
    # position in the graph. For example, for expr = Add(x, Pow(x, 2)), the x in the
    # Pow will have the tuple (1, 0), meaning it is expr.args[1].args[0].
    graphstyle = _graphstyle.copy()
    graphstyle.update(kwargs)

    nodes = []
    edges = []
    def traverse(e, depth, pos=()):
        nodes.append(dotnode(e, styles, labelfunc=labelfunc, pos=pos, repeat=repeat))
        if maxdepth and depth >= maxdepth:
            return
        edges.extend(dotedges(e, atom=atom, pos=pos, repeat=repeat))
        [traverse(arg, depth+1, pos + (i,)) for i, arg in enumerate(e.args) if not atom(arg)]
    traverse(expr, 0)

    return template%{'graphstyle': attrprint(graphstyle, delimiter='\n'),
                     'nodes': '\n'.join(nodes),
                     'edges': '\n'.join(edges)}
