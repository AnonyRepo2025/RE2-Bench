from typing import Dict, Callable
from sympy.core import S, Add, Expr, Basic, Mul
from sympy.logic.boolalg import Boolean
from sympy.assumptions import ask, Q  # type: ignore
from sympy.core.logic import fuzzy_not
from sympy import Abs
from sympy.core import Pow, Rational
from sympy.functions.elementary.complexes import Abs
from sympy.functions import sign
from sympy.functions.elementary.trigonometric import atan
from sympy.core import S
from sympy.matrices.expressions.matexpr import MatrixElement

handlers_dict = {
    'Abs': refine_abs,
    'Pow': refine_Pow,
    'atan2': refine_atan2,
    're': refine_re,
    'im': refine_im,
    'arg': refine_arg,
    'sign': refine_sign,
    'MatrixElement': refine_matrixelement
}  # type: Dict[str, Callable[[Expr, Boolean], Expr]]

def refine_arg(expr, assumptions):
    """
    Handler for complex argument

    Explanation
    ===========

    >>> from sympy.assumptions.refine import refine_arg
    >>> from sympy import Q, arg
    >>> from sympy.abc import x
    >>> refine_arg(arg(x), Q.positive(x))
    0
    >>> refine_arg(arg(x), Q.negative(x))
    pi
    """
    rg = expr.args[0]
    if ask(Q.positive(rg), assumptions):
        return S.Zero
    if ask(Q.negative(rg), assumptions):
        return S.Pi
    return None
