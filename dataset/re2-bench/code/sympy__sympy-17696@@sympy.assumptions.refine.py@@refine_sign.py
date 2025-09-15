from __future__ import print_function, division
from sympy.core import S, Add, Expr, Basic, Mul
from sympy.assumptions import Q, ask
from sympy.core.logic import fuzzy_not
from sympy import Abs
from sympy.core import Pow, Rational
from sympy.functions.elementary.complexes import Abs
from sympy.functions import sign
from sympy.functions.elementary.trigonometric import atan
from sympy.core import S

handlers_dict = {
    'Abs': refine_abs,
    'Pow': refine_Pow,
    'atan2': refine_atan2,
    'Equality': refine_Relational,
    'Unequality': refine_Relational,
    'GreaterThan': refine_Relational,
    'LessThan': refine_Relational,
    'StrictGreaterThan': refine_Relational,
    'StrictLessThan': refine_Relational,
    're': refine_re,
    'im': refine_im,
    'sign': refine_sign
}

def refine_sign(expr, assumptions):
    """
    Handler for sign

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_sign
    >>> from sympy import Symbol, Q, sign, im
    >>> x = Symbol('x', real = True)
    >>> expr = sign(x)
    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))
    1
    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))
    -1
    >>> refine_sign(expr, Q.zero(x))
    0
    >>> y = Symbol('y', imaginary = True)
    >>> expr = sign(y)
    >>> refine_sign(expr, Q.positive(im(y)))
    I
    >>> refine_sign(expr, Q.negative(im(y)))
    -I
    """
    arg = expr.args[0]
    if ask(Q.zero(arg), assumptions):
        return S.Zero
    if ask(Q.real(arg)):
        if ask(Q.positive(arg), assumptions):
            return S.One
        if ask(Q.negative(arg), assumptions):
            return S.NegativeOne
    if ask(Q.imaginary(arg)):
        arg_re, arg_im = arg.as_real_imag()
        if ask(Q.positive(arg_im), assumptions):
            return S.ImaginaryUnit
        if ask(Q.negative(arg_im), assumptions):
            return -S.ImaginaryUnit
    return expr
