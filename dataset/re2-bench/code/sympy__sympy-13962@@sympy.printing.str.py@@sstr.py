from __future__ import print_function, division
from sympy.core import S, Rational, Pow, Basic, Mul
from sympy.core.mul import _keep_coeff
from .printer import Printer
from sympy.printing.precedence import precedence, PRECEDENCE
import mpmath.libmp as mlib
from mpmath.libmp import prec_to_dps
from sympy.utilities import default_sort_key
from sympy.combinatorics.permutations import Permutation, Cycle
from sympy.polys.polyerrors import PolynomialError
from sympy.matrices import Matrix
from sympy.core.sympify import SympifyError



def sstr(expr, **settings):
    """Returns the expression as a string.

    For large expressions where speed is a concern, use the setting
    order='none'. If abbrev=True setting is used then units are printed in
    abbreviated form.

    Examples
    ========

    >>> from sympy import symbols, Eq, sstr
    >>> a, b = symbols('a b')
    >>> sstr(Eq(a + b, 0))
    'Eq(a + b, 0)'
    """

    p = StrPrinter(settings)
    s = p.doprint(expr)

    return s
