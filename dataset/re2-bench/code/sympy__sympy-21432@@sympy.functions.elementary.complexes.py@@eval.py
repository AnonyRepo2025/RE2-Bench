from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
    AppliedUndef)
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.relational import Eq
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import atan, atan2
from sympy import Integral
import sage.all as sage
import sage.all as sage
import sage.all as sage
from sympy.functions.special.delta_functions import Heaviside
from sympy.simplify.simplify import signsimp
from sympy.core.function import expand_mul
from sympy.core.power import Pow
import sage.all as sage
from sympy.functions.special.delta_functions import Heaviside
from sympy.printing.pretty.stringpict import prettyForm
from sympy.functions.elementary.complexes import arg as argument
from sympy import oo, exp_polar, I, Mul, polar_lift, Symbol
from sympy import exp, pi, I
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.delta_functions import DiracDelta



class arg(Function):
    is_extended_real = True
    is_real = True
    is_finite = True
    _singularities = True  # non-holomorphic
    @classmethod
    def eval(cls, arg):
        if isinstance(arg, exp_polar):
            return periodic_argument(arg, oo)
        if not arg.is_Atom:
            c, arg_ = factor_terms(arg).as_coeff_Mul()
            if arg_.is_Mul:
                arg_ = Mul(*[a if (sign(a) not in (-1, 1)) else
                    sign(a) for a in arg_.args])
            arg_ = sign(c)*arg_
        else:
            arg_ = arg
        if any(i.is_extended_positive is None for i in arg_.atoms(AppliedUndef)):
            return
        x, y = arg_.as_real_imag()
        rv = atan2(y, x)
        if rv.is_number:
            return rv
        if arg_ != arg:
            return cls(arg_, evaluate=False)