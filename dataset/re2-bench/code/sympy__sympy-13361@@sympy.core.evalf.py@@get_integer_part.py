from __future__ import print_function, division
import math
import mpmath.libmp as libmp
from mpmath import (
    make_mpc, make_mpf, mp, mpc, mpf, nsum, quadts, quadosc, workprec)
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
        fnan, fnone, fone, fzero, mpf_abs, mpf_add,
        mpf_atan, mpf_atan2, mpf_cmp, mpf_cos, mpf_e, mpf_exp, mpf_log, mpf_lt,
        mpf_mul, mpf_neg, mpf_pi, mpf_pow, mpf_pow_int, mpf_shift, mpf_sin,
        mpf_sqrt, normalize, round_nearest, to_int, to_str)
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from mpmath.libmp.gammazeta import mpf_bernoulli
from .compatibility import SYMPY_INTS, range
from .sympify import sympify
from .singleton import S
from sympy.utilities.iterables import is_sequence
from sympy.functions.elementary.complexes import re, im
from sympy.core.numbers import Float
from sympy.core.numbers import Float
from sympy import cos, sin
from sympy import Abs, Add, log
from sympy import Float, Integer
from sympy.core.numbers import Infinity, NegativeInfinity, Zero
from sympy import Poly
from sympy import Float, hypersimp, lambdify
from sympy import Sum
from sympy import Float
from sympy.functions.combinatorial.numbers import bernoulli
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Exp1, Float, Half, ImaginaryUnit, Integer, NaN, NegativeOne, One, Pi, Rational, Zero
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import atan, cos, sin
from sympy.integrals.integrals import Integral
from sympy import re as re_, im as im_
from sympy.core.add import Add
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy import cos, sin, Wild
from sympy import Float, Number
from sympy.core.expr import _mag
from sympy.core.compatibility import as_int

LG10 = math.log(10, 2)
rnd = round_nearest
INF = float(mpmath_inf)
MINUS_INF = float(-mpmath_inf)
DEFAULT_MAXPREC = 333
evalf_table = None

def get_integer_part(expr, no, options, return_ints=False):
    """
    With no = 1, computes ceiling(expr)
    With no = -1, computes floor(expr)

    Note: this function either gives the exact result or signals failure.
    """
    from sympy.functions.elementary.complexes import re, im
    # The expression is likely less than 2^30 or so
    assumed_size = 30
    ire, iim, ire_acc, iim_acc = evalf(expr, assumed_size, options)

    # We now know the size, so we can calculate how much extra precision
    # (if any) is needed to get within the nearest integer
    if ire and iim:
        gap = max(fastlog(ire) - ire_acc, fastlog(iim) - iim_acc)
    elif ire:
        gap = fastlog(ire) - ire_acc
    elif iim:
        gap = fastlog(iim) - iim_acc
    else:
        # ... or maybe the expression was exactly zero
        return None, None, None, None

    margin = 10

    if gap >= -margin:
        ire, iim, ire_acc, iim_acc = \
            evalf(expr, margin + assumed_size + gap, options)

    # We can now easily find the nearest integer, but to find floor/ceil, we
    # must also calculate whether the difference to the nearest integer is
    # positive or negative (which may fail if very close).
    def calc_part(expr, nexpr):
        from sympy.core.add import Add
        nint = int(to_int(nexpr, rnd))
        n, c, p, b = nexpr
        is_int = (p == 0)
        if not is_int:
            # if there are subs and they all contain integer re/im parts
            # then we can (hopefully) safely substitute them into the
            # expression
            s = options.get('subs', False)
            if s:
                doit = True
                from sympy.core.compatibility import as_int
                for v in s.values():
                    try:
                        as_int(v)
                    except ValueError:
                        try:
                            [as_int(i) for i in v.as_real_imag()]
                            continue
                        except (ValueError, AttributeError):
                            doit = False
                            break
                if doit:
                    expr = expr.subs(s)

            expr = Add(expr, -nint, evaluate=False)
            x, _, x_acc, _ = evalf(expr, 10, options)
            try:
                check_target(expr, (x, None, x_acc, None), 3)
            except PrecisionExhausted:
                if not expr.equals(0):
                    raise PrecisionExhausted
                x = fzero
            nint += int(no*(mpf_cmp(x or fzero, fzero) == no))
        nint = from_int(nint)
        return nint, INF

    re_, im_, re_acc, im_acc = None, None, None, None

    if ire:
        re_, re_acc = calc_part(re(expr, evaluate=False), ire)
    if iim:
        im_, im_acc = calc_part(im(expr, evaluate=False), iim)

    if return_ints:
        return int(to_int(re_ or fzero)), int(to_int(im_ or fzero))
    return re_, im_, re_acc, im_acc
