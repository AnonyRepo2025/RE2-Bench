from __future__ import print_function, division
from .sympify import sympify, _sympify, SympifyError
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex
from .decorators import _sympifyit, call_highest_priority
from .cache import cacheit
from .compatibility import reduce, as_int, default_sort_key, range
from mpmath.libmp import mpf_log, prec_to_dps
from collections import defaultdict
from .mul import Mul
from .add import Add
from .power import Pow
from .function import Derivative, Function
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Integer, Rational
from math import log10, ceil, log
from sympy import Float
from sympy import Abs
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy import Dummy
from sympy import GreaterThan
from sympy import LessThan
from sympy import StrictGreaterThan
from sympy import StrictLessThan
from sympy import Float
from sympy.simplify.simplify import nsimplify, simplify
from sympy.solvers.solveset import solveset
from sympy.polys.polyerrors import NotAlgebraic
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.polyerrors import NotAlgebraic
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.polyerrors import NotAlgebraic
from sympy.series import limit, Limit
from sympy.solvers.solveset import solveset
from sympy.functions.elementary.complexes import conjugate as c
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.complexes import transpose
from sympy.functions.elementary.complexes import conjugate, transpose
from sympy.functions.elementary.complexes import adjoint
from sympy.polys.orderings import monomial_key
from .add import Add
from .mul import Mul
from .exprtools import decompose_power
from sympy import Dummy, Symbol
from .function import count_ops
from .symbol import Symbol
from .add import _unevaluated_Add
from .mul import _unevaluated_Mul
from sympy.utilities.iterables import sift
from sympy import im, re
from .function import _coeff_isneg
from sympy import exp_polar, pi, I, ceiling, Add
from sympy import collect, Dummy, Order, Rational, Symbol
from sympy import Dummy, factorial
from sympy.utilities.misc import filldedent
from sympy.series.limits import limit
from sympy import Dummy, log
from sympy.series.gruntz import calculate_series
from sympy import powsimp
from sympy import collect
from sympy import Dummy, log
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.simplify.radsimp import fraction
from sympy.integrals import integrate
from sympy.simplify import simplify
from sympy.core.function import count_ops
from sympy.simplify import nsimplify
from sympy.core.function import expand_power_base
from sympy.simplify import collect
from sympy.polys import together
from sympy.polys import apart
from sympy.simplify import ratsimp
from sympy.simplify import trigsimp
from sympy.simplify import radsimp
from sympy.simplify import powsimp
from sympy.simplify import combsimp
from sympy.polys import factor
from sympy.assumptions import refine
from sympy.polys import cancel
from sympy.polys.polytools import invert
from sympy.core.numbers import mod_inverse
from sympy import Float
from sympy.utilities.randtest import random_complex_number
from mpmath.libmp.libintmath import giant_steps
from sympy.core.evalf import DEFAULT_MAXPREC as target
from sympy.utilities.misc import filldedent



class Expr(Basic, EvalfMixin):
    __slots__ = []
    _op_priority = 10.0
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __long__ = __int__
    def __neg__(self):
        return Mul(S.NegativeOne, self)
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return Mul(self, other)
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return Mul(other, self)
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rdiv__')
    def __div__(self, other):
        return Mul(self, Pow(other, S.NegativeOne))
    @property
    def is_number(self):
        """Returns True if 'self' has no free symbols.
        It will be faster than `if not self.free_symbols`, however, since
        `is_number` will fail as soon as it hits a free symbol.

        Examples
        ========

        >>> from sympy import log, Integral
        >>> from sympy.abc import x

        >>> x.is_number
        False
        >>> (2*x).is_number
        False
        >>> (2 + log(2)).is_number
        True
        >>> (2 + Integral(2, x)).is_number
        False
        >>> (2 + Integral(2, (x, 1, 2))).is_number
        True

        """
        return all(obj.is_number for obj in self.args)
    def _eval_is_negative(self):
        from sympy.polys.numberfields import minimal_polynomial
        from sympy.polys.polyerrors import NotAlgebraic
        if self.is_number:
            if self.is_real is False:
                return False
            try:
                # check to see that we can get a value
                n2 = self._eval_evalf(2)
                if n2 is None:
                    raise AttributeError
                if n2._prec == 1:  # no significance
                    raise AttributeError
                if n2 == S.NaN:
                    raise AttributeError
            except (AttributeError, ValueError):
                return None
            n, i = self.evalf(2).as_real_imag()
            if not i.is_Number or not n.is_Number:
                return False
            if n._prec != 1 and i._prec != 1:
                return bool(not i and n < 0)
            elif n._prec == 1 and (not i or i._prec == 1) and \
                    self.is_algebraic and not self.has(Function):
                try:
                    if minimal_polynomial(self).is_Symbol:
                        return False
                except (NotAlgebraic, NotImplementedError):
                    pass
    def _eval_power(self, other):
        # subclass to compute self**other for cases when
        # other is not NaN, 0, or 1
        return None
    def as_real_imag(self, deep=True, **hints):
        """Performs complex expansion on 'self' and returns a tuple
           containing collected both real and imaginary parts. This
           method can't be confused with re() and im() functions,
           which does not perform complex expansion at evaluation.

           However it is possible to expand both re() and im()
           functions and get exactly the same results as with
           a single call to this function.

           >>> from sympy import symbols, I

           >>> x, y = symbols('x,y', real=True)

           >>> (x + y*I).as_real_imag()
           (x, y)

           >>> from sympy.abc import z, w

           >>> (z + w*I).as_real_imag()
           (re(z) - im(w), re(w) + im(z))

        """
        from sympy import im, re
        if hints.get('ignore') == self:
            return None
        else:
            return (re(self), im(self))
    def as_base_exp(self):
        # a -> b ** e
        return self, S.One
    def extract_multiplicatively(self, c):
        """Return None if it's not possible to make self in the form
           c * something in a nice way, i.e. preserving the properties
           of arguments of self.

           >>> from sympy import symbols, Rational

           >>> x, y = symbols('x,y', real=True)

           >>> ((x*y)**3).extract_multiplicatively(x**2 * y)
           x*y**2

           >>> ((x*y)**3).extract_multiplicatively(x**4 * y)

           >>> (2*x).extract_multiplicatively(2)
           x

           >>> (2*x).extract_multiplicatively(3)

           >>> (Rational(1, 2)*x).extract_multiplicatively(3)
           x/6

        """
        from .function import _coeff_isneg

        c = sympify(c)
        if self is S.NaN:
            return None
        if c is S.One:
            return self
        elif c == self:
            return S.One

        if c.is_Add:
            cc, pc = c.primitive()
            if cc is not S.One:
                c = Mul(cc, pc, evaluate=False)

        if c.is_Mul:
            a, b = c.as_two_terms()
            x = self.extract_multiplicatively(a)
            if x is not None:
                return x.extract_multiplicatively(b)

        quotient = self / c
        if self.is_Number:
            if self is S.Infinity:
                if c.is_positive:
                    return S.Infinity
            elif self is S.NegativeInfinity:
                if c.is_negative:
                    return S.Infinity
                elif c.is_positive:
                    return S.NegativeInfinity
            elif self is S.ComplexInfinity:
                if not c.is_zero:
                    return S.ComplexInfinity
            elif self.is_Integer:
                if not quotient.is_Integer:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Rational:
                if not quotient.is_Rational:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Float:
                if not quotient.is_Float:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
        elif self.is_NumberSymbol or self.is_Symbol or self is S.ImaginaryUnit:
            if quotient.is_Mul and len(quotient.args) == 2:
                if quotient.args[0].is_Integer and quotient.args[0].is_positive and quotient.args[1] == self:
                    return quotient
            elif quotient.is_Integer and c.is_Number:
                return quotient
        elif self.is_Add:
            cs, ps = self.primitive()
            # assert cs >= 1
            if c.is_Number and c is not S.NegativeOne:
                # assert c != 1 (handled at top)
                if cs is not S.One:
                    if c.is_negative:
                        xc = -(cs.extract_multiplicatively(-c))
                    else:
                        xc = cs.extract_multiplicatively(c)
                    if xc is not None:
                        return xc*ps  # rely on 2-arg Mul to restore Add
                return  # |c| != 1 can only be extracted from cs
            if c == ps:
                return cs
            # check args of ps
            newargs = []
            for arg in ps.args:
                newarg = arg.extract_multiplicatively(c)
                if newarg is None:
                    return  # all or nothing
                newargs.append(newarg)
            # args should be in same order so use unevaluated return
            if cs is not S.One:
                return Add._from_args([cs*t for t in newargs])
            else:
                return Add._from_args(newargs)
        elif self.is_Mul:
            args = list(self.args)
            for i, arg in enumerate(args):
                newarg = arg.extract_multiplicatively(c)
                if newarg is not None:
                    args[i] = newarg
                    return Mul(*args)
        elif self.is_Pow:
            if c.is_Pow and c.base == self.base:
                new_exp = self.exp.extract_additively(c.exp)
                if new_exp is not None:
                    return self.base ** (new_exp)
            elif c == self.base:
                new_exp = self.exp.extract_additively(1)
                if new_exp is not None:
                    return self.base ** (new_exp)
    def extract_additively(self, c):
        """Return self - c if it's possible to subtract c from self and
        make all matching coefficients move towards zero, else return None.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> e = 2*x + 3
        >>> e.extract_additively(x + 1)
        x + 2
        >>> e.extract_additively(3*x)
        >>> e.extract_additively(4)
        >>> (y*(x + 1)).extract_additively(x + 1)
        >>> ((x + 1)*(x + 2*y + 1) + 3).extract_additively(x + 1)
        (x + 1)*(x + 2*y) + 3

        Sometimes auto-expansion will return a less simplified result
        than desired; gcd_terms might be used in such cases:

        >>> from sympy import gcd_terms
        >>> (4*x*(y + 1) + y).extract_additively(x)
        4*x*(y + 1) + x*(4*y + 3) - x*(4*y + 4) + y
        >>> gcd_terms(_)
        x*(4*y + 3) + y

        See Also
        ========
        extract_multiplicatively
        coeff
        as_coefficient

        """

        c = sympify(c)
        if self is S.NaN:
            return None
        if c is S.Zero:
            return self
        elif c == self:
            return S.Zero
        elif self is S.Zero:
            return None

        if self.is_Number:
            if not c.is_Number:
                return None
            co = self
            diff = co - c
            # XXX should we match types? i.e should 3 - .1 succeed?
            if (co > 0 and diff > 0 and diff < co or
                    co < 0 and diff < 0 and diff > co):
                return diff
            return None

        if c.is_Number:
            co, t = self.as_coeff_Add()
            xa = co.extract_additively(c)
            if xa is None:
                return None
            return xa + t

        # handle the args[0].is_Number case separately
        # since we will have trouble looking for the coeff of
        # a number.
        if c.is_Add and c.args[0].is_Number:
            # whole term as a term factor
            co = self.coeff(c)
            xa0 = (co.extract_additively(1) or 0)*c
            if xa0:
                diff = self - co*c
                return (xa0 + (diff.extract_additively(c) or diff)) or None
            # term-wise
            h, t = c.as_coeff_Add()
            sh, st = self.as_coeff_Add()
            xa = sh.extract_additively(h)
            if xa is None:
                return None
            xa2 = st.extract_additively(t)
            if xa2 is None:
                return None
            return xa + xa2

        # whole term as a term factor
        co = self.coeff(c)
        xa0 = (co.extract_additively(1) or 0)*c
        if xa0:
            diff = self - co*c
            return (xa0 + (diff.extract_additively(c) or diff)) or None
        # term-wise
        coeffs = []
        for a in Add.make_args(c):
            ac, at = a.as_coeff_Mul()
            co = self.coeff(at)
            if not co:
                return None
            coc, cot = co.as_coeff_Add()
            xa = coc.extract_additively(ac)
            if xa is None:
                return None
            self -= co*at
            coeffs.append((cot + xa)*at)
        coeffs.append(self)
        return Add(*coeffs)
    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product. """
        return S.One, self