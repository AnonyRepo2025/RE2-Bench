from __future__ import print_function, division
from math import log as _log
from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (_coeff_isneg, expand_complex, expand_multinomial,
    expand_mul)
from .logic import fuzzy_bool, fuzzy_not
from .compatibility import as_int, range
from .evaluate import global_evaluate
from sympy.utilities.iterables import sift
from mpmath.libmp import sqrtrem as mpmath_sqrtrem
from math import sqrt as _sqrt
from .add import Add
from .numbers import Integer
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
from sympy.functions.elementary.exponential import exp_polar
from sympy.assumptions.ask import ask, Q
from sympy import Abs, arg, exp, floor, im, log, re, sign
from sympy import log
from sympy import arg, exp, log, Mul
from sympy import arg, log
from sympy import exp, log, Symbol
from sympy.functions.elementary.complexes import adjoint
from sympy.functions.elementary.complexes import conjugate as c
from sympy.functions.elementary.complexes import transpose
from sympy import atan2, cos, im, re, sin
from sympy.polys.polytools import poly
from sympy import log
from sympy import ceiling, collect, exp, log, O, Order, powsimp
from sympy import exp, log
from sympy import binomial
from sympy import multinomial_coefficients
from sympy.polys.polyutils import basic_from_dict
from sympy import O
from sympy import numer, denom, log, sign, im, factor_terms



class Pow(Expr):
    is_Pow = True
    __slots__ = ['is_commutative']
    @cacheit
    def __new__(cls, b, e, evaluate=None):
        if evaluate is None:
            evaluate = global_evaluate[0]
        from sympy.functions.elementary.exponential import exp_polar

        b = _sympify(b)
        e = _sympify(e)
        if evaluate:
            if e is S.Zero:
                return S.One
            elif e is S.One:
                return b
            # Only perform autosimplification if exponent or base is a Symbol or number
            elif (b.is_Symbol or b.is_number) and (e.is_Symbol or e.is_number) and\
                e.is_integer and _coeff_isneg(b):
                if e.is_even:
                    b = -b
                elif e.is_odd:
                    return -Pow(-b, e)
            if S.NaN in (b, e):  # XXX S.NaN**x -> S.NaN under assumption that x != 0
                return S.NaN
            elif b is S.One:
                if abs(e).is_infinite:
                    return S.NaN
                return S.One
            else:
                # recognize base as E
                if not e.is_Atom and b is not S.Exp1 and b.func is not exp_polar:
                    from sympy import numer, denom, log, sign, im, factor_terms
                    c, ex = factor_terms(e, sign=False).as_coeff_Mul()
                    den = denom(ex)
                    if den.func is log and den.args[0] == b:
                        return S.Exp1**(c*numer(ex))
                    elif den.is_Add:
                        s = sign(im(b))
                        if s.is_Number and s and den == \
                                log(-factor_terms(b, sign=False)) + s*S.ImaginaryUnit*S.Pi:
                            return S.Exp1**(c*numer(ex))

                obj = b._eval_power(e)
                if obj is not None:
                    return obj
        obj = Expr.__new__(cls, b, e)
        obj.is_commutative = (b.is_commutative and e.is_commutative)
        return obj
    @property
    def base(self):
        return self._args[0]
    @property
    def exp(self):
        return self._args[1]
    def _eval_is_even(self):
        if self.exp.is_integer and self.exp.is_positive:
            return self.base.is_even
    def _eval_is_positive(self):
        from sympy import log
        if self.base == self.exp:
            if self.base.is_nonnegative:
                return True
        elif self.base.is_positive:
            if self.exp.is_real:
                return True
        elif self.base.is_negative:
            if self.exp.is_even:
                return True
            if self.exp.is_odd:
                return False
        elif self.base.is_nonpositive:
            if self.exp.is_odd:
                return False
        elif self.base.is_imaginary:
            if self.exp.is_integer:
                m = self.exp % 4
                if m.is_zero:
                    return True
                if m.is_integer and m.is_zero is False:
                    return False
            if self.exp.is_imaginary:
                return log(self.base).is_imaginary
    def _eval_is_negative(self):
        if self.base.is_negative:
            if self.exp.is_odd:
                return True
            if self.exp.is_even:
                return False
        elif self.base.is_positive:
            if self.exp.is_real:
                return False
        elif self.base.is_nonnegative:
            if self.exp.is_nonnegative:
                return False
        elif self.base.is_nonpositive:
            if self.exp.is_even:
                return False
        elif self.base.is_real:
            if self.exp.is_even:
                return False
    def _eval_is_zero(self):
        if self.base.is_zero:
            if self.exp.is_positive:
                return True
            elif self.exp.is_nonpositive:
                return False
        elif self.base.is_zero is False:
            if self.exp.is_finite:
                return False
            elif self.exp.is_infinite:
                if (1 - abs(self.base)).is_positive:
                    return self.exp.is_positive
                elif (1 - abs(self.base)).is_negative:
                    return self.exp.is_negative
        else:
            # when self.base.is_zero is None
            return None
    def _eval_is_integer(self):
        b, e = self.args
        if b.is_rational:
            if b.is_integer is False and e.is_positive:
                return False  # rat**nonneg
        if b.is_integer and e.is_integer:
            if b is S.NegativeOne:
                return True
            if e.is_nonnegative or e.is_positive:
                return True
        if b.is_integer and e.is_negative and (e.is_finite or e.is_integer):
            if fuzzy_not((b - 1).is_zero) and fuzzy_not((b + 1).is_zero):
                return False
        if b.is_Number and e.is_Number:
            check = self.func(*self.args)
            return check.is_Integer
    def _eval_is_real(self):
        from sympy import arg, exp, log, Mul
        real_b = self.base.is_real
        if real_b is None:
            if self.base.func == exp and self.base.args[0].is_imaginary:
                return self.exp.is_imaginary
            return
        real_e = self.exp.is_real
        if real_e is None:
            return
        if real_b and real_e:
            if self.base.is_positive:
                return True
            elif self.base.is_nonnegative:
                if self.exp.is_nonnegative:
                    return True
            else:
                if self.exp.is_integer:
                    return True
                elif self.base.is_negative:
                    if self.exp.is_Rational:
                        return False
        if real_e and self.exp.is_negative:
            return Pow(self.base, -self.exp).is_real
        im_b = self.base.is_imaginary
        im_e = self.exp.is_imaginary
        if im_b:
            if self.exp.is_integer:
                if self.exp.is_even:
                    return True
                elif self.exp.is_odd:
                    return False
            elif im_e and log(self.base).is_imaginary:
                return True
            elif self.exp.is_Add:
                c, a = self.exp.as_coeff_Add()
                if c and c.is_Integer:
                    return Mul(
                        self.base**c, self.base**a, evaluate=False).is_real
            elif self.base in (-S.ImaginaryUnit, S.ImaginaryUnit):
                if (self.exp/2).is_integer is False:
                    return False
        if real_b and im_e:
            if self.base is S.NegativeOne:
                return True
            c = self.exp.coeff(S.ImaginaryUnit)
            if c:
                ok = (c*log(self.base)/S.Pi).is_Integer
                if ok is not None:
                    return ok

        if real_b is False:  # we already know it's not imag
            i = arg(self.base)*self.exp/S.Pi
            return i.is_integer
    def _eval_is_complex(self):
        if all(a.is_complex for a in self.args):
            return True
    def _eval_is_imaginary(self):
        from sympy import arg, log
        if self.base.is_imaginary:
            if self.exp.is_integer:
                odd = self.exp.is_odd
                if odd is not None:
                    return odd
                return

        if self.exp.is_imaginary:
            imlog = log(self.base).is_imaginary
            if imlog is not None:
                return False  # I**i -> real; (2*I)**i -> complex ==> not imaginary

        if self.base.is_real and self.exp.is_real:
            if self.base.is_positive:
                return False
            else:
                rat = self.exp.is_rational
                if not rat:
                    return rat
                if self.exp.is_integer:
                    return False
                else:
                    half = (2*self.exp).is_integer
                    if half:
                        return self.base.is_negative
                    return half

        if self.base.is_real is False:  # we already know it's not imag
            i = arg(self.base)*self.exp/S.Pi
            isodd = (2*i).is_odd
            if isodd is not None:
                return isodd

        if self.exp.is_negative:
            return (1/self).is_imaginary
    def _eval_is_odd(self):
        if self.exp.is_integer:
            if self.exp.is_positive:
                return self.base.is_odd
            elif self.exp.is_nonnegative and self.base.is_odd:
                return True
            elif self.base is S.NegativeOne:
                return True
    def _eval_is_finite(self):
        if self.exp.is_negative:
            if self.base.is_zero:
                return False
            if self.base.is_infinite:
                return True
        c1 = self.base.is_finite
        if c1 is None:
            return
        c2 = self.exp.is_finite
        if c2 is None:
            return
        if c1 and c2:
            if self.exp.is_nonnegative or fuzzy_not(self.base.is_zero):
                return True
    def _eval_is_prime(self):
        if self.exp == S.One:
            return self.base.is_prime
        if self.is_number:
            return self.doit().is_prime

        if self.is_integer and self.is_positive:
            """
            a Power will be non-prime only if both base and exponent
            are greater than 1
            """
            if (self.base-1).is_positive or (self.exp-1).is_positive:
                return False
    def as_base_exp(self):
        """Return base and exp of self.

        If base is 1/Integer, then return Integer, -exp. If this extra
        processing is not needed, the base and exp properties will
        give the raw arguments

        Examples
        ========

        >>> from sympy import Pow, S
        >>> p = Pow(S.Half, 2, evaluate=False)
        >>> p.as_base_exp()
        (2, -2)
        >>> p.args
        (1/2, 2)

        """

        b, e = self.args
        if b.is_Rational and b.p == 1 and b.q != 1:
            return Integer(b.q), -e
        return b, e
    def _eval_expand_power_base(self, **hints):
        """(a*b)**n -> a**n * b**n"""
        force = hints.get('force', False)

        b = self.base
        e = self.exp
        if not b.is_Mul:
            return self

        cargs, nc = b.args_cnc(split_1=False)

        # expand each term - this is top-level-only
        # expansion but we have to watch out for things
        # that don't have an _eval_expand method
        if nc:
            nc = [i._eval_expand_power_base(**hints)
                if hasattr(i, '_eval_expand_power_base') else i
                for i in nc]

            if e.is_Integer:
                if e.is_positive:
                    rv = Mul(*nc*e)
                else:
                    rv = 1/Mul(*nc*-e)
                if cargs:
                    rv *= Mul(*cargs)**e
                return rv

            if not cargs:
                return self.func(Mul(*nc), e, evaluate=False)

            nc = [Mul(*nc)]

        # sift the commutative bases
        sifted = sift(cargs, lambda x: x.is_real)
        maybe_real = sifted[True] + sifted[None]
        other = sifted[False]
        def pred(x):
            if x is S.ImaginaryUnit:
                return S.ImaginaryUnit
            polar = x.is_polar
            if polar:
                return True
            if polar is None:
                return fuzzy_bool(x.is_nonnegative)
        sifted = sift(maybe_real, pred)
        nonneg = sifted[True]
        other += sifted[None]
        neg = sifted[False]
        imag = sifted[S.ImaginaryUnit]
        if imag:
            I = S.ImaginaryUnit
            i = len(imag) % 4
            if i == 0:
                pass
            elif i == 1:
                other.append(I)
            elif i == 2:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
            else:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
                other.append(I)
            del imag

        # bring out the bases that can be separated from the base

        if force or e.is_integer:
            # treat all commutatives the same and put nc in other
            cargs = nonneg + neg + other
            other = nc
        else:
            # this is just like what is happening automatically, except
            # that now we are doing it for an arbitrary exponent for which
            # no automatic expansion is done

            assert not e.is_Integer

            # handle negatives by making them all positive and putting
            # the residual -1 in other
            if len(neg) > 1:
                o = S.One
                if not other and neg[0].is_Number:
                    o *= neg.pop(0)
                if len(neg) % 2:
                    o = -o
                for n in neg:
                    nonneg.append(-n)
                if o is not S.One:
                    other.append(o)
            elif neg and other:
                if neg[0].is_Number and neg[0] is not S.NegativeOne:
                    other.append(S.NegativeOne)
                    nonneg.append(-neg[0])
                else:
                    other.extend(neg)
            else:
                other.extend(neg)
            del neg

            cargs = nonneg
            other += nc

        rv = S.One
        if cargs:
            rv *= Mul(*[self.func(b, e, evaluate=False) for b in cargs])
        if other:
            rv *= self.func(Mul(*other), e, evaluate=False)
        return rv
    def _eval_is_rational(self):
        p = self.func(*self.as_base_exp())  # in case it's unevaluated
        if not p.is_Pow:
            return p.is_rational
        b, e = p.as_base_exp()
        if e.is_Rational and b.is_Rational:
            # we didn't check that e is not an Integer
            # because Rational**Integer autosimplifies
            return False
        if e.is_integer:
            if b.is_rational:
                if fuzzy_not(b.is_zero) or e.is_nonnegative:
                    return True
                if b == e:  # always rational, even for 0**0
                    return True
            elif b.is_irrational:
                return e.is_zero
    def _eval_is_algebraic(self):
        if self.base.is_zero or (self.base - 1).is_zero:
            return True
        elif self.exp.is_rational:
            if self.base.is_algebraic is False:
                return self.exp.is_nonzero
            return self.base.is_algebraic
        elif self.base.is_algebraic and self.exp.is_algebraic:
            if ((fuzzy_not(self.base.is_zero)
                and fuzzy_not((self.base - 1).is_zero))
                or self.base.is_integer is False
                or self.base.is_irrational):
                return self.exp.is_rational