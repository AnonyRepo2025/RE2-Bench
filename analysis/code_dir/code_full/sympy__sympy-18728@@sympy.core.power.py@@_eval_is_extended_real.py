def getit(self):
    try:
        return self._assumptions[fact]
    except KeyError:
        if self._assumptions is self.default_assumptions:
            self._assumptions = self.default_assumptions.copy()
        return _ask(fact, self)

def func(self):
    return self.__class__

def _ask(fact, obj):
    assumptions = obj._assumptions
    handler_map = obj._prop_handler
    assumptions._tell(fact, None)
    try:
        evaluate = handler_map[fact]
    except KeyError:
        pass
    else:
        a = evaluate(obj)
        if a is not None:
            assumptions.deduce_all_facts(((fact, a),))
            return a
    prereq = list(_assume_rules.prereq[fact])
    shuffle(prereq)
    for pk in prereq:
        if pk in assumptions:
            continue
        if pk in handler_map:
            _ask(pk, obj)
            ret_val = assumptions.get(fact)
            if ret_val is not None:
                return ret_val
    return None

def _tell(self, k, v):
    if k in self and self[k] is not None:
        if self[k] == v:
            return False
        else:
            raise InconsistentAssumptions(self, k, v)
    else:
        self[k] = v
        return True

def _eval_is_extended_positive(self):
    return self._eval_is_extended_positive_negative(positive=True)

def _eval_is_extended_positive_negative(self, positive):
    from sympy.polys.numberfields import minimal_polynomial
    from sympy.polys.polyerrors import NotAlgebraic
    if self.is_number:
        if self.is_extended_real is False:
            return False
        try:
            n2 = self._eval_evalf(2)
        except ValueError:
            return None
        if n2 is None:
            return None
        if getattr(n2, '_prec', 1) == 1:
            return None
        if n2 is S.NaN:
            return None
        r, i = self.evalf(2).as_real_imag()
        if not i.is_Number or not r.is_Number:
            return False
        if r._prec != 1 and i._prec != 1:
            return bool(not i and (r > 0 if positive else r < 0))
        elif r._prec == 1 and (not i or i._prec == 1) and self.is_algebraic and (not self.has(Function)):
            try:
                if minimal_polynomial(self).is_Symbol:
                    return False
            except (NotAlgebraic, NotImplementedError):
                pass

def _eval_evalf(self, prec):
    return Float._new(self._as_mpf_val(prec), prec)

def _as_mpf_val(self, prec):
    return mlib.from_int(self.p, prec, rnd)

def _new(cls, _mpf_, _prec, zero=True):
    if zero and _mpf_ == fzero:
        return S.Zero
    elif _mpf_ == _mpf_nan:
        return S.NaN
    elif _mpf_ == _mpf_inf:
        return S.Infinity
    elif _mpf_ == _mpf_ninf:
        return S.NegativeInfinity
    obj = Expr.__new__(cls)
    obj._mpf_ = mpf_norm(_mpf_, _prec)
    obj._prec = _prec
    return obj

def __new__(cls, *args):
    obj = object.__new__(cls)
    obj._assumptions = cls.default_assumptions
    obj._mhash = None
    obj._args = args
    return obj

def mpf_norm(mpf, prec):
    sign, man, expt, bc = mpf
    if not man:
        if not bc:
            return fzero
        else:
            return mpf
    from mpmath.libmp.backend import MPZ
    rv = mpf_normalize(sign, MPZ(man), expt, bc, prec, rnd)
    return rv

def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
    from sympy import Float, Number
    n = n if n is not None else 15
    if subs and is_sequence(subs):
        raise TypeError('subs must be given as a dictionary')
    if n == 1 and isinstance(self, Number):
        from sympy.core.expr import _mag
        rv = self.evalf(2, subs, maxn, chop, strict, quad, verbose)
        m = _mag(rv)
        rv = rv.round(1 - m)
        return rv
    if not evalf_table:
        _create_evalf_table()
    prec = dps_to_prec(n)
    options = {'maxprec': max(prec, int(maxn * LG10)), 'chop': chop, 'strict': strict, 'verbose': verbose}
    if subs is not None:
        options['subs'] = subs
    if quad is not None:
        options['quad'] = quad
    try:
        result = evalf(self, prec + 4, options)
    except NotImplementedError:
        v = self._eval_evalf(prec)
        if v is None:
            return self
        elif not v.is_number:
            return v
        try:
            result = evalf(v, prec, options)
        except NotImplementedError:
            return v
    re, im, re_acc, im_acc = result
    if re:
        p = max(min(prec, re_acc), 1)
        re = Float._new(re, p)
    else:
        re = S.Zero
    if im:
        p = max(min(prec, im_acc), 1)
        im = Float._new(im, p)
        return re + im * S.ImaginaryUnit
    else:
        return re

def evalf(x, prec, options):
    from sympy import re as re_, im as im_
    try:
        rf = evalf_table[x.func]
        r = rf(x, prec, options)
    except KeyError:
        if 'subs' in options:
            x = x.subs(evalf_subs(prec, options['subs']))
        xe = x._eval_evalf(prec)
        if xe is None:
            raise NotImplementedError
        as_real_imag = getattr(xe, 'as_real_imag', None)
        if as_real_imag is None:
            raise NotImplementedError
        re, im = as_real_imag()
        if re.has(re_) or im.has(im_):
            raise NotImplementedError
        if re == 0:
            re = None
            reprec = None
        elif re.is_number:
            re = re._to_mpmath(prec, allow_ints=False)._mpf_
            reprec = prec
        else:
            raise NotImplementedError
        if im == 0:
            im = None
            imprec = None
        elif im.is_number:
            im = im._to_mpmath(prec, allow_ints=False)._mpf_
            imprec = prec
        else:
            raise NotImplementedError
        r = (re, im, reprec, imprec)
    if options.get('verbose'):
        print('### input', x)
        print('### output', to_str(r[0] or fzero, 50))
        print('### raw', r)
        print()
    chop = options.get('chop', False)
    if chop:
        if chop is True:
            chop_prec = prec
        else:
            chop_prec = int(round(-3.321 * math.log10(chop) + 2.5))
            if chop_prec == 3:
                chop_prec -= 1
        r = chop_parts(r, chop_prec)
    if options.get('strict'):
        check_target(x, r, prec)
    return r

def as_real_imag(self, deep=True, **hints):
    from sympy import im, re
    if hints.get('ignore') == self:
        return None
    else:
        return (re(self), im(self))

def __eq__(self, other):
    try:
        other = _sympify(other)
    except SympifyError:
        return NotImplemented
    if not self:
        return not other
    if other.is_NumberSymbol:
        if other.is_irrational:
            return False
        return other.__eq__(self)
    if other.is_Float:
        return self._mpf_ == other._mpf_
    if other.is_Rational:
        return other.__eq__(self)
    if other.is_Number:
        ompf = other._as_mpf_val(self._prec)
        return bool(mlib.mpf_eq(self._mpf_, ompf))
    return False



from __future__ import print_function, division
from math import log as _log
from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import _coeff_isneg, expand_complex, expand_multinomial, expand_mul
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and
from .compatibility import as_int, HAS_GMPY, gmpy
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from mpmath.libmp import sqrtrem as mpmath_sqrtrem
from math import sqrt as _sqrt
from .add import Add
from .numbers import Integer
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
from sympy.functions.elementary.exponential import exp_polar
from sympy.core.relational import Relational
from sympy.assumptions.ask import ask, Q
from sympy import arg, exp, floor, im, log, re, sign
from sympy.ntheory import totient
from .mod import Mod
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
from sympy import exp, log, I, arg
from sympy import ceiling, collect, exp, log, O, Order, powsimp
from sympy import exp, log
from sympy import binomial
from sympy import multinomial_coefficients
from sympy.polys.polyutils import basic_from_dict
from sympy import O

class Pow(Expr):
    is_Pow = True
    __slots__ = ('is_commutative',)

    @property
    def base(self):
        return self._args[0]

    @property
    def exp(self):
        return self._args[1]

    def _eval_is_extended_real(self):
        from sympy import arg, exp, log, Mul
        real_b = self.base.is_extended_real
        if real_b is None:
            if self.base.func == exp and self.base.args[0].is_imaginary:
                return self.exp.is_imaginary
            return
        real_e = self.exp.is_extended_real
        if real_e is None:
            return
        if real_b and real_e:
            if self.base.is_extended_positive:
                return True
            elif self.base.is_extended_nonnegative and self.exp.is_extended_nonnegative:
                return True
            elif self.exp.is_integer and self.base.is_extended_nonzero:
                return True
            elif self.exp.is_integer and self.exp.is_nonnegative:
                return True
            elif self.base.is_extended_negative:
                if self.exp.is_Rational:
                    return False
        if real_e and self.exp.is_extended_negative and (self.base.is_zero is False):
            return Pow(self.base, -self.exp).is_extended_real
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
                    return Mul(self.base ** c, self.base ** a, evaluate=False).is_extended_real
            elif self.base in (-S.ImaginaryUnit, S.ImaginaryUnit):
                if (self.exp / 2).is_integer is False:
                    return False
        if real_b and im_e:
            if self.base is S.NegativeOne:
                return True
            c = self.exp.coeff(S.ImaginaryUnit)
            if c:
                if self.base.is_rational and c.is_rational:
                    if self.base.is_nonzero and (self.base - 1).is_nonzero and c.is_nonzero:
                        return False
                ok = (c * log(self.base) / S.Pi).is_integer
                if ok is not None:
                    return ok
        if real_b is False:
            i = arg(self.base) * self.exp / S.Pi
            if i.is_complex:
                return i.is_integer