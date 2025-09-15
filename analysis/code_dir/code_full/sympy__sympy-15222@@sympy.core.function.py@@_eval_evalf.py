def func(self):
    return self.__class__

def args(self):
    return self._args

def _to_mpmath(self, prec, allow_ints=True):
    errmsg = 'cannot convert to mpmath number'
    if allow_ints and self.is_Integer:
        return self.p
    if hasattr(self, '_as_mpf_val'):
        return make_mpf(self._as_mpf_val(prec))
    try:
        re, im, _, _ = evalf(self, prec, {})
        if im:
            if not re:
                re = fzero
            return make_mpc((re, im))
        elif re:
            return make_mpf(re)
        else:
            return make_mpf(fzero)
    except NotImplementedError:
        v = self._eval_evalf(prec)
        if v is None:
            raise ValueError(errmsg)
        if v.is_Float:
            return make_mpf(v._mpf_)
        re, im = v.as_real_imag()
        if allow_ints and re.is_Integer:
            re = from_int(re.p)
        elif re.is_Float:
            re = re._mpf_
        else:
            raise ValueError(errmsg)
        if allow_ints and im.is_Integer:
            im = from_int(im.p)
        elif im.is_Float:
            im = im._mpf_
        else:
            raise ValueError(errmsg)
        return make_mpc((re, im))

def evalf(x, prec, options):
    from sympy import re as re_, im as im_
    try:
        rf = evalf_table[x.func]
        r = rf(x, prec, options)
    except KeyError:
        try:
            if 'subs' in options:
                x = x.subs(evalf_subs(prec, options['subs']))
            xe = x._eval_evalf(prec)
            re, im = xe.as_real_imag()
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
        except AttributeError:
            raise NotImplementedError
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

def func(self):
    return self.__class__

def evalf_add(v, prec, options):
    res = pure_complex(v)
    if res:
        h, c = res
        re, _, re_acc, _ = evalf(h, prec, options)
        im, _, im_acc, _ = evalf(c, prec, options)
        return (re, im, re_acc, im_acc)
    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)
    i = 0
    target_prec = prec
    while 1:
        options['maxprec'] = min(oldmaxprec, 2 * prec)
        terms = [evalf(arg, prec + 10, options) for arg in v.args]
        re, re_acc = add_terms([a[0::2] for a in terms if a[0]], prec, target_prec)
        im, im_acc = add_terms([a[1::2] for a in terms if a[1]], prec, target_prec)
        acc = complex_accuracy((re, im, re_acc, im_acc))
        if acc >= target_prec:
            if options.get('verbose'):
                print('ADD: wanted', target_prec, 'accurate bits, got', re_acc, im_acc)
            break
        else:
            if prec - target_prec > options['maxprec']:
                break
            prec = prec + max(10 + 2 ** i, target_prec - acc)
            i += 1
            if options.get('verbose'):
                print('ADD: restarting with prec', prec)
    options['maxprec'] = oldmaxprec
    if iszero(re, scaled=True):
        re = scaled_zero(re)
    if iszero(im, scaled=True):
        im = scaled_zero(im)
    return (re, im, re_acc, im_acc)

def pure_complex(v, or_real=False):
    h, t = v.as_coeff_Add()
    if not t:
        if or_real:
            return (h, t)
        return
    c, i = t.as_coeff_Mul()
    if i is S.ImaginaryUnit:
        return (h, c)

def as_coeff_Add(self, rational=False):
    coeff, args = (self.args[0], self.args[1:])
    if coeff.is_Number and (not rational) or coeff.is_Rational:
        return (coeff, self._new_rawargs(*args))
    return (S.Zero, self)

def _new_rawargs(self, *args, **kwargs):
    if kwargs.pop('reeval', True) and self.is_commutative is False:
        is_commutative = None
    else:
        is_commutative = self.is_commutative
    return self._from_args(args, is_commutative)

def _from_args(cls, args, is_commutative=None):
    if len(args) == 0:
        return cls.identity
    elif len(args) == 1:
        return args[0]
    obj = super(AssocOp, cls).__new__(cls, *args)
    if is_commutative is None:
        is_commutative = fuzzy_and((a.is_commutative for a in args))
    obj.is_commutative = is_commutative
    return obj

def as_coeff_Mul(self, rational=False):
    return (S.One, self)



from __future__ import print_function, division
from .add import Add
from .assumptions import ManagedProperties, _assume_defined
from .basic import Basic
from .cache import cacheit
from .compatibility import iterable, is_sequence, as_int, ordered, Iterable
from .decorators import _sympifyit
from .expr import Expr, AtomicExpr
from .numbers import Rational, Float
from .operations import LatticeOp
from .rules import Transform
from .singleton import S
from .sympify import sympify
from sympy.core.containers import Tuple, Dict
from sympy.core.logic import fuzzy_and
from sympy.core.compatibility import string_types, with_metaclass, range
from sympy.utilities import default_sort_key
from sympy.utilities.misc import filldedent
from sympy.utilities.iterables import has_dups
from sympy.core.evaluate import global_evaluate
import sys
import mpmath
import mpmath.libmp as mlib
import inspect
from collections import Counter
from sympy.core.symbol import Dummy, Symbol
from sympy import Integral, Symbol
from sympy.core.relational import Relational
from sympy.simplify.radsimp import fraction
from sympy.logic.boolalg import BooleanFunction
from sympy.utilities.misc import func_name
from sympy.core.power import Pow
from sympy.polys.rootoftools import RootOf
from sympy.sets.sets import FiniteSet
from sympy.sets.fancysets import Naturals0
from sympy.sets.sets import FiniteSet
from sympy.core.evalf import pure_complex
from sympy.sets.fancysets import Naturals0
from sympy.utilities.misc import filldedent
from sympy import Order
from sympy.sets.sets import FiniteSet
from sympy import Order
import sage.all as sage
import sage.all as sage
from sympy.sets.sets import Set, FiniteSet
from sympy.matrices.common import MatrixCommon
from sympy import Integer
from sympy.tensor.array import Array, NDimArray, derive_by_array
from sympy.utilities.misc import filldedent
from sympy import Integer
import mpmath
from sympy.core.expr import Expr
import sage.all as sage
from ..calculus.finite_diff import _as_finite_diff
from sympy.sets.sets import FiniteSet
from sympy import Symbol
from sympy.printing import StrPrinter
from inspect import signature
from sympy import oo, zoo, nan
import sympy
from sympy.core.exprtools import factor_terms
from sympy.simplify.simplify import signsimp
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from mpmath import mpf, mpc

class Function(Application, Expr):

    def _eval_evalf(self, prec):
        try:
            if isinstance(self, AppliedUndef):
                raise AttributeError
            fname = self.func.__name__
            if not hasattr(mpmath, fname):
                from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
                fname = MPMATH_TRANSLATIONS[fname]
            func = getattr(mpmath, fname)
        except (AttributeError, KeyError):
            try:
                return Float(self._imp_(*[i.evalf(prec) for i in self.args]), prec)
            except (AttributeError, TypeError, ValueError):
                return
        try:
            args = [arg._to_mpmath(prec + 5) for arg in self.args]

            def bad(m):
                from mpmath import mpf, mpc
                if isinstance(m, mpf):
                    m = m._mpf_
                    return m[1] != 1 and m[-1] == 1
                elif isinstance(m, mpc):
                    m, n = m._mpc_
                    return m[1] != 1 and m[-1] == 1 and (n[1] != 1) and (n[-1] == 1)
                else:
                    return False
            if any((bad(a) for a in args)):
                raise ValueError
        except ValueError:
            return
        with mpmath.workprec(prec):
            v = func(*args)
        return Expr._from_mpmath(v, prec)