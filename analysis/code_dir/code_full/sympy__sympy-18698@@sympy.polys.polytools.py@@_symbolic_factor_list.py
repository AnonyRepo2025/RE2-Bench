def make_args(cls, expr):
    if isinstance(expr, cls):
        return expr.args
    else:
        return (sympify(expr),)

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    is_sympy = getattr(a, '__sympy__', None)
    if is_sympy is not None:
        return a
    if isinstance(a, CantSympify):
        raise SympifyError(a)
    cls = getattr(a, '__class__', None)
    if cls is None:
        cls = type(a)
    conv = converter.get(cls, None)
    if conv is not None:
        return conv(a)
    for superclass in getmro(cls):
        try:
            return converter[superclass](a)
        except KeyError:
            continue
    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a
    if evaluate is None:
        evaluate = global_parameters.evaluate
    if type(a).__module__ == 'numpy':
        import numpy as np
        if np.isscalar(a):
            return _convert_numpy_types(a, locals=locals, convert_xor=convert_xor, strict=strict, rational=rational, evaluate=evaluate)
    _sympy_ = getattr(a, '_sympy_', None)
    if _sympy_ is not None:
        try:
            return a._sympy_()
        except AttributeError:
            pass
    if not strict:
        flat = getattr(a, 'flat', None)
        if flat is not None:
            shape = getattr(a, 'shape', None)
            if shape is not None:
                from ..tensor.array import Array
                return Array(a.flat, a.shape)
    if not isinstance(a, str):
        for coerce in (float, int):
            try:
                coerced = coerce(a)
            except (TypeError, ValueError):
                continue
            try:
                return sympify(coerced)
            except SympifyError:
                continue
    if strict:
        raise SympifyError(a)
    if iterable(a):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor, rational=rational) for x in a])
        except TypeError:
            pass
    if isinstance(a, dict):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor, rational=rational) for x in a.items()])
        except TypeError:
            pass
    try:
        from .compatibility import unicode
        a = unicode(a)
    except Exception as exc:
        raise SympifyError(a, exc)
    from sympy.parsing.sympy_parser import parse_expr, TokenError, standard_transformations
    from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
    from sympy.parsing.sympy_parser import rationalize as t_rationalize
    transformations = standard_transformations
    if rational:
        transformations += (t_rationalize,)
    if convert_xor:
        transformations += (t_convert_xor,)
    try:
        a = a.replace('\n', '')
        expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
    except (TokenError, SyntaxError) as exc:
        raise SympifyError('could not parse %r' % a, exc)
    return expr

def _poly_from_expr(expr, opt):
    orig, expr = (expr, sympify(expr))
    if not isinstance(expr, Basic):
        raise PolificationFailed(opt, orig, expr)
    elif expr.is_Poly:
        poly = expr.__class__._from_poly(expr, opt)
        opt.gens = poly.gens
        opt.domain = poly.domain
        if opt.polys is None:
            opt.polys = True
        return (poly, opt)
    elif opt.expand:
        expr = expr.expand()
    rep, opt = _dict_from_expr(expr, opt)
    if not opt.gens:
        raise PolificationFailed(opt, orig, expr)
    monoms, coeffs = list(zip(*list(rep.items())))
    domain = opt.domain
    if domain is None:
        opt.domain, coeffs = construct_domain(coeffs, opt=opt)
    else:
        coeffs = list(map(domain.from_sympy, coeffs))
    rep = dict(list(zip(monoms, coeffs)))
    poly = Poly._from_dict(rep, opt)
    if opt.polys is None:
        opt.polys = False
    return (poly, opt)

def getter(self):
    try:
        return self[cls.option]
    except KeyError:
        return cls.default()

def default(cls):
    return True

def expand(self, deep=True, modulus=None, power_base=True, power_exp=True, mul=True, log=True, multinomial=True, basic=True, **hints):
    from sympy.simplify.radsimp import fraction
    hints.update(power_base=power_base, power_exp=power_exp, mul=mul, log=log, multinomial=multinomial, basic=basic)
    expr = self
    if hints.pop('frac', False):
        n, d = [a.expand(deep=deep, modulus=modulus, **hints) for a in fraction(self)]
        return n / d
    elif hints.pop('denom', False):
        n, d = fraction(self)
        return n / d.expand(deep=deep, modulus=modulus, **hints)
    elif hints.pop('numer', False):
        n, d = fraction(self)
        return n.expand(deep=deep, modulus=modulus, **hints) / d

    def _expand_hint_key(hint):
        if hint == 'mul':
            return 'mulz'
        return hint
    for hint in sorted(hints.keys(), key=_expand_hint_key):
        use_hint = hints[hint]
        if use_hint:
            hint = '_eval_expand_' + hint
            expr, hit = Expr._expand_hint(expr, hint, deep=deep, **hints)
    while True:
        was = expr
        if hints.get('multinomial', False):
            expr, _ = Expr._expand_hint(expr, '_eval_expand_multinomial', deep=deep, **hints)
        if hints.get('mul', False):
            expr, _ = Expr._expand_hint(expr, '_eval_expand_mul', deep=deep, **hints)
        if hints.get('log', False):
            expr, _ = Expr._expand_hint(expr, '_eval_expand_log', deep=deep, **hints)
        if expr == was:
            break
    if modulus is not None:
        modulus = sympify(modulus)
        if not modulus.is_Integer or modulus <= 0:
            raise ValueError('modulus must be a positive integer, got %s' % modulus)
        terms = []
        for term in Add.make_args(expr):
            coeff, tail = term.as_coeff_Mul(rational=True)
            coeff %= modulus
            if coeff:
                terms.append(coeff * tail)
        expr = Add(*terms)
    return expr

def _expand_hint_key(hint):
    if hint == 'mul':
        return 'mulz'
    return hint

def _expand_hint(expr, hint, deep=True, **hints):
    hit = False
    if deep and getattr(expr, 'args', ()) and (not expr.is_Atom):
        sargs = []
        for arg in expr.args:
            arg, arghit = Expr._expand_hint(arg, hint, **hints)
            hit |= arghit
            sargs.append(arg)
        if hit:
            expr = expr.func(*sargs)
    if hasattr(expr, hint):
        newexpr = getattr(expr, hint)(**hints)
        if newexpr != expr:
            return (newexpr, True)
    return (expr, hit)

def args(self):
    return self._args

def _eval_expand_multinomial(self, **hints):
    base, exp = self.args
    result = self
    if exp.is_Rational and exp.p > 0 and base.is_Add:
        if not exp.is_Integer:
            n = Integer(exp.p // exp.q)
            if not n:
                return result
            else:
                radical, result = (self.func(base, exp - n), [])
                expanded_base_n = self.func(base, n)
                if expanded_base_n.is_Pow:
                    expanded_base_n = expanded_base_n._eval_expand_multinomial()
                for term in Add.make_args(expanded_base_n):
                    result.append(term * radical)
                return Add(*result)
        n = int(exp)
        if base.is_commutative:
            order_terms, other_terms = ([], [])
            for b in base.args:
                if b.is_Order:
                    order_terms.append(b)
                else:
                    other_terms.append(b)
            if order_terms:
                f = Add(*other_terms)
                o = Add(*order_terms)
                if n == 2:
                    return expand_multinomial(f ** n, deep=False) + n * f * o
                else:
                    g = expand_multinomial(f ** (n - 1), deep=False)
                    return expand_mul(f * g, deep=False) + n * g * o
            if base.is_number:
                a, b = base.as_real_imag()
                if a.is_Rational and b.is_Rational:
                    if not a.is_Integer:
                        if not b.is_Integer:
                            k = self.func(a.q * b.q, n)
                            a, b = (a.p * b.q, a.q * b.p)
                        else:
                            k = self.func(a.q, n)
                            a, b = (a.p, a.q * b)
                    elif not b.is_Integer:
                        k = self.func(b.q, n)
                        a, b = (a * b.q, b.p)
                    else:
                        k = 1
                    a, b, c, d = (int(a), int(b), 1, 0)
                    while n:
                        if n & 1:
                            c, d = (a * c - b * d, b * c + a * d)
                            n -= 1
                        a, b = (a * a - b * b, 2 * a * b)
                        n //= 2
                    I = S.ImaginaryUnit
                    if k == 1:
                        return c + I * d
                    else:
                        return Integer(c) / k + I * d / k
            p = other_terms
            from sympy import multinomial_coefficients
            from sympy.polys.polyutils import basic_from_dict
            expansion_dict = multinomial_coefficients(len(p), n)
            return basic_from_dict(expansion_dict, *p)
        elif n == 2:
            return Add(*[f * g for f in base.args for g in base.args])
        else:
            multi = (base ** (n - 1))._eval_expand_multinomial()
            if multi.is_Add:
                return Add(*[f * g for f in base.args for g in multi.args])
            else:
                return Add(*[f * multi for f in base.args])
    elif exp.is_Rational and exp.p < 0 and base.is_Add and (abs(exp.p) > exp.q):
        return 1 / self.func(base, -exp)._eval_expand_multinomial()
    elif exp.is_Add and base.is_Number:
        coeff, tail = (S.One, S.Zero)
        for term in exp.args:
            if term.is_Number:
                coeff *= self.func(base, term)
            else:
                tail += term
        return coeff * self.func(base, tail)
    else:
        return result

def __ne__(self, other):
    return not self == other

def __eq__(self, other):
    try:
        other = _sympify(other)
        if not isinstance(other, Expr):
            return False
    except (SympifyError, SyntaxError):
        return False
    if not (self.is_Number and other.is_Number) and type(self) != type(other):
        return False
    a, b = (self._hashable_content(), other._hashable_content())
    if a != b:
        return False
    for a, b in zip(a, b):
        if not isinstance(a, Expr):
            continue
        if a.is_Number and type(a) != type(b):
            return False
    return True

def _sympify(a):
    return sympify(a, strict=True)

def _hashable_content(self):
    return self._args

def _eval_expand_power_base(self, **hints):
    force = hints.get('force', False)
    b = self.base
    e = self.exp
    if not b.is_Mul:
        return self
    cargs, nc = b.args_cnc(split_1=False)
    if nc:
        nc = [i._eval_expand_power_base(**hints) if hasattr(i, '_eval_expand_power_base') else i for i in nc]
        if e.is_Integer:
            if e.is_positive:
                rv = Mul(*nc * e)
            else:
                rv = Mul(*[i ** (-1) for i in nc[::-1]] * -e)
            if cargs:
                rv *= Mul(*cargs) ** e
            return rv
        if not cargs:
            return self.func(Mul(*nc), e, evaluate=False)
        nc = [Mul(*nc)]
    other, maybe_real = sift(cargs, lambda x: x.is_extended_real is False, binary=True)

    def pred(x):
        if x is S.ImaginaryUnit:
            return S.ImaginaryUnit
        polar = x.is_polar
        if polar:
            return True
        if polar is None:
            return fuzzy_bool(x.is_extended_nonnegative)
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
    if force or e.is_integer:
        cargs = nonneg + neg + other
        other = nc
    else:
        assert not e.is_Integer
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
        if e.is_Rational:
            npow, cargs = sift(cargs, lambda x: x.is_Pow and x.exp.is_Rational and x.base.is_number, binary=True)
            rv = Mul(*[self.func(b.func(*b.args), e) for b in npow])
        rv *= Mul(*[self.func(b, e, evaluate=False) for b in cargs])
    if other:
        rv *= self.func(Mul(*other), e, evaluate=False)
    return rv



from __future__ import print_function, division
from functools import wraps, reduce
from operator import mul
from sympy.core import S, Basic, Expr, I, Integer, Add, Mul, Dummy, Tuple
from sympy.core.basic import preorder_traversal
from sympy.core.compatibility import iterable, ordered
from sympy.core.decorators import _sympifyit
from sympy.core.function import Derivative
from sympy.core.mul import _keep_coeff
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import OperationNotSupported, DomainError, CoercionFailed, UnificationFailed, GeneratorsNeeded, PolynomialError, MultivariatePolynomialError, ExactQuotientFailed, PolificationFailed, ComputationFailed, GeneratorsError
from sympy.polys.polyutils import basic_from_dict, _sort_gens, _unify_gens, _dict_reorder, _dict_from_expr, _parallel_dict_from_expr
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, sift, public, filldedent
from sympy.utilities.exceptions import SymPyDeprecationWarning
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.relational import Equality
from sympy.simplify.simplify import simplify
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import factor_terms
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.rings import xring
from sympy.polys.dispersion import dispersionset
from sympy.polys.dispersion import dispersion
from sympy.functions.elementary.complexes import sign
from sympy.core.add import Add
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.simplify.simplify import bottom_up
from sympy.polys.rings import PolyRing
from sympy.polys.rings import xring
from sympy.polys.rings import xring
from sympy.core.numbers import ilcm
from sympy.core.exprtools import factor_nc

def _symbolic_factor_list(expr, opt, method):
    coeff, factors = (S.One, [])
    args = [i._eval_factor() if hasattr(i, '_eval_factor') else i for i in Mul.make_args(expr)]
    for arg in args:
        if arg.is_Number:
            coeff *= arg
            continue
        elif arg.is_Pow:
            base, exp = arg.args
            if base.is_Number and exp.is_Number:
                coeff *= arg
                continue
            if base.is_Number:
                factors.append((base, exp))
                continue
        else:
            base, exp = (arg, S.One)
        try:
            poly, _ = _poly_from_expr(base, opt)
        except PolificationFailed as exc:
            factors.append((exc.expr, exp))
        else:
            func = getattr(poly, method + '_list')
            _coeff, _factors = func()
            if _coeff is not S.One:
                if exp.is_Integer:
                    coeff *= _coeff ** exp
                elif _coeff.is_positive:
                    factors.append((_coeff, exp))
                else:
                    _factors.append((_coeff, S.One))
            if exp is S.One:
                factors.extend(_factors)
            elif exp.is_integer:
                factors.extend([(f, k * exp) for f, k in _factors])
            else:
                other = []
                for f, k in _factors:
                    if f.as_expr().is_positive:
                        factors.append((f, k * exp))
                    else:
                        other.append((f, k))
                factors.append((_factors_product(other), exp))
    if method == 'sqf':
        factors = [(reduce(mul, (f for f, _ in factors if _ == k)), k) for k in set((i for _, i in factors))]
    return (coeff, factors)