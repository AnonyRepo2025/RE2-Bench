def simplify(expr, ratio=1.7, measure=count_ops, fu=False):
    expr = sympify(expr)
    try:
        return expr._eval_simplify(ratio=ratio, measure=measure)
    except AttributeError:
        pass
    original_expr = expr = signsimp(expr)
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.functions.special.bessel import BesselBase
    from sympy import Sum, Product
    if not isinstance(expr, Basic) or not expr.args:
        return expr
    if not isinstance(expr, (Add, Mul, Pow, ExpBase)):
        if isinstance(expr, Function) and hasattr(expr, 'inverse'):
            if len(expr.args) == 1 and len(expr.args[0].args) == 1 and isinstance(expr.args[0], expr.inverse(argindex=1)):
                return simplify(expr.args[0].args[0], ratio=ratio, measure=measure, fu=fu)
        return expr.func(*[simplify(x, ratio=ratio, measure=measure, fu=fu) for x in expr.args])

    def shorter(*choices):
        if not has_variety(choices):
            return choices[0]
        return min(choices, key=measure)
    expr = bottom_up(expr, lambda w: w.normal())
    expr = Mul(*powsimp(expr).as_content_primitive())
    _e = cancel(expr)
    expr1 = shorter(_e, _mexpand(_e).cancel())
    expr2 = shorter(together(expr, deep=True), together(expr1, deep=True))
    if ratio is S.Infinity:
        expr = expr2
    else:
        expr = shorter(expr2, expr1, expr)
    if not isinstance(expr, Basic):
        return expr
    expr = factor_terms(expr, sign=False)
    expr = hyperexpand(expr)
    expr = piecewise_fold(expr)
    if expr.has(BesselBase):
        expr = besselsimp(expr)
    if expr.has(TrigonometricFunction) and (not fu) or expr.has(HyperbolicFunction):
        expr = trigsimp(expr, deep=True)
    if expr.has(log):
        expr = shorter(expand_log(expr, deep=True), logcombine(expr))
    if expr.has(CombinatorialFunction, gamma):
        expr = combsimp(expr)
    if expr.has(Sum):
        expr = sum_simplify(expr)
    if expr.has(Product):
        expr = product_simplify(expr)
    short = shorter(powsimp(expr, combine='exp', deep=True), powsimp(expr), expr)
    short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
    if short.has(TrigonometricFunction, HyperbolicFunction, ExpBase):
        short = exptrigsimp(short, simplify=False)
    hollow_mul = Transform(lambda x: Mul(*x.args), lambda x: x.is_Mul and len(x.args) == 2 and x.args[0].is_Number and x.args[1].is_Add and x.is_commutative)
    expr = short.xreplace(hollow_mul)
    numer, denom = expr.as_numer_denom()
    if denom.is_Add:
        n, d = fraction(radsimp(1 / denom, symbolic=False, max_terms=1))
        if n is not S.One:
            expr = (numer * n).expand() / d
    if expr.could_extract_minus_sign():
        n, d = fraction(expr)
        if d != 0:
            expr = signsimp(-n / -d)
    if measure(expr) > ratio * measure(original_expr):
        expr = original_expr
    return expr

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    if evaluate is None:
        if global_evaluate[0] is False:
            evaluate = global_evaluate[0]
        else:
            evaluate = True
    try:
        if a in sympy_classes:
            return a
    except TypeError:
        pass
    try:
        cls = a.__class__
    except AttributeError:
        cls = type(a)
    if cls in sympy_classes:
        return a
    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a
    if type(a).__module__ == 'numpy':
        import numpy as np
        if np.isscalar(a):
            if not isinstance(a, np.floating):
                func = converter[complex] if np.iscomplex(a) else sympify
                return func(np.asscalar(a))
            else:
                try:
                    from sympy.core.numbers import Float
                    prec = np.finfo(a).nmant
                    a = str(list(np.reshape(np.asarray(a), (1, np.size(a)))[0]))[1:-1]
                    return Float(a, precision=prec)
                except NotImplementedError:
                    raise SympifyError('Translation for numpy float : %s is not implemented' % a)
    try:
        return converter[cls](a)
    except KeyError:
        for superclass in getmro(cls):
            try:
                return converter[superclass](a)
            except KeyError:
                continue
    if isinstance(a, CantSympify):
        raise SympifyError(a)
    try:
        return a._sympy_()
    except AttributeError:
        pass
    if not isinstance(a, string_types):
        for coerce in (float, int):
            try:
                return sympify(coerce(a))
            except (TypeError, ValueError, AttributeError, SympifyError):
                continue
    if strict:
        raise SympifyError(a)
    try:
        from ..tensor.array import Array
        return Array(a.flat, a.shape)
    except AttributeError:
        pass
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

def __hash__(self):
    h = self._mhash
    if h is None:
        h = hash((type(self).__name__,) + self._hashable_content())
        self._mhash = h
    return h

def _hashable_content(self):
    return self._args

def signsimp(expr, evaluate=None):
    if evaluate is None:
        evaluate = global_evaluate[0]
    expr = sympify(expr)
    if not isinstance(expr, Expr) or expr.is_Atom:
        return expr
    e = sub_post(sub_pre(expr))
    if not isinstance(e, Expr) or e.is_Atom:
        return e
    if e.is_Add:
        return e.func(*[signsimp(a) for a in e.args])
    if evaluate:
        e = e.xreplace({m: --m for m in e.atoms(Mul) if --m != m})
    return e

def sub_pre(e):
    reps = [a for a in e.atoms(Add) if a.could_extract_minus_sign()]
    reps.sort(key=default_sort_key)
    e = e.xreplace(dict(((a, Mul._from_args([S.NegativeOne, -a])) for a in reps)))
    if isinstance(e, Basic):
        negs = {}
        for a in sorted(e.atoms(Add), key=default_sort_key):
            if a in reps or a.could_extract_minus_sign():
                negs[a] = Mul._from_args([S.One, S.NegativeOne, -a])
        e = e.xreplace(negs)
    return e

def atoms(self, *types):
    if types:
        types = tuple([t if isinstance(t, type) else type(t) for t in types])
    else:
        types = (Atom,)
    result = set()
    for expr in preorder_traversal(self):
        if isinstance(expr, types):
            result.add(expr)
    return result

def __init__(self, node, keys=None):
    self._skip_flag = False
    self._pt = self._preorder_traversal(node, keys)

def __iter__(self):
    return self

def __next__(self):
    return next(self._pt)

def _preorder_traversal(self, node, keys):
    yield node
    if self._skip_flag:
        self._skip_flag = False
        return
    if isinstance(node, Basic):
        if not keys and hasattr(node, '_argset'):
            args = node._argset
        else:
            args = node.args
        if keys:
            if keys != True:
                args = ordered(args, keys, default=False)
            else:
                args = ordered(args)
        for arg in args:
            for subtree in self._preorder_traversal(arg, keys):
                yield subtree
    elif iterable(node):
        for item in node:
            for subtree in self._preorder_traversal(item, keys):
                yield subtree

def args(self):
    return self._args

def xreplace(self, rule):
    value, _ = self._xreplace(rule)
    return value

def _xreplace(self, rule):
    if self in rule:
        return (rule[self], True)
    elif rule:
        args = []
        changed = False
        for a in self.args:
            try:
                a_xr = a._xreplace(rule)
                args.append(a_xr[0])
                changed |= a_xr[1]
            except AttributeError:
                args.append(a)
        args = tuple(args)
        if changed:
            return (self.func(*args), True)
    return (self, False)

def sub_post(e):
    replacements = []
    for node in preorder_traversal(e):
        if isinstance(node, Mul) and node.args[0] is S.One and (node.args[1] is S.NegativeOne):
            replacements.append((node, -Mul._from_args(node.args[2:])))
    for node, replacement in replacements:
        e = e.xreplace({node: replacement})
    return e



from sympy import Order, S, log, limit, lcm_list, pi, Abs
from sympy.core.basic import Basic
from sympy.core import Add, Mul, Pow
from sympy.logic.boolalg import And
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.numbers import _sympifyit, oo
from sympy.core.sympify import _sympify
from sympy.sets.sets import Interval, Intersection, FiniteSet, Union, Complement, EmptySet
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.utilities import filldedent
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.solvers.solveset import solveset, _has_rational_power
from sympy.solvers.solveset import solveset
from sympy import simplify, lcm_list
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import TrigonometricFunction, sin, cos, csc, sec
from sympy.solvers.decompogen import decompogen
from sympy.core.relational import Relational
from sympy.solvers.solveset import solveset
from sympy.functions.elementary.miscellaneous import real_root
from sympy.solvers.decompogen import compogen
AccumBounds = AccumulationBounds

def periodicity(f, symbol, check=False):
    from sympy import simplify, lcm_list
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.trigonometric import TrigonometricFunction, sin, cos, csc, sec
    from sympy.solvers.decompogen import decompogen
    from sympy.core.relational import Relational

    def _check(orig_f, period):
        new_f = orig_f.subs(symbol, symbol + period)
        if new_f.equals(orig_f):
            return period
        else:
            raise NotImplementedError(filldedent('\n                The period of the given function cannot be verified.\n                When `%s` was replaced with `%s + %s` in `%s`, the result\n                was `%s` which was not recognized as being the same as\n                the original function.\n                So either the period was wrong or the two forms were\n                not recognized as being equal.\n                Set check=False to obtain the value.' % (symbol, symbol, period, orig_f, new_f)))
    orig_f = f
    f = simplify(orig_f)
    period = None
    if symbol not in f.free_symbols:
        return S.Zero
    if isinstance(f, Relational):
        f = f.lhs - f.rhs
    if isinstance(f, TrigonometricFunction):
        try:
            period = f.period(symbol)
        except NotImplementedError:
            pass
    if isinstance(f, Abs):
        arg = f.args[0]
        if isinstance(arg, (sec, csc, cos)):
            arg = sin(arg.args[0])
        period = periodicity(arg, symbol)
        if period is not None and isinstance(arg, sin):
            orig_f = Abs(arg)
            try:
                return _check(orig_f, period / 2)
            except NotImplementedError as err:
                if check:
                    raise NotImplementedError(err)
    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)
        if base_has_sym and (not expo_has_sym):
            period = periodicity(base, symbol)
        elif expo_has_sym and (not base_has_sym):
            period = periodicity(expo, symbol)
        else:
            period = _periodicity(f.args, symbol)
    elif f.is_Mul:
        coeff, g = f.as_independent(symbol, as_Add=False)
        if isinstance(g, TrigonometricFunction) or coeff is not S.One:
            period = periodicity(g, symbol)
        else:
            period = _periodicity(g.args, symbol)
    elif f.is_Add:
        k, g = f.as_independent(symbol)
        if k is not S.Zero:
            return periodicity(g, symbol)
        period = _periodicity(g.args, symbol)
    elif period is None:
        from sympy.solvers.decompogen import compogen
        g_s = decompogen(f, symbol)
        num_of_gs = len(g_s)
        if num_of_gs > 1:
            for index, g in enumerate(reversed(g_s)):
                start_index = num_of_gs - 1 - index
                g = compogen(g_s[start_index:], symbol)
                if g != orig_f and g != f:
                    period = periodicity(g, symbol)
                    if period is not None:
                        break
    if period is not None:
        if check:
            return _check(orig_f, period)
        return period
    return None