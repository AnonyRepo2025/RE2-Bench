from __future__ import print_function, division
from collections import defaultdict
from sympy import SYMPY_DEBUG
from sympy.core.evaluate import global_evaluate
from sympy.core.compatibility import iterable, ordered, default_sort_key
from sympy.core import expand_power_base, sympify, Add, S, Mul, Derivative, Pow, symbols, expand_mul
from sympy.core.numbers import Rational
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.mul import _keep_coeff, _unevaluated_Mul
from sympy.core.function import _mexpand
from sympy.core.add import _unevaluated_Add
from sympy.functions import exp, sqrt, log
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.simplify import signsimp
from sympy.simplify.simplify import nsimplify
from sympy.simplify.powsimp import powsimp, powdenest

expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand

def radsimp(expr, symbolic=True, max_terms=4):
    """
    Rationalize the denominator by removing square roots.

    Note: the expression returned from radsimp must be used with caution
    since if the denominator contains symbols, it will be possible to make
    substitutions that violate the assumptions of the simplification process:
    that for a denominator matching a + b*sqrt(c), a != +/-b*sqrt(c). (If
    there are no symbols, this assumptions is made valid by collecting terms
    of sqrt(c) so the match variable ``a`` does not contain ``sqrt(c)``.) If
    you do not want the simplification to occur for symbolic denominators, set
    ``symbolic`` to False.

    If there are more than ``max_terms`` radical terms then the expression is
    returned unchanged.

    Examples
    ========

    >>> from sympy import radsimp, sqrt, Symbol, denom, pprint, I
    >>> from sympy import factor_terms, fraction, signsimp
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b, c

    >>> radsimp(1/(2 + sqrt(2)))
    (-sqrt(2) + 2)/2
    >>> x,y = map(Symbol, 'xy')
    >>> e = ((2 + 2*sqrt(2))*x + (2 + sqrt(8))*y)/(2 + sqrt(2))
    >>> radsimp(e)
    sqrt(2)*(x + y)

    No simplification beyond removal of the gcd is done. One might
    want to polish the result a little, however, by collecting
    square root terms:

    >>> r2 = sqrt(2)
    >>> r5 = sqrt(5)
    >>> ans = radsimp(1/(y*r2 + x*r2 + a*r5 + b*r5)); pprint(ans)
        ___       ___       ___       ___
      \/ 5 *a + \/ 5 *b - \/ 2 *x - \/ 2 *y
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    >>> n, d = fraction(ans)
    >>> pprint(factor_terms(signsimp(collect_sqrt(n))/d, radical=True))
            ___             ___
          \/ 5 *(a + b) - \/ 2 *(x + y)
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    If radicals in the denominator cannot be removed or there is no denominator,
    the original expression will be returned.

    >>> radsimp(sqrt(2)*x + sqrt(2))
    sqrt(2)*x + sqrt(2)

    Results with symbols will not always be valid for all substitutions:

    >>> eq = 1/(a + b*sqrt(c))
    >>> eq.subs(a, b*sqrt(c))
    1/(2*b*sqrt(c))
    >>> radsimp(eq).subs(a, b*sqrt(c))
    nan

    If symbolic=False, symbolic denominators will not be transformed (but
    numeric denominators will still be processed):

    >>> radsimp(eq, symbolic=False)
    1/(a + b*sqrt(c))

    """
    from sympy.simplify.simplify import signsimp

    syms = symbols("a:d A:D")
    def _num(rterms):
        # return the multiplier that will simplify the expression described
        # by rterms [(sqrt arg, coeff), ... ]
        a, b, c, d, A, B, C, D = syms
        if len(rterms) == 2:
            reps = dict(list(zip([A, a, B, b], [j for i in rterms for j in i])))
            return (
            sqrt(A)*a - sqrt(B)*b).xreplace(reps)
        if len(rterms) == 3:
            reps = dict(list(zip([A, a, B, b, C, c], [j for i in rterms for j in i])))
            return (
            (sqrt(A)*a + sqrt(B)*b - sqrt(C)*c)*(2*sqrt(A)*sqrt(B)*a*b - A*a**2 -
            B*b**2 + C*c**2)).xreplace(reps)
        elif len(rterms) == 4:
            reps = dict(list(zip([A, a, B, b, C, c, D, d], [j for i in rterms for j in i])))
            return ((sqrt(A)*a + sqrt(B)*b - sqrt(C)*c - sqrt(D)*d)*(2*sqrt(A)*sqrt(B)*a*b
                - A*a**2 - B*b**2 - 2*sqrt(C)*sqrt(D)*c*d + C*c**2 +
                D*d**2)*(-8*sqrt(A)*sqrt(B)*sqrt(C)*sqrt(D)*a*b*c*d + A**2*a**4 -
                2*A*B*a**2*b**2 - 2*A*C*a**2*c**2 - 2*A*D*a**2*d**2 + B**2*b**4 -
                2*B*C*b**2*c**2 - 2*B*D*b**2*d**2 + C**2*c**4 - 2*C*D*c**2*d**2 +
                D**2*d**4)).xreplace(reps)
        elif len(rterms) == 1:
            return sqrt(rterms[0][0])
        else:
            raise NotImplementedError

    def ispow2(d, log2=False):
        if not d.is_Pow:
            return False
        e = d.exp
        if e.is_Rational and e.q == 2 or symbolic and denom(e) == 2:
            return True
        if log2:
            q = 1
            if e.is_Rational:
                q = e.q
            elif symbolic:
                d = denom(e)
                if d.is_Integer:
                    q = d
            if q != 1 and log(q, 2).is_Integer:
                return True
        return False

    def handle(expr):
        # Handle first reduces to the case
        # expr = 1/d, where d is an add, or d is base**p/2.
        # We do this by recursively calling handle on each piece.
        from sympy.simplify.simplify import nsimplify

        n, d = fraction(expr)

        if expr.is_Atom or (d.is_Atom and n.is_Atom):
            return expr
        elif not n.is_Atom:
            n = n.func(*[handle(a) for a in n.args])
            return _unevaluated_Mul(n, handle(1/d))
        elif n is not S.One:
            return _unevaluated_Mul(n, handle(1/d))
        elif d.is_Mul:
            return _unevaluated_Mul(*[handle(1/d) for d in d.args])

        # By this step, expr is 1/d, and d is not a mul.
        if not symbolic and d.free_symbols:
            return expr

        if ispow2(d):
            d2 = sqrtdenest(sqrt(d.base))**numer(d.exp)
            if d2 != d:
                return handle(1/d2)
        elif d.is_Pow and (d.exp.is_integer or d.base.is_positive):
            # (1/d**i) = (1/d)**i
            return handle(1/d.base)**d.exp

        if not (d.is_Add or ispow2(d)):
            return 1/d.func(*[handle(a) for a in d.args])

        # handle 1/d treating d as an Add (though it may not be)

        keep = True  # keep changes that are made

        # flatten it and collect radicals after checking for special
        # conditions
        d = _mexpand(d)

        # did it change?
        if d.is_Atom:
            return 1/d

        # is it a number that might be handled easily?
        if d.is_number:
            _d = nsimplify(d)
            if _d.is_Number and _d.equals(d):
                return 1/_d

        while True:
            # collect similar terms
            collected = defaultdict(list)
            for m in Add.make_args(d):  # d might have become non-Add
                p2 = []
                other = []
                for i in Mul.make_args(m):
                    if ispow2(i, log2=True):
                        p2.append(i.base if i.exp is S.Half else i.base**(2*i.exp))
                    elif i is S.ImaginaryUnit:
                        p2.append(S.NegativeOne)
                    else:
                        other.append(i)
                collected[tuple(ordered(p2))].append(Mul(*other))
            rterms = list(ordered(list(collected.items())))
            rterms = [(Mul(*i), Add(*j)) for i, j in rterms]
            nrad = len(rterms) - (1 if rterms[0][0] is S.One else 0)
            if nrad < 1:
                break
            elif nrad > max_terms:
                # there may have been invalid operations leading to this point
                # so don't keep changes, e.g. this expression is troublesome
                # in collecting terms so as not to raise the issue of 2834:
                # r = sqrt(sqrt(5) + 5)
                # eq = 1/(sqrt(5)*r + 2*sqrt(5)*sqrt(-sqrt(5) + 5) + 5*r)
                keep = False
                break
            if len(rterms) > 4:
                # in general, only 4 terms can be removed with repeated squaring
                # but other considerations can guide selection of radical terms
                # so that radicals are removed
                if all([x.is_Integer and (y**2).is_Rational for x, y in rterms]):
                    nd, d = rad_rationalize(S.One, Add._from_args(
                        [sqrt(x)*y for x, y in rterms]))
                    n *= nd
                else:
                    # is there anything else that might be attempted?
                    keep = False
                break
            from sympy.simplify.powsimp import powsimp, powdenest

            num = powsimp(_num(rterms))
            n *= num
            d *= num
            d = powdenest(_mexpand(d), force=symbolic)
            if d.is_Atom:
                break

        if not keep:
            return expr
        return _unevaluated_Mul(n, 1/d)

    coeff, expr = expr.as_coeff_Add()
    expr = expr.normal()
    old = fraction(expr)
    n, d = fraction(handle(expr))
    if old != (n, d):
        if not d.is_Atom:
            was = (n, d)
            n = signsimp(n, evaluate=False)
            d = signsimp(d, evaluate=False)
            u = Factors(_unevaluated_Mul(n, 1/d))
            u = _unevaluated_Mul(*[k**v for k, v in u.factors.items()])
            n, d = fraction(u)
            if old == (n, d):
                n, d = was
        n = expand_mul(n)
        if d.is_Number or d.is_Add:
            n2, d2 = fraction(gcd_terms(_unevaluated_Mul(n, 1/d)))
            if d2.is_Number or (d2.count_ops() <= d.count_ops()):
                n, d = [signsimp(i) for i in (n2, d2)]
                if n.is_Mul and n.args[0].is_Number:
                    n = n.func(*n.args)

    return coeff + _unevaluated_Mul(n, 1/d)
