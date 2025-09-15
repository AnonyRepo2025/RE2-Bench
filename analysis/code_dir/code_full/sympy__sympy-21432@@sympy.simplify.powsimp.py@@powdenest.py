

from collections import defaultdict
from sympy.core.function import expand_log, count_ops
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.compatibility import ordered, default_sort_key, reduce
from sympy.core.numbers import Integer, Rational
from sympy.core.mul import prod, _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.simplify import posify
from sympy.simplify.simplify import logcombine
_y = Dummy('y')

def powdenest(eq, force=False, polar=False):
    from sympy.simplify.simplify import posify
    if force:

        def _denest(b, e):
            if not isinstance(b, (Pow, exp)):
                return (b.is_positive, Pow(b, e, evaluate=False))
            return _denest(b.base, b.exp * e)
        reps = []
        for p in eq.atoms(Pow, exp):
            if isinstance(p.base, (Pow, exp)):
                ok, dp = _denest(*p.args)
                if ok is not False:
                    reps.append((p, dp))
        if reps:
            eq = eq.subs(reps)
        eq, reps = posify(eq)
        return powdenest(eq, force=False, polar=polar).xreplace(reps)
    if polar:
        eq, rep = polarify(eq)
        return unpolarify(powdenest(unpolarify(eq, exponents_only=True)), rep)
    new = powsimp(sympify(eq))
    return new.xreplace(Transform(_denest_pow, filter=lambda m: m.is_Pow or isinstance(m, exp)))