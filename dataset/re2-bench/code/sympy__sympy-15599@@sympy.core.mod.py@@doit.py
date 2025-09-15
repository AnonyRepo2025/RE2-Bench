from __future__ import print_function, division
from sympy.core.numbers import nan, Integer
from sympy.core.compatibility import integer_types
from .function import Function
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.exprtools import gcd_terms
from sympy.polys.polytools import gcd



class Mod(Function):

    @classmethod
    def eval(cls, p, q):
        from sympy.core.add import Add
        from sympy.core.mul import Mul
        from sympy.core.singleton import S
        from sympy.core.exprtools import gcd_terms
        from sympy.polys.polytools import gcd

        def doit(p, q):
            """Try to return p % q if both are numbers or +/-p is known
            to be less than or equal q.
            """

            if q == S.Zero:
                raise ZeroDivisionError("Modulo by zero")
            if p.is_infinite or q.is_infinite or p is nan or q is nan:
                return nan
            if p == S.Zero or p == q or p == -q or (p.is_integer and q == 1):
                return S.Zero

            if q.is_Number:
                if p.is_Number:
                    return p%q
                if q == 2:
                    if p.is_even:
                        return S.Zero
                    elif p.is_odd:
                        return S.One

            if hasattr(p, '_eval_Mod'):
                rv = getattr(p, '_eval_Mod')(q)
                if rv is not None:
                    return rv

            r = p/q
            try:
                d = int(r)
            except TypeError:
                pass
            else:
                if isinstance(d, integer_types):
                    rv = p - d*q
                    if (rv*q < 0) == True:
                        rv += q
                    return rv

            d = abs(p)
            for _ in range(2):
                d -= abs(q)
                if d.is_negative:
                    if q.is_positive:
                        if p.is_positive:
                            return d + q
                        elif p.is_negative:
                            return -d
                    elif q.is_negative:
                        if p.is_positive:
                            return d
                        elif p.is_negative:
                            return -d + q
                    break

        rv = doit(p, q)
        if rv is not None:
            return rv

        if isinstance(p, cls):
            qinner = p.args[1]
            if qinner % q == 0:
                return cls(p.args[0], q)
            elif (qinner*(q - qinner)).is_nonnegative:
                # |qinner| < |q| and have same sign
                return p
        elif isinstance(-p, cls):
            qinner = (-p).args[1]
            if qinner % q == 0:
                return cls(-(-p).args[0], q)
            elif (qinner*(q + qinner)).is_nonpositive:
                # |qinner| < |q| and have different sign
                return p
        elif isinstance(p, Add):
            both_l = non_mod_l, mod_l = [], []
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)
            if mod_l and all(inner.args[1] == q for inner in mod_l):
                net = Add(*non_mod_l) + Add(*[i.args[0] for i in mod_l])
                return cls(net, q)

        elif isinstance(p, Mul):
            both_l = non_mod_l, mod_l = [], []
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)

            if mod_l and all(inner.args[1] == q for inner in mod_l):
                non_mod_l = [cls(x, q) for x in non_mod_l]
                mod = []
                non_mod = []
                for j in non_mod_l:
                    if isinstance(j, cls):
                        mod.append(j.args[0])
                    else:
                        non_mod.append(j)
                prod_mod = Mul(*mod)
                prod_non_mod = Mul(*non_mod)
                prod_mod1 = Mul(*[i.args[0] for i in mod_l])
                net = prod_mod1*prod_mod
                return prod_non_mod*cls(net, q)

            if q.is_Integer and q is not S.One:
                _ = []
                for i in non_mod_l:
                    if i.is_Integer and (i % q is not S.Zero):
                        _.append(i%q)
                    else:
                        _.append(i)
                non_mod_l = _

            p = Mul(*(non_mod_l + mod_l))

        G = gcd(p, q)
        if G != 1:
            p, q = [
                gcd_terms(i/G, clear=False, fraction=False) for i in (p, q)]
        pwas, qwas = p, q

        if p.is_Add:
            args = []
            for i in p.args:
                a = cls(i, q)
                if a.count(cls) > i.count(cls):
                    args.append(i)
                else:
                    args.append(a)
            if args != list(p.args):
                p = Add(*args)

        else:
            cp, p = p.as_coeff_Mul()
            cq, q = q.as_coeff_Mul()
            ok = False
            if not cp.is_Rational or not cq.is_Rational:
                r = cp % cq
                if r == 0:
                    G *= cq
                    p *= int(cp/cq)
                    ok = True
            if not ok:
                p = cp*p
                q = cq*q

        if p.could_extract_minus_sign() and q.could_extract_minus_sign():
            G, p, q = [-i for i in (G, p, q)]

        rv = doit(p, q)
        if rv is not None:
            return rv*G

        if G.is_Float and G == 1:
            p *= G
            return cls(p, q, evaluate=False)
        elif G.is_Mul and G.args[0].is_Float and G.args[0] == 1:
            p = G.args[0]*p
            G = Mul._from_args(G.args[1:])
        return G*cls(p, q, evaluate=(p, q) != (pwas, qwas))