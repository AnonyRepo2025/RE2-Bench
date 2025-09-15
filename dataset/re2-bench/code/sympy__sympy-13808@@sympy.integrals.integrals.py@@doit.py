from __future__ import print_function, division
from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.compatibility import is_sequence, range
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import diff
from sympy.core.mul import Mul
from sympy.core.numbers import oo, pi
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.core.sympify import sympify
from sympy.integrals.manualintegrate import manualintegrate
from sympy.integrals.trigonometry import trigintegrate
from sympy.integrals.meijerint import meijerint_definite, meijerint_indefinite
from sympy.matrices import MatrixBase
from sympy.utilities.misc import filldedent
from sympy.polys import Poly, PolynomialError
from sympy.functions import Piecewise, sqrt, sign, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.series import limit
from sympy.series.order import Order
from sympy.series.formal import FormalPowerSeries
from sympy.geometry import Curve
from sympy.solvers.solvers import solve, posify
from sympy.integrals.deltafunctions import deltaintegrate
from sympy.integrals.singularityfunctions import singularityintegrate
from sympy.integrals.heurisch import heurisch, heurisch_wrapper
from sympy.integrals.rationaltools import ratint
from sympy.integrals.risch import risch_integrate
from sympy.concrete.summations import Sum
import sage.all as sage
from sympy.integrals.risch import NonElementaryIntegral
from sympy.integrals.meijerint import _debug
from sympy.integrals.meijerint import _debug



class Integral(AddWithLimits):
    __slots__ = ['is_commutative']
    def _eval_is_zero(self):
        # This is a very naive and quick test, not intended to do the integral to
        # answer whether it is zero or not, e.g. Integral(sin(x), (x, 0, 2*pi))
        # is zero but this routine should return None for that case. But, like
        # Mul, there are trivial situations for which the integral will be
        # zero so we check for those.
        if self.function.is_zero:
            return True
        got_none = False
        for l in self.limits:
            if len(l) == 3:
                z = (l[1] == l[2]) or (l[1] - l[2]).is_zero
                if z:
                    return True
                elif z is None:
                    got_none = True
        free = self.function.free_symbols
        for xab in self.limits:
            if len(xab) == 1:
                free.add(xab[0])
                continue
            if len(xab) == 2 and xab[0] not in free:
                if xab[1].is_zero:
                    return True
                elif xab[1].is_zero is None:
                    got_none = True
            # take integration symbol out of free since it will be replaced
            # with the free symbols in the limits
            free.discard(xab[0])
            # add in the new symbols
            for i in xab[1:]:
                free.update(i.free_symbols)
        if self.function.is_zero is False and got_none is False:
            return False
    def doit(self, **hints):
        """
        Perform the integration using any hints given.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i
        >>> Integral(x**i, (i, 1, 3)).doit()
        Piecewise((x**3/log(x) - x/log(x),
            (x > 1) | ((x >= 0) & (x < 1))), (2, True))

        See Also
        ========

        sympy.integrals.trigonometry.trigintegrate
        sympy.integrals.risch.heurisch
        sympy.integrals.rationaltools.ratint
        as_sum : Approximate the integral using a sum
        """
        if not hints.get('integrals', True):
            return self

        deep = hints.get('deep', True)
        meijerg = hints.get('meijerg', None)
        conds = hints.get('conds', 'piecewise')
        risch = hints.get('risch', None)
        manual = hints.get('manual', None)
        eval_kwargs = dict(meijerg=meijerg, risch=risch, manual=manual,
            conds=conds)

        if conds not in ['separate', 'piecewise', 'none']:
            raise ValueError('conds must be one of "separate", "piecewise", '
                             '"none", got: %s' % conds)

        if risch and any(len(xab) > 1 for xab in self.limits):
            raise ValueError('risch=True is only allowed for indefinite integrals.')

        # check for the trivial zero
        if self.is_zero:
            return S.Zero

        # now compute and check the function
        function = self.function
        if deep:
            function = function.doit(**hints)
        if function.is_zero:
            return S.Zero

        # hacks to handle special cases
        if isinstance(function, MatrixBase):
            return function.applyfunc(
                lambda f: self.func(f, self.limits).doit(**hints))

        if isinstance(function, FormalPowerSeries):
            if len(self.limits) > 1:
                raise NotImplementedError
            xab = self.limits[0]
            if len(xab) > 1:
                return function.integrate(xab, **eval_kwargs)
            else:
                return function.integrate(xab[0], **eval_kwargs)

        # There is no trivial answer and special handling
        # is done so continue

        undone_limits = []
        # ulj = free symbols of any undone limits' upper and lower limits
        ulj = set()
        for xab in self.limits:
            # compute uli, the free symbols in the
            # Upper and Lower limits of limit I
            if len(xab) == 1:
                uli = set(xab[:1])
            elif len(xab) == 2:
                uli = xab[1].free_symbols
            elif len(xab) == 3:
                uli = xab[1].free_symbols.union(xab[2].free_symbols)
            # this integral can be done as long as there is no blocking
            # limit that has been undone. An undone limit is blocking if
            # it contains an integration variable that is in this limit's
            # upper or lower free symbols or vice versa
            if xab[0] in ulj or any(v[0] in uli for v in undone_limits):
                undone_limits.append(xab)
                ulj.update(uli)
                function = self.func(*([function] + [xab]))
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue

            if function.has(Abs, sign) and (
                (len(xab) < 3 and all(x.is_real for x in xab)) or
                (len(xab) == 3 and all(x.is_real and x.is_finite for
                 x in xab[1:]))):
                    # some improper integrals are better off with Abs
                    xr = Dummy("xr", real=True)
                    function = (function.xreplace({xab[0]: xr})
                        .rewrite(Piecewise).xreplace({xr: xab[0]}))
            elif function.has(Min, Max):
                function = function.rewrite(Piecewise)
            if function.has(Piecewise) and \
                not isinstance(function, Piecewise):
                    function = piecewise_fold(function)
            if isinstance(function, Piecewise):
                if len(xab) == 1:
                    antideriv = function._eval_integral(xab[0],
                        **eval_kwargs)
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
            else:
                # There are a number of tradeoffs in using the
                # Meijer G method. It can sometimes be a lot faster
                # than other methods, and sometimes slower. And
                # there are certain types of integrals for which it
                # is more likely to work than others. These
                # heuristics are incorporated in deciding what
                # integration methods to try, in what order. See the
                # integrate() docstring for details.
                def try_meijerg(function, xab):
                    ret = None
                    if len(xab) == 3 and meijerg is not False:
                        x, a, b = xab
                        try:
                            res = meijerint_definite(function, x, a, b)
                        except NotImplementedError:
                            from sympy.integrals.meijerint import _debug
                            _debug('NotImplementedError '
                                'from meijerint_definite')
                            res = None
                        if res is not None:
                            f, cond = res
                            if conds == 'piecewise':
                                ret = Piecewise(
                                    (f, cond),
                                    (self.func(
                                    function, (x, a, b)), True))
                            elif conds == 'separate':
                                if len(self.limits) != 1:
                                    raise ValueError(filldedent('''
                                        conds=separate not supported in
                                        multiple integrals'''))
                                ret = f, cond
                            else:
                                ret = f
                    return ret

                meijerg1 = meijerg
                if (meijerg is not False and
                        len(xab) == 3 and xab[1].is_real and xab[2].is_real
                        and not function.is_Poly and
                        (xab[1].has(oo, -oo) or xab[2].has(oo, -oo))):
                    ret = try_meijerg(function, xab)
                    if ret is not None:
                        function = ret
                        continue
                    meijerg1 = False
                # If the special meijerg code did not succeed in
                # finding a definite integral, then the code using
                # meijerint_indefinite will not either (it might
                # find an antiderivative, but the answer is likely
                # to be nonsensical). Thus if we are requested to
                # only use Meijer G-function methods, we give up at
                # this stage. Otherwise we just disable G-function
                # methods.
                if meijerg1 is False and meijerg is True:
                    antideriv = None
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
                    if antideriv is None and meijerg is True:
                        ret = try_meijerg(function, xab)
                        if ret is not None:
                            function = ret
                            continue

            if not isinstance(antideriv, Integral) and antideriv is not None:
                sym = xab[0]
                for atan_term in antideriv.atoms(atan):
                    atan_arg = atan_term.args[0]
                    # Checking `atan_arg` to be linear combination of `tan` or `cot`
                    for tan_part in atan_arg.atoms(tan):
                        x1 = Dummy('x1')
                        tan_exp1 = atan_arg.subs(tan_part, x1)
                        # The coefficient of `tan` should be constant
                        coeff = tan_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = tan_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a-pi/2)/pi)))
                    for cot_part in atan_arg.atoms(cot):
                        x1 = Dummy('x1')
                        cot_exp1 = atan_arg.subs(cot_part, x1)
                        # The coefficient of `cot` should be constant
                        coeff = cot_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = cot_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a)/pi)))

            if antideriv is None:
                undone_limits.append(xab)
                function = self.func(*([function] + [xab])).factor()
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            else:
                if len(xab) == 1:
                    function = antideriv
                else:
                    if len(xab) == 3:
                        x, a, b = xab
                    elif len(xab) == 2:
                        x, b = xab
                        a = None
                    else:
                        raise NotImplementedError

                    if deep:
                        if isinstance(a, Basic):
                            a = a.doit(**hints)
                        if isinstance(b, Basic):
                            b = b.doit(**hints)

                    if antideriv.is_Poly:
                        gens = list(antideriv.gens)
                        gens.remove(x)

                        antideriv = antideriv.as_expr()

                        function = antideriv._eval_interval(x, a, b)
                        function = Poly(function, *gens)
                    else:
                        def is_indef_int(g, x):
                            return (isinstance(g, Integral) and
                                    any(i == (x,) for i in g.limits))

                        def eval_factored(f, x, a, b):
                            # _eval_interval for integrals with
                            # (constant) factors
                            # a single indefinite integral is assumed
                            args = []
                            for g in Mul.make_args(f):
                                if is_indef_int(g, x):
                                    args.append(g._eval_interval(x, a, b))
                                else:
                                    args.append(g)
                            return Mul(*args)

                        integrals, others = [], []
                        for f in Add.make_args(antideriv):
                            if any(is_indef_int(g, x)
                                   for g in Mul.make_args(f)):
                                integrals.append(f)
                            else:
                                others.append(f)
                        uneval = Add(*[eval_factored(f, x, a, b)
                                       for f in integrals])
                        try:
                            evalued = Add(*others)._eval_interval(x, a, b)
                            function = uneval + evalued
                        except NotImplementedError:
                            # This can happen if _eval_interval depends in a
                            # complicated way on limits that cannot be computed
                            undone_limits.append(xab)
                            function = self.func(*([function] + [xab]))
                            factored_function = function.factor()
                            if not isinstance(factored_function, Integral):
                                function = factored_function
        return function
    def _eval_integral(self, f, x, meijerg=None, risch=None, manual=None,
                       conds='piecewise'):
        """
        Calculate the anti-derivative to the function f(x).

        The following algorithms are applied (roughly in this order):

        1. Simple heuristics (based on pattern matching and integral table):

           - most frequently used functions (e.g. polynomials, products of
             trig functions)

        2. Integration of rational functions:

           - A complete algorithm for integrating rational functions is
             implemented (the Lazard-Rioboo-Trager algorithm).  The algorithm
             also uses the partial fraction decomposition algorithm
             implemented in apart() as a preprocessor to make this process
             faster.  Note that the integral of a rational function is always
             elementary, but in general, it may include a RootSum.

        3. Full Risch algorithm:

           - The Risch algorithm is a complete decision
             procedure for integrating elementary functions, which means that
             given any elementary function, it will either compute an
             elementary antiderivative, or else prove that none exists.
             Currently, part of transcendental case is implemented, meaning
             elementary integrals containing exponentials, logarithms, and
             (soon!) trigonometric functions can be computed.  The algebraic
             case, e.g., functions containing roots, is much more difficult
             and is not implemented yet.

           - If the routine fails (because the integrand is not elementary, or
             because a case is not implemented yet), it continues on to the
             next algorithms below.  If the routine proves that the integrals
             is nonelementary, it still moves on to the algorithms below,
             because we might be able to find a closed-form solution in terms
             of special functions.  If risch=True, however, it will stop here.

        4. The Meijer G-Function algorithm:

           - This algorithm works by first rewriting the integrand in terms of
             very general Meijer G-Function (meijerg in SymPy), integrating
             it, and then rewriting the result back, if possible.  This
             algorithm is particularly powerful for definite integrals (which
             is actually part of a different method of Integral), since it can
             compute closed-form solutions of definite integrals even when no
             closed-form indefinite integral exists.  But it also is capable
             of computing many indefinite integrals as well.

           - Another advantage of this method is that it can use some results
             about the Meijer G-Function to give a result in terms of a
             Piecewise expression, which allows to express conditionally
             convergent integrals.

           - Setting meijerg=True will cause integrate() to use only this
             method.

        5. The "manual integration" algorithm:

           - This algorithm tries to mimic how a person would find an
             antiderivative by hand, for example by looking for a
             substitution or applying integration by parts. This algorithm
             does not handle as many integrands but can return results in a
             more familiar form.

           - Sometimes this algorithm can evaluate parts of an integral; in
             this case integrate() will try to evaluate the rest of the
             integrand using the other methods here.

           - Setting manual=True will cause integrate() to use only this
             method.

        6. The Heuristic Risch algorithm:

           - This is a heuristic version of the Risch algorithm, meaning that
             it is not deterministic.  This is tried as a last resort because
             it can be very slow.  It is still used because not enough of the
             full Risch algorithm is implemented, so that there are still some
             integrals that can only be computed using this method.  The goal
             is to implement enough of the Risch and Meijer G-function methods
             so that this can be deleted.

        """
        from sympy.integrals.deltafunctions import deltaintegrate
        from sympy.integrals.singularityfunctions import singularityintegrate
        from sympy.integrals.heurisch import heurisch, heurisch_wrapper
        from sympy.integrals.rationaltools import ratint
        from sympy.integrals.risch import risch_integrate

        if risch:
            try:
                return risch_integrate(f, x, conds=conds)
            except NotImplementedError:
                return None

        if manual:
            try:
                result = manualintegrate(f, x)
                if result is not None and result.func != Integral:
                    return result
            except (ValueError, PolynomialError):
                pass

        eval_kwargs = dict(meijerg=meijerg, risch=risch, manual=manual,
            conds=conds)

        # if it is a poly(x) then let the polynomial integrate itself (fast)
        #
        # It is important to make this check first, otherwise the other code
        # will return a sympy expression instead of a Polynomial.
        #
        # see Polynomial for details.
        if isinstance(f, Poly) and not meijerg:
            return f.integrate(x)

        # Piecewise antiderivatives need to call special integrate.
        if isinstance(f, Piecewise):
            return f.piecewise_integrate(x, **eval_kwargs)

        # let's cut it short if `f` does not depend on `x`; if
        # x is only a dummy, that will be handled below
        if not f.has(x):
            return f*x

        # try to convert to poly(x) and then integrate if successful (fast)
        poly = f.as_poly(x)
        if poly is not None and not meijerg:
            return poly.integrate().as_expr()

        if risch is not False:
            try:
                result, i = risch_integrate(f, x, separate_integral=True,
                    conds=conds)
            except NotImplementedError:
                pass
            else:
                if i:
                    # There was a nonelementary integral. Try integrating it.

                    # if no part of the NonElementaryIntegral is integrated by
                    # the Risch algorithm, then use the original function to
                    # integrate, instead of re-written one
                    if result == 0:
                        from sympy.integrals.risch import NonElementaryIntegral
                        return NonElementaryIntegral(f, x).doit(risch=False)
                    else:
                        return result + i.doit(risch=False)
                else:
                    return result

        # since Integral(f=g1+g2+...) == Integral(g1) + Integral(g2) + ...
        # we are going to handle Add terms separately,
        # if `f` is not Add -- we only have one term

        # Note that in general, this is a bad idea, because Integral(g1) +
        # Integral(g2) might not be computable, even if Integral(g1 + g2) is.
        # For example, Integral(x**x + x**x*log(x)).  But many heuristics only
        # work term-wise.  So we compute this step last, after trying
        # risch_integrate.  We also try risch_integrate again in this loop,
        # because maybe the integral is a sum of an elementary part and a
        # nonelementary part (like erf(x) + exp(x)).  risch_integrate() is
        # quite fast, so this is acceptable.
        parts = []
        args = Add.make_args(f)
        for g in args:
            coeff, g = g.as_independent(x)

            # g(x) = const
            if g is S.One and not meijerg:
                parts.append(coeff*x)
                continue

            # g(x) = expr + O(x**n)
            order_term = g.getO()

            if order_term is not None:
                h = self._eval_integral(g.removeO(), x, **eval_kwargs)

                if h is not None:
                    h_order_expr = self._eval_integral(order_term.expr, x, **eval_kwargs)

                    if h_order_expr is not None:
                        h_order_term = order_term.func(
                            h_order_expr, *order_term.variables)
                        parts.append(coeff*(h + h_order_term))
                        continue

                # NOTE: if there is O(x**n) and we fail to integrate then
                # there is no point in trying other methods because they
                # will fail, too.
                return None

            #               c
            # g(x) = (a*x+b)
            if g.is_Pow and not g.exp.has(x) and not meijerg:
                a = Wild('a', exclude=[x])
                b = Wild('b', exclude=[x])

                M = g.base.match(a*x + b)

                if M is not None:
                    if g.exp == -1:
                        h = log(g.base)
                    elif conds != 'piecewise':
                        h = g.base**(g.exp + 1) / (g.exp + 1)
                    else:
                        h1 = log(g.base)
                        h2 = g.base**(g.exp + 1) / (g.exp + 1)
                        h = Piecewise((h2, Ne(g.exp, -1)), (h1, True))

                    parts.append(coeff * h / M[a])
                    continue

            #        poly(x)
            # g(x) = -------
            #        poly(x)
            if g.is_rational_function(x) and not meijerg:
                parts.append(coeff * ratint(g, x))
                continue

            if not meijerg:
                # g(x) = Mul(trig)
                h = trigintegrate(g, x, conds=conds)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # g(x) has at least a DiracDelta term
                h = deltaintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # g(x) has at least a Singularity Function term
                h = singularityintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # Try risch again.
                if risch is not False:
                    try:
                        h, i = risch_integrate(g, x,
                            separate_integral=True, conds=conds)
                    except NotImplementedError:
                        h = None
                    else:
                        if i:
                            h = h + i.doit(risch=False)

                        parts.append(coeff*h)
                        continue

                # fall back to heurisch
                try:
                    if conds == 'piecewise':
                        h = heurisch_wrapper(g, x, hints=[])
                    else:
                        h = heurisch(g, x, hints=[])
                except PolynomialError:
                    # XXX: this exception means there is a bug in the
                    # implementation of heuristic Risch integration
                    # algorithm.
                    h = None
            else:
                h = None

            if meijerg is not False and h is None:
                # rewrite using G functions
                try:
                    h = meijerint_indefinite(g, x)
                except NotImplementedError:
                    from sympy.integrals.meijerint import _debug
                    _debug('NotImplementedError from meijerint_definite')
                    res = None
                if h is not None:
                    parts.append(coeff * h)
                    continue

            if h is None and manual is not False:
                try:
                    result = manualintegrate(g, x)
                    if result is not None and not isinstance(result, Integral):
                        if result.has(Integral):
                            # try to have other algorithms do the integrals
                            # manualintegrate can't handle
                            result = result.func(*[
                                arg.doit(manual=False) if
                                arg.has(Integral) else arg
                                for arg in result.args
                            ]).expand(multinomial=False,
                                      log=False,
                                      power_exp=False,
                                      power_base=False)
                        if not result.has(Integral):
                            parts.append(coeff * result)
                            continue
                except (ValueError, PolynomialError):
                    # can't handle some SymPy expressions
                    pass

            # if we failed maybe it was because we had
            # a product that could have been expanded,
            # so let's try an expansion of the whole
            # thing before giving up; we don't try this
            # at the outset because there are things
            # that cannot be solved unless they are
            # NOT expanded e.g., x**x*(1+log(x)). There
            # should probably be a checker somewhere in this
            # routine to look for such cases and try to do
            # collection on the expressions if they are already
            # in an expanded form
            if not h and len(args) == 1:
                f = f.expand(mul=True, deep=False)
                if f.is_Add:
                    # Note: risch will be identical on the expanded
                    # expression, but maybe it will be able to pick out parts,
                    # like x*(exp(x) + erf(x)).
                    return self._eval_integral(f, x, **eval_kwargs)

            if h is not None:
                parts.append(coeff * h)
            else:
                return None

        return Add(*parts)