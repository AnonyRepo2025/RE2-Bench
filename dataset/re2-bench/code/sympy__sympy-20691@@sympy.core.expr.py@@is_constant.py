from typing import Tuple as tTuple
from collections.abc import Iterable
from functools import reduce
from .sympify import sympify, _sympify, SympifyError
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .cache import cacheit
from .compatibility import as_int, default_sort_key
from .kind import NumberKind
from sympy.utilities.misc import func_name
from mpmath.libmp import mpf_log, prec_to_dps
from collections import defaultdict
from .mul import Mul
from .add import Add
from .power import Pow
from .function import Function, _derivative_dispatch
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Integer, Rational
from math import log10, ceil, log
from sympy import Float
from sympy import Abs
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.integers import floor
from sympy import Dummy
from .relational import GreaterThan
from .relational import LessThan
from .relational import StrictGreaterThan
from .relational import StrictLessThan
from sympy import Float
from sympy.simplify.simplify import nsimplify, simplify
from sympy.solvers.solvers import solve
from sympy.polys.polyerrors import NotAlgebraic
from sympy.polys.numberfields import minimal_polynomial
from sympy.core.numbers import pure_complex
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.polyerrors import NotAlgebraic
from sympy.series import limit, Limit
from sympy.solvers.solveset import solveset
from sympy.sets.sets import Interval
from sympy.functions.elementary.exponential import log
from sympy.calculus.util import AccumBounds
from sympy.functions.elementary.complexes import conjugate as c
from sympy import log
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.complexes import transpose
from sympy.functions.elementary.complexes import conjugate, transpose
from sympy.functions.elementary.complexes import adjoint
from sympy.polys.orderings import monomial_key
from sympy.polys import Poly, PolynomialError
from .numbers import Number, NumberSymbol
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
from .mul import _unevaluated_Mul
from .add import _unevaluated_Add
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy import exp_polar, pi, I, ceiling, Add
from sympy import collect, Dummy, Order, Rational, Symbol, ceiling
from sympy import Order, Dummy, PoleError
from sympy.functions import exp, log
from sympy.series.gruntz import mrv, rewrite
from sympy import Dummy, factorial
from sympy.utilities.misc import filldedent
from sympy.series.limits import limit
from sympy import Dummy, log, Piecewise, piecewise_fold
from sympy.series.gruntz import calculate_series
from sympy import powsimp
from sympy import collect
from sympy import Dummy, log
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.simplify.radsimp import fraction
from sympy.integrals import integrate
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
from sympy.simplify import gammasimp
from sympy.polys import factor
from sympy.polys import cancel
from sympy.polys.polytools import invert
from sympy.core.numbers import mod_inverse
from sympy.core.numbers import Float
from sympy.matrices.expressions.matexpr import _LeftRightArgs
from sympy import Piecewise, Eq
from sympy import Tuple, MatrixExpr
from sympy.matrices.common import MatrixCommon
from sympy.calculus.util import AccumBounds
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.testing.randtest import random_complex_number
from mpmath.libmp.libintmath import giant_steps
from sympy.core.evalf import DEFAULT_MAXPREC as target
from sympy.solvers.solvers import denoms
from sympy.utilities.misc import filldedent
from sympy.core.numbers import mod_inverse



class Expr(Basic, EvalfMixin):
    __slots__ = ()  # type: tTuple[str, ...]
    is_scalar = True  # self derivative is 1
    _op_priority = 10.0
    __round__ = round
    def __hash__(self) -> int:
        # hash cannot be cached using cache_it because infinite recurrence
        # occurs as hash is needed for setting cache dictionary keys
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h
    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.
        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args
    def __eq__(self, other):
        try:
            other = _sympify(other)
            if not isinstance(other, Expr):
                return False
        except (SympifyError, SyntaxError):
            return False
        # check for pure number expr
        if  not (self.is_Number and other.is_Number) and (
                type(self) != type(other)):
            return False
        a, b = self._hashable_content(), other._hashable_content()
        if a != b:
            return False
        # check number *in* an expression
        for a, b in zip(a, b):
            if not isinstance(a, Expr):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True
    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return Mul(self, other)
    @property
    def is_number(self):
        """Returns True if ``self`` has no free symbols and no
        undefined functions (AppliedUndef, to be precise). It will be
        faster than ``if not self.free_symbols``, however, since
        ``is_number`` will fail as soon as it hits a free symbol
        or undefined function.

        Examples
        ========

        >>> from sympy import Integral, cos, sin, pi
        >>> from sympy.core.function import Function
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> x.is_number
        False
        >>> f(1).is_number
        False
        >>> (2*x).is_number
        False
        >>> (2 + Integral(2, x)).is_number
        False
        >>> (2 + Integral(2, (x, 1, 2))).is_number
        True

        Not all numbers are Numbers in the SymPy sense:

        >>> pi.is_number, pi.is_Number
        (True, False)

        If something is a number it should evaluate to a number with
        real and imaginary parts that are Numbers; the result may not
        be comparable, however, since the real and/or imaginary part
        of the result may not have precision.

        >>> cos(1).is_number and cos(1).is_comparable
        True

        >>> z = cos(1)**2 + sin(1)**2 - 1
        >>> z.is_number
        True
        >>> z.is_comparable
        False

        See Also
        ========

        sympy.core.basic.Basic.is_comparable
        """
        return all(obj.is_number for obj in self.args)
    def is_constant(self, *wrt, **flags):
        """Return True if self is constant, False if not, or None if
        the constancy could not be determined conclusively.

        Explanation
        ===========

        If an expression has no free symbols then it is a constant. If
        there are free symbols it is possible that the expression is a
        constant, perhaps (but not necessarily) zero. To test such
        expressions, a few strategies are tried:

        1) numerical evaluation at two random points. If two such evaluations
        give two different values and the values have a precision greater than
        1 then self is not constant. If the evaluations agree or could not be
        obtained with any precision, no decision is made. The numerical testing
        is done only if ``wrt`` is different than the free symbols.

        2) differentiation with respect to variables in 'wrt' (or all free
        symbols if omitted) to see if the expression is constant or not. This
        will not always lead to an expression that is zero even though an
        expression is constant (see added test in test_expr.py). If
        all derivatives are zero then self is constant with respect to the
        given symbols.

        3) finding out zeros of denominator expression with free_symbols.
        It won't be constant if there are zeros. It gives more negative
        answers for expression that are not constant.

        If neither evaluation nor differentiation can prove the expression is
        constant, None is returned unless two numerical values happened to be
        the same and the flag ``failing_number`` is True -- in that case the
        numerical value will be returned.

        If flag simplify=False is passed, self will not be simplified;
        the default is True since self should be simplified before testing.

        Examples
        ========

        >>> from sympy import cos, sin, Sum, S, pi
        >>> from sympy.abc import a, n, x, y
        >>> x.is_constant()
        False
        >>> S(2).is_constant()
        True
        >>> Sum(x, (x, 1, 10)).is_constant()
        True
        >>> Sum(x, (x, 1, n)).is_constant()
        False
        >>> Sum(x, (x, 1, n)).is_constant(y)
        True
        >>> Sum(x, (x, 1, n)).is_constant(n)
        False
        >>> Sum(x, (x, 1, n)).is_constant(x)
        True
        >>> eq = a*cos(x)**2 + a*sin(x)**2 - a
        >>> eq.is_constant()
        True
        >>> eq.subs({x: pi, a: 2}) == eq.subs({x: pi, a: 3}) == 0
        True

        >>> (0**x).is_constant()
        False
        >>> x.is_constant()
        False
        >>> (x**x).is_constant()
        False
        >>> one = cos(x)**2 + sin(x)**2
        >>> one.is_constant()
        True
        >>> ((one - 1)**(x + 1)).is_constant() in (True, False) # could be 0 or 1
        True
        """

        def check_denominator_zeros(expression):
            from sympy.solvers.solvers import denoms

            retNone = False
            for den in denoms(expression):
                z = den.is_zero
                if z is True:
                    return True
                if z is None:
                    retNone = True
            if retNone:
                return None
            return False

        simplify = flags.get('simplify', True)

        if self.is_number:
            return True
        free = self.free_symbols
        if not free:
            return True  # assume f(1) is some constant

        # if we are only interested in some symbols and they are not in the
        # free symbols then this expression is constant wrt those symbols
        wrt = set(wrt)
        if wrt and not wrt & free:
            return True
        wrt = wrt or free

        # simplify unless this has already been done
        expr = self
        if simplify:
            expr = expr.simplify()

        # is_zero should be a quick assumptions check; it can be wrong for
        # numbers (see test_is_not_constant test), giving False when it
        # shouldn't, but hopefully it will never give True unless it is sure.
        if expr.is_zero:
            return True

        # Don't attempt subsitution or differentiation with non-number symbols
        wrt_number = {sym for sym in wrt if sym.kind is NumberKind}

        # try numerical evaluation to see if we get two different values
        failing_number = None
        if wrt_number == free:
            # try 0 (for a) and 1 (for b)
            try:
                a = expr.subs(list(zip(free, [0]*len(free))),
                    simultaneous=True)
                if a is S.NaN:
                    # evaluation may succeed when substitution fails
                    a = expr._random(None, 0, 0, 0, 0)
            except ZeroDivisionError:
                a = None
            if a is not None and a is not S.NaN:
                try:
                    b = expr.subs(list(zip(free, [1]*len(free))),
                        simultaneous=True)
                    if b is S.NaN:
                        # evaluation may succeed when substitution fails
                        b = expr._random(None, 1, 0, 1, 0)
                except ZeroDivisionError:
                    b = None
                if b is not None and b is not S.NaN and b.equals(a) is False:
                    return False
                # try random real
                b = expr._random(None, -1, 0, 1, 0)
                if b is not None and b is not S.NaN and b.equals(a) is False:
                    return False
                # try random complex
                b = expr._random()
                if b is not None and b is not S.NaN:
                    if b.equals(a) is False:
                        return False
                    failing_number = a if a.is_number else b

        # now we will test each wrt symbol (or all free symbols) to see if the
        # expression depends on them or not using differentiation. This is
        # not sufficient for all expressions, however, so we don't return
        # False if we get a derivative other than 0 with free symbols.
        for w in wrt_number:
            deriv = expr.diff(w)
            if simplify:
                deriv = deriv.simplify()
            if deriv != 0:
                if not (pure_complex(deriv, or_real=True)):
                    if flags.get('failing_number', False):
                        return failing_number
                    elif deriv.free_symbols:
                        # dead line provided _random returns None in such cases
                        return None
                return False
        cd = check_denominator_zeros(self)
        if cd is True:
            return False
        elif cd is None:
            return None
        return True
    def _eval_is_positive(self):
        finite = self.is_finite
        if finite is False:
            return False
        extended_positive = self.is_extended_positive
        if finite is True:
            return extended_positive
        if extended_positive is False:
            return False
    def _eval_is_negative(self):
        finite = self.is_finite
        if finite is False:
            return False
        extended_negative = self.is_extended_negative
        if finite is True:
            return extended_negative
        if extended_negative is False:
            return False
    def _eval_is_extended_positive_negative(self, positive):
        from sympy.core.numbers import pure_complex
        from sympy.polys.numberfields import minimal_polynomial
        from sympy.polys.polyerrors import NotAlgebraic
        if self.is_number:
            if self.is_extended_real is False:
                return False

            # check to see that we can get a value
            try:
                n2 = self._eval_evalf(2)
            # XXX: This shouldn't be caught here
            # Catches ValueError: hypsum() failed to converge to the requested
            # 34 bits of accuracy
            except ValueError:
                return None
            if n2 is None:
                return None
            if getattr(n2, '_prec', 1) == 1:  # no significance
                return None
            if n2 is S.NaN:
                return None

            f = self.evalf(2)
            if f.is_Float:
                match = f, S.Zero
            else:
                match = pure_complex(f)
            if match is None:
                return False
            r, i = match
            if not (i.is_Number and r.is_Number):
                return False
            if r._prec != 1 and i._prec != 1:
                return bool(not i and ((r > 0) if positive else (r < 0)))
            elif r._prec == 1 and (not i or i._prec == 1) and \
                    self.is_algebraic and not self.has(Function):
                try:
                    if minimal_polynomial(self).is_Symbol:
                        return False
                except (NotAlgebraic, NotImplementedError):
                    pass
    def _eval_is_extended_positive(self):
        return self._eval_is_extended_positive_negative(positive=True)
    def _eval_is_extended_negative(self):
        return self._eval_is_extended_positive_negative(positive=False)
    def as_base_exp(self):
        # a -> b ** e
        return self, S.One