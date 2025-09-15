from collections import defaultdict
from functools import cmp_to_key, reduce
import operator
from .sympify import sympify
from .basic import Basic
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .numbers import Rational
from .power import Pow
from .add import Add, _addsort, _unevaluated_Add
from sympy.simplify.simplify import bottom_up
from sympy.calculus.util import AccumBounds
from sympy.matrices.expressions import MatrixExpr
from sympy.core.numbers import I, Float
from sympy import Abs, expand_mul, im, re
from sympy import fraction
from sympy import Integer, factorial, prod, Sum, Max
from sympy.ntheory.multinomial import multinomial_coefficients_iterator
from .function import AppliedUndef
from .symbol import Symbol, symbols, Dummy
from sympy.series.limitseq import difference_delta as dd
from sympy.simplify.simplify import signsimp
from .symbol import Dummy
from sympy import trailing
from sympy import trailing, fraction
from sympy import trailing, fraction
from sympy.functions.elementary.complexes import sign
from sympy.ntheory.factor_ import multiplicity
from sympy.simplify.powsimp import powdenest
from sympy.simplify.radsimp import fraction
from sympy import degree, Mul, Order, ceiling, powsimp, PolynomialError, PoleError
from itertools import product
from sympy.core.power import integer_nthroot
from sympy.functions.elementary.complexes import sign
from sympy.utilities.iterables import sift
from sympy import exp

_args_sortkey = cmp_to_key(Basic.compare)
mul = AssocOpDispatcher('mul')

class Mul(Expr, AssocOp):
    __slots__ = ()
    is_Mul = True
    _args_type = Expr
    _kind_dispatcher = KindDispatcher("Mul_kind_dispatcher", commutative=True)
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)
    def _eval_is_rational(self):
        r = _fuzzy_group((a.is_rational for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            return self.is_zero
    def _eval_is_integer(self):
        from sympy import trailing
        is_rational = self._eval_is_rational()
        if is_rational is False:
            return False

        numerators = []
        denominators = []
        for a in self.args:
            if a.is_integer:
                if abs(a) is not S.One:
                    numerators.append(a)
            elif a.is_Rational:
                n, d = a.as_numer_denom()
                if abs(n) is not S.One:
                    numerators.append(n)
                if d is not S.One:
                    denominators.append(d)
            elif a.is_Pow:
                b, e = a.as_base_exp()
                if not b.is_integer or not e.is_integer: return
                if e.is_negative:
                    denominators.append(2 if a is S.Half else Pow(a, S.NegativeOne))
                else:
                    # for integer b and positive integer e: a = b**e would be integer
                    assert not e.is_positive
                    # for self being rational and e equal to zero: a = b**e would be 1
                    assert not e.is_zero
                    return # sign of e unknown -> self.is_integer cannot be decided
            else:
                return

        if not denominators:
            return True

        allodd = lambda x: all(i.is_odd for i in x)
        alleven = lambda x: all(i.is_even for i in x)
        anyeven = lambda x: any(i.is_even for i in x)

        if allodd(numerators) and anyeven(denominators):
            return False
        elif anyeven(numerators) and denominators == [2]:
            return True
        elif alleven(numerators) and allodd(denominators
                ) and (Mul(*denominators, evaluate=False) - 1
                ).is_positive:
            return False
        if len(denominators) == 1:
            d = denominators[0]
            if d.is_Integer and d.is_even:
                # if minimal power of 2 in num vs den is not
                # negative then we have an integer
                if (Add(*[i.as_base_exp()[1] for i in
                        numerators if i.is_even]) - trailing(d.p)
                        ).is_nonnegative:
                    return True
        if len(numerators) == 1:
            n = numerators[0]
            if n.is_Integer and n.is_even:
                # if minimal power of 2 in den vs num is positive
                # then we have have a non-integer
                if (Add(*[i.as_base_exp()[1] for i in
                        denominators if i.is_even]) - trailing(n.p)
                        ).is_positive:
                    return False