from __future__ import print_function, division
from collections import defaultdict
from functools import cmp_to_key
from .basic import Basic
from .compatibility import reduce, is_sequence, range
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp
from .cache import cacheit
from .numbers import ilcm, igcd
from .expr import Expr
from .mul import Mul, _keep_coeff, prod
from sympy.core.numbers import Rational
from sympy.calculus.util import AccumBounds
from sympy.matrices.expressions import MatrixExpr
from sympy import oo, I, expand_mul
from sympy.core.exprtools import _monotonic_sign
from sympy.core.exprtools import _monotonic_sign
from sympy.core.exprtools import _monotonic_sign
from sympy.core.exprtools import _monotonic_sign
from sympy import Order
from sympy import expand_mul, factor_terms
from sympy.core.compatibility import default_sort_key
from sympy.series.limitseq import difference_delta as dd
from sympy.core.numbers import I, Float
from sympy.core.evalf import pure_complex
from sympy.core.mul import _unevaluated_Mul
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_multinomial
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt

_args_sortkey = cmp_to_key(Basic.compare)

class Add(Expr, AssocOp):
    __slots__ = []
    is_Add = True
    _eval_is_real = lambda self: _fuzzy_group(
        (a.is_real for a in self.args), quick_exit=True)
    _eval_is_complex = lambda self: _fuzzy_group(
        (a.is_complex for a in self.args), quick_exit=True)
    _eval_is_antihermitian = lambda self: _fuzzy_group(
        (a.is_antihermitian for a in self.args), quick_exit=True)
    _eval_is_finite = lambda self: _fuzzy_group(
        (a.is_finite for a in self.args), quick_exit=True)
    _eval_is_hermitian = lambda self: _fuzzy_group(
        (a.is_hermitian for a in self.args), quick_exit=True)
    _eval_is_integer = lambda self: _fuzzy_group(
        (a.is_integer for a in self.args), quick_exit=True)
    _eval_is_rational = lambda self: _fuzzy_group(
        (a.is_rational for a in self.args), quick_exit=True)
    _eval_is_algebraic = lambda self: _fuzzy_group(
        (a.is_algebraic for a in self.args), quick_exit=True)
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)
    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation. """
        coeff, args = self.args[0], self.args[1:]

        if coeff.is_Number and not rational or coeff.is_Rational:
            return coeff, self._new_rawargs(*args)
        return S.Zero, self
    def _eval_power(self, e):
        if e.is_Rational and self.is_number:
            from sympy.core.evalf import pure_complex
            from sympy.core.mul import _unevaluated_Mul
            from sympy.core.exprtools import factor_terms
            from sympy.core.function import expand_multinomial
            from sympy.functions.elementary.complexes import sign
            from sympy.functions.elementary.miscellaneous import sqrt
            ri = pure_complex(self)
            if ri:
                r, i = ri
                if e.q == 2:
                    D = sqrt(r**2 + i**2)
                    if D.is_Rational:
                        # (r, i, D) is a Pythagorean triple
                        root = sqrt(factor_terms((D - r)/2))**e.p
                        return root*expand_multinomial((
                            # principle value
                            (D + r)/abs(i) + sign(i)*S.ImaginaryUnit)**e.p)
                elif e == -1:
                    return _unevaluated_Mul(
                        r - i*S.ImaginaryUnit,
                        1/(r**2 + i**2))