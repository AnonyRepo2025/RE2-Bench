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
from sympy.tensor.tensor import TensExpr
from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy
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
    def _eval_is_zero(self):
        if self.is_commutative is False:
            # issue 10528: there is no way to know if a nc symbol
            # is zero or not
            return
        nz = []
        z = 0
        im_or_z = False
        im = False
        for a in self.args:
            if a.is_real:
                if a.is_zero:
                    z += 1
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im = True
            elif (S.ImaginaryUnit*a).is_real:
                im_or_z = True
            else:
                return
        if z == len(self.args):
            return True
        if len(nz) == 0 or len(nz) == len(self.args):
            return None
        b = self.func(*nz)
        if b.is_zero:
            if not im_or_z and not im:
                return True
            if im and not im_or_z:
                return False
        if b.is_zero is False:
            return False