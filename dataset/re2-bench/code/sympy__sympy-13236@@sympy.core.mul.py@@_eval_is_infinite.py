from __future__ import print_function, division
from collections import defaultdict
from functools import cmp_to_key
import operator
from .sympify import sympify
from .basic import Basic
from .singleton import S
from .operations import AssocOp
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .compatibility import reduce, range
from .expr import Expr
from .evaluate import global_distribute
from .numbers import Rational
from .power import Pow
from .add import Add, _addsort, _unevaluated_Add
from sympy.simplify.simplify import bottom_up
from sympy.calculus.util import AccumBounds
from sympy.core.numbers import I, Float
from sympy import Abs, expand_mul, im, re
from sympy import fraction
from sympy.series.limitseq import difference_delta as dd
from sympy import Wild
from sympy.functions.elementary.complexes import sign
from sympy.ntheory.factor_ import multiplicity
from sympy.simplify.powsimp import powdenest
from sympy.simplify.radsimp import fraction
from sympy import Order, powsimp
from sympy.core.power import integer_nthroot
from sympy.functions.elementary.complexes import sign
from sympy import exp

_args_sortkey = cmp_to_key(Basic.compare)

class Mul(Expr, AssocOp):
    __slots__ = []
    is_Mul = True
    _eval_is_finite = lambda self: _fuzzy_group(
        a.is_finite for a in self.args)
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)
    _eval_is_complex = lambda self: _fuzzy_group(
        (a.is_complex for a in self.args), quick_exit=True)
    def _eval_is_infinite(self):
        if any(a.is_infinite for a in self.args):
            if any(a.is_zero for a in self.args):
                return S.NaN.is_infinite
            if any(a.is_zero is None for a in self.args):
                return None
            return True