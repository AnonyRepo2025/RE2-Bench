from __future__ import print_function, division
from sympy.core.sympify import _sympify, sympify
from sympy.core.basic import Basic, _aresame
from sympy.core.cache import cacheit
from sympy.core.compatibility import ordered, range
from sympy.core.logic import fuzzy_and
from sympy.core.evaluate import global_evaluate
from sympy import Order
from .expr import Add, Expr
from sympy import Mul
from .function import WildFunction
from .symbol import Wild
from .add import Add
from .mul import Mul
from .symbol import Symbol
from .function import AppliedUndef
from sympy.simplify.radsimp import collect

class AssocOp(Basic):
    __slots__ = ['is_commutative']

    @cacheit
    def __new__(cls, *args, **options):
        from sympy import Order
        args = list(map(_sympify, args))
        args = [a for a in args if a is not cls.identity]
        if not options.pop('evaluate', global_evaluate[0]):
            return cls._from_args(args)
        if len(args) == 0:
            return cls.identity
        if len(args) == 1:
            return args[0]
        c_part, nc_part, order_symbols = cls.flatten(args)
        is_commutative = not nc_part
        obj = cls._from_args(c_part + nc_part, is_commutative)
        obj = cls._exec_constructor_postprocessors(obj)
        if order_symbols is not None:
            return Order(obj, *order_symbols)
        return obj

    @classmethod
    def _from_args(cls, args, is_commutative=None):
        if len(args) == 0:
            return cls.identity
        elif len(args) == 1:
            return args[0]
        obj = super(AssocOp, cls).__new__(cls, *args)
        if is_commutative is None:
            is_commutative = fuzzy_and((a.is_commutative for a in args))
        obj.is_commutative = is_commutative
        return obj

    def _eval_evalf(self, prec):
        from .add import Add
        from .mul import Mul
        from .symbol import Symbol
        from .function import AppliedUndef
        if isinstance(self, (Mul, Add)):
            x, tail = self.as_independent(Symbol, AppliedUndef)
            if not (tail is self.identity or (isinstance(x, AssocOp) and x.is_Function) or (x is self.identity and isinstance(tail, AssocOp))):
                x = x._evalf(prec) if x is not self.identity else self.identity
                args = []
                tail_args = tuple(self.func.make_args(tail))
                for a in tail_args:
                    newa = a._eval_evalf(prec)
                    if newa is None:
                        args.append(a)
                    else:
                        args.append(newa)
                return self.func(x, *args)
        args = []
        for a in self.args:
            newa = a._eval_evalf(prec)
            if newa is None:
                args.append(a)
            else:
                args.append(newa)
        return self.func(*args)

    @classmethod
    def make_args(cls, expr):
        if isinstance(expr, cls):
            return expr.args
        else:
            return (sympify(expr),)